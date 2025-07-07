from pathlib import Path
import time
from typing import NamedTuple
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optax
import optuna
import typer
from rich.console import Console
from rich.progress import track

from jaxrl.config import Config, EnvironmentConfig, LearnerConfig, LinearObsEncoderConfig, LoggerConfig, ModelConfig, OptimizerConfig, PPOConfig, TransformerActorCriticConfig, TransformerBlockConfig
from jaxrl.envs.environment import Environment
from jaxrl.envs.memory.n_back import NBackMemory
from jaxrl.envs.memory.return_2d import ReturnEnv
from jaxrl.envs.vmap_wrapper import VmapWrapper
from jaxrl.experiment import Experiment
from jaxrl.optimizer import create_optimizer
from jaxrl.transformer.network import TransformerActorCritic
from jaxrl.transformer.rollout import Rollout, RolloutState
from jaxrl.types import TimeStep
from jaxrl.checkpointer import Checkpointer


def create_env(env_config: EnvironmentConfig, length: int) -> Environment:
    match env_config.env_type:
        case 'nback':
            return NBackMemory(env_config.max_n, env_config.max_value, length)
        case 'return':
            return ReturnEnv()
        case _:
            raise ValueError(f'Unknown environment type: {env_config.type}')


class TrainingLogs(NamedTuple):
    n: jax.Array
    rewards: jax.Array
    value_loss: jax.Array
    actor_loss: jax.Array
    entropy_loss: jax.Array
    total_loss: jax.Array


def create_training_logs() -> TrainingLogs:
    return TrainingLogs(
        n=jnp.array(0.0),
        rewards=jnp.array(0.0),
        value_loss=jnp.array(0.0),
        actor_loss=jnp.array(0.0),
        entropy_loss=jnp.array(0.0),
        total_loss=jnp.array(0.0)
    )


def add_seq_dim(ts: TimeStep):
    return TimeStep(
        obs=rearrange(ts.obs, 'b ... -> b 1 ...'),
        time=rearrange(ts.time, 'b ... -> b 1 ...'),
        last_action=rearrange(ts.last_action, 'b ... -> b 1 ...'),
        last_reward=rearrange(ts.last_reward, 'b ... -> b 1 ...'),
        action_mask=rearrange(ts.action_mask, 'b ... -> b 1 ...') if ts.action_mask is not None else None,
    )

def evaluate(model: TransformerActorCritic, rollout: Rollout, rngs: nnx.Rngs, env: Environment, hypers: PPOConfig):
    reset_key = rngs.env()
    env_state, timestep = env.reset(reset_key)

    rollout_state = rollout.create_state()
    kv_cache = model.create_kv_cache(rollout.batch_size, rollout.trajectory_length, dtype=jnp.float32)

    def _step(i, x):
        rollout_state, rngs, env_state, timestep, kv_cache = x

        action_key = rngs.action()
        env_key = rngs.env()

        value, policy, kv_cache = model(add_seq_dim(timestep), kv_cache)

        action = policy.sample(seed=action_key)
        log_prob = policy.log_prob(action).squeeze(axis=-1)
        action = action.squeeze(axis=-1)
        value = value.squeeze(axis=-1)
        # jax.debug.breakpoint()

        env_state, next_timestep = env.step(env_state, action, env_key)

        rollout_state = rollout.store(
            rollout_state,
            step=i,
            timestep=timestep,
            next_timestep=next_timestep,
            log_prob=log_prob,
            value=value,
        )

        return rollout_state, rngs, env_state, next_timestep, kv_cache

    rollout_state, rngs, _, _, _ = nnx.fori_loop(
        0,
        rollout.trajectory_length,
        _step,
        init_val=(rollout_state, rngs, env_state, timestep, kv_cache)
    )

    rollout_state = rollout.calculate_advantage(rollout_state, discount=hypers.discount, gae_lambda=hypers.gae_lambda)

    return rollout_state, rngs

def ppo_loss(model: TransformerActorCritic, rollout: RolloutState, hypers: PPOConfig, logs: TrainingLogs):
    batch_obs = jax.lax.stop_gradient(rollout.obs)
    batch_target = jax.lax.stop_gradient(rollout.targets)
    batch_log_prob = jax.lax.stop_gradient(rollout.log_prob)
    batch_actions = jax.lax.stop_gradient(rollout.actions)
    batch_advantage = jax.lax.stop_gradient(rollout.advantages)
    batch_values = jax.lax.stop_gradient(rollout.values[..., :-1])
    batch_rewards = jax.lax.stop_gradient(rollout.rewards)

    batch_last_actions = jax.lax.stop_gradient(rollout.last_actions)
    batch_last_rewards = jax.lax.stop_gradient(rollout.last_rewards)

    # TODO: make this conditional
    # batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

    positions = jnp.arange(batch_obs.shape[1], dtype=jnp.int32)[None, :]

    values, policy, _ = model(TimeStep(
        obs=batch_obs,
        time=positions,
        last_action=batch_last_actions,
        last_reward=batch_last_rewards,
        action_mask=None
    ))
    log_probs = policy.log_prob(batch_actions)

    value_pred_clipped = batch_values + jnp.clip(values - batch_values, -hypers.vf_clip, hypers.vf_clip)

    value_losses = jnp.square(values - batch_target)
    value_losses_clipped = jnp.square(value_pred_clipped - batch_target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_probs - batch_log_prob)

    pg_loss1 = ratio * batch_advantage
    pg_loss2 = jnp.clip(ratio, 1.0 - hypers.vf_clip, 1.0 + hypers.vf_clip) * batch_advantage

    actor_loss = -jnp.minimum(pg_loss1, pg_loss2).mean()

    # Entropy regularization
    entropy = policy.entropy()
    entropy_loss = -entropy.mean()

    total_loss = hypers.vf_coef * value_loss + actor_loss + hypers.entropy_coef * entropy_loss


    # jax.debug.breakpoint()
    logs = logs._replace(
        n=logs.n + 1,
        rewards=logs.rewards + batch_rewards.mean(),
        value_loss=logs.value_loss + value_loss,
        actor_loss=logs.actor_loss + actor_loss,
        entropy_loss=logs.entropy_loss + entropy_loss,
        total_loss=logs.total_loss + total_loss
    )

    return total_loss, logs


def train(optimizer: nnx.Optimizer, rollout: Rollout, rngs: nnx.Rngs, env: Environment, hypers: PPOConfig):
    def _step(i, x):
        optimizer, rngs, logs = x

        rollout_state, rngs = evaluate(optimizer.model, rollout, rngs, env, hypers)

        grads, logs = nnx.grad(ppo_loss, has_aux=True)(optimizer.model, rollout_state, hypers, logs)
        optimizer.update(grads)
        return optimizer, rngs, logs

    logs = create_training_logs()
    optimizer, rngs, logs = nnx.fori_loop(0, hypers.minibatch_count, _step, init_val=(optimizer, rngs, logs))

    logs = TrainingLogs(
        n=jnp.array(1.0),
        rewards=logs.rewards / logs.n,
        value_loss=logs.value_loss / logs.n,
        actor_loss=logs.actor_loss / logs.n,
        entropy_loss=logs.entropy_loss / logs.n,
        total_loss=logs.total_loss / logs.n
    )

    return optimizer, rngs, logs

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def enjoy():
    experiment: Experiment = Experiment.load("test")
    max_steps = 128 #experiment.config.max_env_steps
    num_envs = 1

    env = NBackMemory(n=12, max_value=2, length=max_steps)
    env = VmapWrapper(env, num_envs)

    obs_spec = env.observation_spec
    action_spec = env.action_spec
    rngs = nnx.Rngs(default=experiment.default_seed)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        obs_spec.shape[0],
        action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs
    )
    optimizer = nnx.Optimizer(model=model, tx=create_optimizer(experiment.config.learner.optimizer, experiment.config.update_steps))

    with Checkpointer(experiment.checkpoints_dir) as checkpointer:
        optimizer = checkpointer.restore_latest(optimizer)

    model = optimizer.model

    kv_cache = model.create_kv_cache(num_envs, max_steps, dtype=jnp.float32)

    guess = []
    correct = []
    reward = []

    env_state, timestep = env.reset(rngs.env())
    for i in range(max_steps):
        action_key = rngs.action()
        env_key = rngs.env()
        value, policy, kv_cache = model(add_seq_dim(timestep), kv_cache)
        action = policy.sample(seed=action_key)
        action = action.squeeze(axis=-1)

        env_state, timestep = env.step(env_state, action, env_key)
        print(f"t: {timestep.obs}, action: {action.item()}, reward: {timestep.last_reward.item()}, value: {value.item()}")

        guess.append(action.item())
        correct.append(int(env_state.labels[..., i-1].item()))
        reward.append(timestep.last_reward.item())

    print("data: ", env_state.data[0])
    print("guess: ", jnp.array(guess))
    print("label: ", env_state.labels.astype(jnp.int32)[0])
    print("reward: ", jnp.array(reward, dtype=jnp.int32))
    print(sum(reward) / max_steps)


def train_run(
    experiment: Experiment,
    trial: optuna.Trial | None = None,
):
    max_steps = experiment.config.max_env_steps

    logger = experiment.create_logger()
    checkpointer = Checkpointer(experiment.checkpoints_dir)
    checkpoint_interval = 200

    env = create_env(experiment.config.environment, max_steps) #NBackMemory(n=12, max_value=2, length=max_steps)
    env = VmapWrapper(env, experiment.config.num_envs)
    batch_size = env.num_agents

    rngs = nnx.Rngs(default=experiment.default_seed)
    rollout = Rollout(batch_size, max_steps, env.observation_spec, env.action_spec)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs
    )

    optimizer = nnx.Optimizer(model=model, tx=create_optimizer(experiment.config.learner.optimizer, experiment.config.update_steps * experiment.config.learner.trainer.minibatch_count))
    trainer_hypers = experiment.config.learner.trainer
    jitted_train = nnx.jit(train, static_argnums=(1, 3, 4))

    env_steps_per_update = batch_size * max_steps

    logs = None
    for i in track(range(experiment.config.update_steps), description="Training", disable=False):
        start_time = time.time()
        optimizer, rngs, logs = jitted_train(optimizer, rollout, rngs, env, trainer_hypers)

        # this should be delayed n-1 for jax to use async dispatch
        logger.log(logs._asdict(), i)

        stop_time = time.time()

        sps = env_steps_per_update / (stop_time - start_time)
        print(int(sps * experiment.config.learner.trainer.minibatch_count))

        if trial:
            trial.report(logs.rewards.item(), i)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

        if i % checkpoint_interval == checkpoint_interval - 1:
            checkpointer.save(optimizer, i)

    logger.close()
    checkpointer.close()

    if logs:
        return logs.rewards.item()
    return -1.0


def objective(trial: optuna.Trial):
    config=Config(
        seed="random",
        num_envs=64,
        max_env_steps=128,
        update_steps=1000,
        learner=LearnerConfig(
            model=TransformerActorCriticConfig(
                obs_encoder=LinearObsEncoderConfig(),
                hidden_features=128,
                num_layers=3,
                activation="gelu",
                norm="layer_norm",
                transformer_block=TransformerBlockConfig(
                    num_heads=4,
                    ffn_size=128,
                    glu=False,
                    gtrxl_gate=False,
                )
            ),
            optimizer=OptimizerConfig(
                type="adamw",
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                weight_decay=0.0,
                eps=1e-8,
                beta1=0.9,
                beta2=0.999,
                max_norm=trial.suggest_float("max_norm", 0.1, 1.0),
            ),
            trainer=PPOConfig(
                trainer_type="ppo",
                minibatch_count=100,
                vf_coef=trial.suggest_float("vf_coef", 0.5, 2.0),
                entropy_coef=trial.suggest_float("entropy_coef", 0.0, 0.01),
                vf_clip=trial.suggest_float("vf_clip", 0.1, 0.3),
                discount=trial.suggest_float("discount", 0.9, 0.99),
                gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
            ),
        ),
        logger=LoggerConfig(),
    )

    return train_run(
        experiment=Experiment.from_config(config=config, unique_token=f"trial_{trial.number}"),
        trial=trial,
    )


@app.command()
def sweep():
    """Runs an Optuna sweep."""
    storage_name = "sqlite:///jaxrl_study.db"
    study_name = "jaxrl_study"

    import optunahub
    module = optunahub.load_module(package="samplers/auto_sampler")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True,
        sampler=module.AutoSampler(), #optuna.samplers.TPESampler(n_startup_trials=10),
        # pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=300)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    console = Console()
    console.print("Study statistics: ")
    console.print(f"  Number of finished trials: {len(study.trials)}")
    console.print(f"  Number of pruned trials: {len(pruned_trials)}")
    console.print(f"  Number of complete trials: {len(complete_trials)}")

    console.print("Best trial:")
    trial = study.best_trial

    console.print(f"  Value: {trial.value}")

    console.print("  Params: ")
    for key, value in trial.params.items():
        console.print(f"    {key}: {value}")


@app.command("train")
def train_cmd():
    experiment = Experiment.from_config_file(Path("./config/return.json"))

    train_run(experiment)

if __name__ == '__main__':
    app()
