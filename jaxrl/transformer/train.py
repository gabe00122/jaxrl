from typing import NamedTuple
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optuna
import typer
from rich.console import Console
from rich.progress import track

from jaxrl.config import Config, EnvironmentConfig, LearnerConfig, LinearObsEncoderConfig, LoggerConfig, ModelConfig, OptimizerConfig, PPOConfig, TransformerActorCriticConfig, TransformerBlockConfig
from jaxrl.envs.environment import Environment
from jaxrl.envs.memory.n_back import NBackMemory
from jaxrl.envs.vmap_wrapper import VmapWrapper
from jaxrl.experiment import Experiment
from jaxrl.optimizer import create_optimizer
from jaxrl.transformer.network import TransformerActorCritic
from jaxrl.transformer.rollout import Rollout
from jaxrl.types import TimeStep
from jaxrl.checkpointer import Checkpointer


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
        step_type=rearrange(ts.step_type, 'b ... -> b 1 ...'),
        action_mask=rearrange(ts.action_mask, 'b ... -> b 1 ...') if ts.action_mask is not None else None,
    )

def evaluate(model: TransformerActorCritic, rollout: Rollout, rngs: nnx.Rngs, env: Environment, hypers: PPOConfig):
    reset_key = rngs.env()
    env_state, timestep = env.reset(reset_key)
    kv_cache = model.create_kv_cache(rollout.batch_size, rollout.trajectory_length, dtype=jnp.float32)

    def _step(i, x):
        rollout, rngs, env_state, ts, kv_cache = x

        action_key = rngs.action()
        env_key = rngs.env()

        value, policy, kv_cache = model(add_seq_dim(ts), kv_cache)

        action = policy.sample(seed=action_key)
        log_prob = policy.log_prob(action).squeeze(axis=-1)
        action = action.squeeze(axis=-1)
        value = value.squeeze(axis=-1)
        # jax.debug.breakpoint()

        env_state, next_timestep = env.step(env_state, action, env_key)

        rollout.store(
            step=i,
            obs=ts.obs,
            action_mask=ts.action_mask,
            action=next_timestep.last_action,
            reward=next_timestep.last_reward,
            log_prob=log_prob,
            value=value
        )

        return rollout, rngs, env_state, next_timestep, kv_cache

    rollout, rngs, _, _, _ = nnx.fori_loop(
        0,
        rollout.trajectory_length,
        _step,
        init_val=(rollout, rngs, env_state, timestep, kv_cache)
    )

    rollout.calculate_advantage(discount=hypers.discount, gae_lambda=hypers.gae_lambda)

    return rollout, rngs

def ppo_loss(model: TransformerActorCritic, rollout: Rollout, batch_idx: jax.Array, hypers: PPOConfig, logs: TrainingLogs):
    batch_obs = rollout.obs.value[batch_idx]
    batch_target = rollout.targets.value[batch_idx]
    batch_log_prob = rollout.log_prob.value[batch_idx]
    batch_actions = rollout.actions.value[batch_idx]
    batch_advantage = rollout.advantages.value[batch_idx]
    batch_values = rollout.values.value[batch_idx, :-1]
    batch_rewards = rollout.rewards.value[batch_idx]

    # TODO: make this conditional
    # batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

    positions = jnp.arange(batch_obs.shape[1], dtype=jnp.int32)[None, :]

    values, policy, _ = model(TimeStep(
        obs=batch_obs,
        time=positions,
        last_action=batch_actions,
        last_reward=batch_rewards,
        step_type=jnp.zeros_like(batch_target, dtype=jnp.int32),
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
    rollout, rngs = evaluate(optimizer.model, rollout, rngs, env, hypers)

    minibatch_size = rollout.batch_size // hypers.minibatch_count

    batch_idx = jnp.arange(rollout.batch_size, dtype=jnp.int32)
    batch_idx = jnp.reshape(batch_idx, (hypers.minibatch_count, minibatch_size))

    def _step(i, x):
        optimizer, rollout, logs = x
        grads, logs = nnx.grad(ppo_loss, has_aux=True)(optimizer.model, rollout, batch_idx[i], hypers, logs)
        optimizer.update(grads)
        return optimizer, rollout, logs

    logs = create_training_logs()
    optimizer, rollout, logs = nnx.fori_loop(0, hypers.minibatch_count, _step, init_val=(optimizer, rollout, logs))

    logs = TrainingLogs(
        n=jnp.array(1.0),
        rewards=logs.rewards / logs.n,
        value_loss=logs.value_loss / logs.n,
        actor_loss=logs.actor_loss / logs.n,
        entropy_loss=logs.entropy_loss / logs.n,
        total_loss=logs.total_loss / logs.n
    )

    return optimizer, rngs, logs

app = typer.Typer()


def train_run(
    experiment: Experiment,
    trial: optuna.Trial | None = None,
):
    max_steps = experiment.config.max_env_steps

    logger = experiment.create_logger()
    checkpointer = Checkpointer(experiment.checkpoints_dir)
    checkpoint_interval = 200

    env = NBackMemory(n=2, max_value=5, length=max_steps)
    env = VmapWrapper(env, experiment.config.num_envs)
    batch_size = env.num_agents

    obs_spec = env.observation_spec
    action_spec = env.action_spec

    rngs = nnx.Rngs(default=experiment.default_seed)
    rollout = Rollout(batch_size, max_steps, obs_spec, action_spec)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        obs_spec.shape[0],
        action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs
    )

    optimizer = nnx.Optimizer(model=model, tx=create_optimizer(experiment.config.learner.optimizer, experiment.config.update_steps))
    trainer_hypers = experiment.config.learner.trainer
    jitted_train = nnx.jit(train, static_argnums=(3, 4))

    logs = None
    for i in track(range(experiment.config.update_steps), description="Training"):
        optimizer, rngs, logs = jitted_train(optimizer, rollout, rngs, env, trainer_hypers)

        # this should be delayed n-1 for jax to use async dispatch
        logger.log(logs._asdict(), i)
        if trial:
            trial.report(logs.rewards.item(), i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

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
        num_envs=2048,
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
                    ffn_size=256,
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
                learner_type="ppo",
                minibatch_count=1,
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
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='maximize',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(objective, n_trials=200)

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
    config=Config(
        seed="random",
        num_envs=128,
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
                    ffn_size=256,
                )
            ),
            optimizer=OptimizerConfig(
                type="adamw",
                learning_rate=1e-3,
                weight_decay=0.0,
                eps=1e-8,
                beta1=0.9,
                beta2=0.999,
                max_norm=0.5,
            ),
            trainer=PPOConfig(
                learner_type="ppo",
                minibatch_count=1,
                vf_coef=1.0,
                entropy_coef=0.001,
                vf_clip=0.2,
                discount=0.99,
                gae_lambda=0.9,
            ),
        ),
        logger=LoggerConfig(),
    )

    train_run(Experiment.from_config("trial_0", config))

if __name__ == '__main__':
    app()
