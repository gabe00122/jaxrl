from functools import partial
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

# from rlax import vmpo_loss, LagrangePenalty

import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


from jaxrl.config import Config, EnvironmentConfig, GridCnnObsEncoderConfig, LearnerConfig, LinearObsEncoderConfig, LoggerConfig, ModelConfig, OptimizerConfig, PPOConfig, ReturnConfig, TransformerActorCriticConfig, TransformerBlockConfig
from jaxrl.envs.environment import Environment
from jaxrl.envs.memory.n_back import NBackMemory
from jaxrl.envs.memory.return_2d import ReturnClient, ReturnEnv
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
            return ReturnEnv(env_config)
        case _:
            raise ValueError(f'Unknown environment type: {env_config.type}')


class TrainingLogs(NamedTuple):
    rewards: jax.Array
    value_loss: jax.Array
    actor_loss: jax.Array
    entropy_loss: jax.Array
    total_loss: jax.Array


def create_training_logs() -> TrainingLogs:
    return TrainingLogs(
        rewards=jnp.array(0.0),
        value_loss=jnp.array(0.0),
        actor_loss=jnp.array(0.0),
        entropy_loss=jnp.array(0.0),
        total_loss=jnp.array(0.0)
    )


def add_seq_dim(ts: TimeStep):
    return jax.tree_util.tree_map(lambda x: rearrange(x, 'b ... -> b 1 ...'), ts)
    # return TimeStep(
    #     obs=rearrange(ts.obs, 'b ... -> b 1 ...'),
    #     time=rearrange(ts.time, 'b ... -> b 1 ...'),
    #     last_action=rearrange(ts.last_action, 'b ... -> b 1 ...'),
    #     last_reward=rearrange(ts.last_reward, 'b ... -> b 1 ...'),
    #     action_mask=rearrange(ts.action_mask, 'b ... -> b 1 ...') if ts.action_mask is not None else None,
    # )

def evaluate(model: TransformerActorCritic, rollout: Rollout, rngs: nnx.Rngs, env: Environment, hypers: PPOConfig):
    reset_key = rngs.env()
    env_state, timestep = env.reset(reset_key)

    rollout_state = rollout.create_state()
    kv_cache = model.create_kv_cache(rollout.batch_size, rollout.trajectory_length)

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

def ppo_loss(model: TransformerActorCritic, rollout: RolloutState, hypers: PPOConfig):
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
    entropy_loss = -policy.entropy().mean()

    total_loss = hypers.vf_coef * value_loss + actor_loss + hypers.entropy_coef * entropy_loss

    logs = TrainingLogs(
        rewards=batch_rewards.sum() / batch_obs.shape[0],
        value_loss=value_loss,
        actor_loss=actor_loss,
        entropy_loss=entropy_loss,
        total_loss=total_loss
    )

    return total_loss, logs


def train(optimizer: nnx.Optimizer, rngs: nnx.Rngs, rollout: Rollout, env: Environment, config: Config):
    hypers = config.learner.trainer

    def _local_grad(model, rngs):
        rollout_state, rngs = evaluate(model, rollout, rngs, env, hypers)
        grad, logs = nnx.grad(ppo_loss, has_aux=True)(model, rollout_state, hypers)
        return grad, logs, rngs

    def _global_grad(model, rngs):
        grads, logs, rngs = nnx.vmap(_local_grad, in_axes=(None, 0), out_axes=(0, 0, 0))(model, rngs)

        logs = jax.tree_util.tree_map(lambda x: jnp.mean(x), logs)
        grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)

        return grad, (logs, rngs)

    def _global_step(i, x):
        optimizer, logs, rngs = x

        grad, (step_logs, rngs) = _global_grad(optimizer.model, rngs)
        optimizer.update(grad)

        logs = jax.tree_util.tree_map(lambda x, y: x + y, logs, step_logs)

        return optimizer, logs, rngs

    logs = create_training_logs()

    optimizer, logs, rngs = nnx.fori_loop(0, config.updates_per_jit, _global_step, init_val=(optimizer, logs, rngs))

    logs = jax.tree_util.tree_map(lambda x: x / config.updates_per_jit, logs)

    return optimizer, rngs, logs

app = typer.Typer(pretty_exceptions_show_locals=False)

@app.command()
def enjoy():
    experiment: Experiment = Experiment.load("lazy-bear-a9bmts")
    max_steps = experiment.config.max_env_steps

    env = create_env(experiment.config.environment, max_steps)

    obs_spec = env.observation_spec
    action_spec = env.action_spec
    rngs = nnx.Rngs(default=0)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        obs_spec,
        action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs
    )
    optimizer = nnx.Optimizer(model=model, tx=create_optimizer(experiment.config.learner.optimizer, experiment.config.update_steps))

    with Checkpointer(experiment.checkpoints_dir) as checkpointer:
        optimizer = checkpointer.restore_latest(optimizer)

    model = optimizer.model

    kv_cache = model.create_kv_cache(env.num_agents, max_steps)
    client = ReturnClient(env)

    @nnx.jit
    def step(timestep, kv_cache, env_state, rngs):
        action_key = rngs.action()
        env_key = rngs.env()
        _, policy, kv_cache = model(add_seq_dim(timestep), kv_cache)
        actions = policy.sample(seed=action_key)
        actions = jnp.squeeze(actions, axis=-1)

        env_state, timestep = env.step(env_state, actions, env_key)

        return env_state, timestep, kv_cache, rngs

    for _ in range(5):
        env_state, timestep = env.reset(rngs.env())
        for _ in range(max_steps):
            env_state, timestep, kv_cache, rngs = step(timestep, kv_cache, env_state, rngs)
            client.render(env_state)

    client.save_video()

def replicate_model(optimizer, sharding):
    state = nnx.state(optimizer)
    state = jax.device_put(state, sharding)
    nnx.update(optimizer, state)


def train_run(
    experiment: Experiment,
    trial: optuna.Trial | None = None,
):
    mesh = Mesh(devices=jax.devices(), axis_names=('batch',))
    replicate_sharding = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P('batch'))

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

    replicate_model(optimizer, replicate_sharding)

    rng = jax.random.PRNGKey(experiment.default_seed)
    device_rngs = jax.random.split(rng, len(jax.devices()))
    device_rngs = jax.device_put(device_rngs, batch_sharding)
    rngs = nnx.Rngs(default=device_rngs)

    jitted_train = nnx.jit(train, static_argnums=(2, 3, 4))

    env_steps_per_update = batch_size * max_steps * device_rngs.shape[0] * experiment.config.updates_per_jit
    outer_updates = experiment.config.update_steps // experiment.config.updates_per_jit

    logs = None
    for i in track(range(outer_updates), description="Training", disable=False):
        start_time = time.time()

        with mesh:
            optimizer, rngs, logs = jitted_train(optimizer, rngs, rollout, env, experiment.config)

        # this should be delayed n-1 for jax to use async dispatch
        logger.log(logs._asdict(), i)

        stop_time = time.time()

        delta_time = (stop_time - start_time)
        print(int(env_steps_per_update / delta_time))
        print(experiment.config.updates_per_jit / delta_time)

        if trial:
            trial.report(logs.rewards.item(), i)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

        # if i % checkpoint_interval == checkpoint_interval - 1:
        #     checkpointer.save(optimizer, i)

    checkpointer.save(optimizer, experiment.config.update_steps)

    logger.close()
    checkpointer.close()

    if logs:
        return logs.rewards.item()
    return -1.0


def objective(trial: optuna.Trial):
    config=Config(
        seed=0,
        num_envs=32,
        max_env_steps=256,
        update_steps=20000,
        updates_per_jit=100,
        environment=ReturnConfig(
            num_agents=16
        ),
        learner=LearnerConfig(
            model=TransformerActorCriticConfig(
                obs_encoder=GridCnnObsEncoderConfig(),
                hidden_features=128,
                num_layers=3,
                activation="gelu",
                norm="layer_norm",
                dtype="bfloat16",
                param_dtype="float32",
                transformer_block=TransformerBlockConfig(
                    num_heads=4,
                    ffn_size=512,
                    glu=False,
                    gtrxl_gate=False,
                )
            ),
            optimizer=OptimizerConfig(
                type="adamw",
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
                eps=1e-8,
                beta1=0.9,
                beta2=0.999,
                max_norm=trial.suggest_float("max_norm", 0.1, 1.0),
            ),
            trainer=PPOConfig(
                trainer_type="ppo",
                minibatch_count=1,
                vf_coef=trial.suggest_float("vf_coef", 0.5, 2.0),
                entropy_coef=trial.suggest_float("entropy_coef", 0.0, 0.01),
                vf_clip=trial.suggest_float("vf_clip", 0.1, 0.3),
                discount=trial.suggest_float("discount", 0.9, 0.99),
                gae_lambda=trial.suggest_float("gae_lambda", 0.9, 0.99),
            ),
        ),
        logger=LoggerConfig(
            use_wandb=True
        ),
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
def train_cmd(distributed: bool = False):
    if distributed:
        jax.distributed.initialize()
    experiment = Experiment.from_config_file(Path("./config/return.json"))

    train_run(experiment)

if __name__ == '__main__':
    app()
