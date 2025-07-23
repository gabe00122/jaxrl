from functools import partial
import time
from typing import NamedTuple
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optuna
from rich.progress import track

# from rlax import vmpo_loss, LagrangePenalty

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding


from jaxrl.config import Config, PPOConfig
from jaxrl.envs.create import create_env
from jaxrl.envs.environment import Environment
from jaxrl.envs.vmap_wrapper import VmapWrapper
from jaxrl.experiment import Experiment
from jaxrl.optimizer import create_optimizer
from jaxrl.transformer.network import TransformerActorCritic
from jaxrl.transformer.rollout import Rollout, RolloutState
from jaxrl.types import TimeStep
from jaxrl.checkpointer import Checkpointer


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
        total_loss=jnp.array(0.0),
    )


def add_seq_dim(ts: TimeStep):
    return jax.tree_util.tree_map(lambda x: rearrange(x, "b ... -> b 1 ..."), ts)


def evaluate(
    model: TransformerActorCritic,
    rollout: Rollout,
    rngs: nnx.Rngs,
    env: Environment,
    hypers: PPOConfig,
):
    reset_key = rngs.env()
    env_state, timestep = env.reset(reset_key)

    rollout_state = rollout.create_state()
    kv_cache = model.create_kv_cache(rollout.batch_size)

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
        init_val=(rollout_state, rngs, env_state, timestep, kv_cache),
    )

    rollout_state = rollout.calculate_advantage(
        rollout_state, discount=hypers.discount, gae_lambda=hypers.gae_lambda
    )

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

    if hypers.normalize_advantage:
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

    positions = jnp.arange(batch_obs.shape[1], dtype=jnp.int32)[None, :]

    values, policy, _ = model(
        TimeStep(
            obs=batch_obs,
            time=positions,
            last_action=batch_last_actions,
            last_reward=batch_last_rewards,
            action_mask=None,
        )
    )
    log_probs = policy.log_prob(batch_actions)

    value_pred_clipped = batch_values + jnp.clip(
        values - batch_values, -hypers.vf_clip, hypers.vf_clip
    )

    value_losses = jnp.square(values - batch_target)
    value_losses_clipped = jnp.square(value_pred_clipped - batch_target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_probs - batch_log_prob)

    pg_loss1 = ratio * batch_advantage
    pg_loss2 = (
        jnp.clip(ratio, 1.0 - hypers.vf_clip, 1.0 + hypers.vf_clip) * batch_advantage
    )

    actor_loss = -jnp.minimum(pg_loss1, pg_loss2).mean()

    # Entropy regularization
    entropy_loss = -policy.entropy().mean()

    total_loss = (
        hypers.vf_coef * value_loss + actor_loss + hypers.entropy_coef * entropy_loss
    )

    logs = TrainingLogs(
        rewards=batch_rewards.sum() / batch_obs.shape[0],
        value_loss=value_loss,
        actor_loss=actor_loss,
        entropy_loss=entropy_loss,
        total_loss=total_loss,
    )

    return total_loss, logs

def train(
    optimizer: nnx.Optimizer,
    rngs: nnx.Rngs,
    rollout: Rollout,
    env: Environment,
    config: Config,
):
    hypers = config.learner.trainer

    @partial(nnx.vmap, in_axes=(None, 0), out_axes=(0, 0))
    @jax.named_scope("evaluate")
    def _vec_rollout(model, rngs) -> tuple[RolloutState, nnx.Rngs]:
        return evaluate(model, rollout, rngs, env, hypers)

    @partial(nnx.vmap, in_axes=(None, 0), out_axes=(0, 0))
    @partial(nnx.grad, has_aux=True)
    @jax.named_scope("gradient")
    def _vec_grad(model, rollout_state):
        return ppo_loss(model, rollout_state, hypers)

    def _minibatch_step(i, x):
        optimizer, rollout_state, logs = x

        grads, step_logs = _vec_grad(optimizer.model, rollout_state)

        # combine accross devices
        step_logs = jax.tree_util.tree_map(lambda x: jnp.mean(x), step_logs)
        grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)

        optimizer.update(grad)
        logs = jax.tree_util.tree_map(lambda x, y: x + y, logs, step_logs)

        return optimizer, rollout_state, logs

    def _global_step(i, x):
        optimizer, logs, rngs = x
        rollout_state, rngs = _vec_rollout(optimizer.model, rngs)
        optimizer, rollout_state, logs = nnx.fori_loop(
            0,
            hypers.minibatch_count,
            _minibatch_step,
            init_val=(optimizer, rollout_state, logs)
        )

        return optimizer, logs, rngs

    logs = create_training_logs()

    optimizer, logs, rngs = nnx.fori_loop(
        0, config.updates_per_jit, _global_step, init_val=(optimizer, logs, rngs)
    )

    logs = jax.tree_util.tree_map(lambda x: x / (config.updates_per_jit * hypers.minibatch_count), logs)

    return optimizer, rngs, logs


def replicate_model(optimizer, sharding):
    state = nnx.state(optimizer)
    state = jax.device_put(state, sharding)
    nnx.update(optimizer, state)

def block_all(xs):
  jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)
  return xs

def train_run(
    experiment: Experiment,
    trial: optuna.Trial | None = None,
    profile: bool = False
):
    mesh = Mesh(devices=jax.devices(), axis_names=("batch",))
    replicate_sharding = NamedSharding(mesh, P())
    batch_sharding = NamedSharding(mesh, P("batch"))

    max_steps = experiment.config.max_env_steps

    logger = experiment.create_logger()
    checkpointer = Checkpointer(experiment.checkpoints_url)

    env = create_env(
        experiment.config.environment, max_steps
    )
    env = VmapWrapper(env, experiment.config.num_envs)
    batch_size = env.num_agents

    rngs = nnx.Rngs(default=experiment.default_seed)
    rollout = Rollout(batch_size, max_steps, env.observation_spec, env.action_spec)

    model = TransformerActorCritic(
        experiment.config.learner.model,
        env.observation_spec,
        env.action_spec.num_actions,
        max_seq_length=max_steps,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(
        model=model,
        tx=create_optimizer(
            experiment.config.learner.optimizer,
            experiment.config.update_steps
            * experiment.config.learner.trainer.minibatch_count,
        ),
    )

    replicate_model(optimizer, replicate_sharding)

    rng = jax.random.PRNGKey(experiment.default_seed)
    device_rngs = jax.random.split(rng, len(jax.devices()))
    device_rngs = jax.device_put(device_rngs, batch_sharding)
    rngs = nnx.Rngs(default=device_rngs)

    jitted_train = nnx.jit(train, static_argnums=(2, 3, 4))

    env_steps_per_update = (
        batch_size
        * max_steps
        * device_rngs.shape[0]
        * experiment.config.updates_per_jit
    )
    outer_updates = experiment.config.update_steps // experiment.config.updates_per_jit

    logs = None
    for i in track(range(outer_updates), description="Training", disable=False):
        start_time = time.time()

        with mesh:
            optimizer, rngs, logs = jitted_train(optimizer, rngs, rollout, env, experiment.config)

            if profile and i >= 4:
                with jax.profiler.trace("/tmp/jax-trace"):
                    optimizer, rngs, logs = jitted_train(optimizer, rngs, rollout, env, experiment.config)
                    block_all(nnx.state(optimizer))

                break


        # this should be delayed n-1 for jax to use async dispatch
        logger.log(logs._asdict(), i)

        stop_time = time.time()

        delta_time = stop_time - start_time
        print(int(env_steps_per_update / delta_time))
        print(experiment.config.updates_per_jit / delta_time)

        if trial:
            trial.report(logs.rewards.item(), i)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

        if i % (outer_updates // 5) == (outer_updates // 5) - 1:
            checkpointer.save(optimizer, i * experiment.config.updates_per_jit)

    checkpointer.save(optimizer, experiment.config.update_steps)

    logger.close()
    checkpointer.close()

    if logs:
        return logs.rewards.item()
    return -1.0
