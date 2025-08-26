from functools import partial
import time
from typing import NamedTuple
import jax
import jax.numpy as jnp
from flax import nnx
from einops import rearrange
import optuna
from rich.progress import track
from rich.console import Console

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
from jaxrl.util import count_parameters, format_count


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
    carry = model.initialize_carry(rollout.batch_size, rngs)

    def _step(i, x):
        rollout_state, rngs, env_state, timestep, carry = x

        action_key = rngs.action()
        env_key = rngs.env()

        value_rep, policy, carry = model(add_seq_dim(timestep), carry)

        action = policy.sample(seed=action_key)
        log_prob = policy.log_prob(action).squeeze(axis=-1)
        action = action.squeeze(axis=-1)
        value = model.get_value(value_rep).squeeze(axis=-1)

        env_state, next_timestep = env.step(env_state, action, env_key)

        rollout_state = rollout.store(
            rollout_state,
            step=i,
            timestep=timestep,
            next_timestep=next_timestep,
            log_prob=log_prob,
            value=value,
        )

        return rollout_state, rngs, env_state, next_timestep, carry

    rollout_state, rngs, _, _, _ = nnx.fori_loop(
        0,
        rollout.trajectory_length,
        _step,
        init_val=(rollout_state, rngs, env_state, timestep, carry),
    )

    # save the last value
    value_rep, _, _ = model(add_seq_dim(timestep), carry)
    value = model.get_value(value_rep).squeeze(axis=-1)
    rollout_state._replace(
        values=rollout_state.values.at[:, -1].set(value)
    )

    rollout_state = rollout.calculate_advantage2(
        rollout_state, discount=hypers.discount, gae_lambda=hypers.gae_lambda
    )

    return rollout_state, rngs

def ppo_loss(model: TransformerActorCritic, rollout: RolloutState, hypers: PPOConfig):
    batch_obs = rollout.obs
    batch_target = rollout.targets
    batch_log_prob = rollout.log_prob
    batch_actions = rollout.actions
    batch_action_masks = rollout.action_mask
    batch_advantage = rollout.advantages
    # batch_values = rollout.values[..., :-1])
    # batch_rewards = rollout.rewards
    batch_terminated = rollout.terminated

    batch_last_actions = rollout.last_actions
    batch_last_rewards = rollout.last_rewards

    if hypers.normalize_advantage:
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

    positions = jnp.arange(batch_obs.shape[1], dtype=jnp.int32)[None, :]

    value_rep, policy, _ = model(
        TimeStep(
            obs=batch_obs,
            time=positions,
            terminated=batch_terminated,
            last_action=batch_last_actions,
            last_reward=batch_last_rewards,
            action_mask=batch_action_masks,
        ),
    )
    log_probs = policy.log_prob(batch_actions)

    value_loss = model.get_value_loss(value_rep, batch_target)

    # value_pred_clipped = batch_values + jnp.clip(
    #     values - batch_values, -hypers.vf_clip, hypers.vf_clip
    # )

    # value_losses = jnp.square(values - batch_target)
    # value_losses_clipped = jnp.square(value_pred_clipped - batch_target)
    # value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

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
        rewards=jnp.array(0.0),
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

    @partial(nnx.grad, has_aux=True)
    def _vec_grad(model, rollout_state):
        return ppo_loss(model, rollout_state, hypers)

    def _minibatch_step(carry, rollout_state):
        optimizer, logs = carry
        grad, step_logs = _vec_grad(optimizer.model, rollout_state)

        # combine accross devices
        # step_logs = jax.tree_util.tree_map(lambda x: jnp.mean(x), step_logs)
        # grad = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), grads)

        optimizer.update(grad)
        logs = jax.tree.map(lambda x, y: x + y, logs, step_logs)

        return (optimizer, logs)

    def _epoch_step(i, x):
        optimizer, rollout_state, logs, rngs = x

        minibatch_rng = rngs.shuffle()
        minibatch_rollout_state = rollout.create_minibatches(rollout_state, hypers.minibatch_count, minibatch_rng)
        optimizer, logs = nnx.scan(_minibatch_step, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)((optimizer, logs), minibatch_rollout_state)

        return optimizer, rollout_state, logs, rngs

    def _global_step(i, x):
        optimizer, logs, rngs = x
        rollout_state, rngs = evaluate(optimizer.model, rollout, rngs, env, hypers)
        optimizer, rollout_state, logs, rngs = nnx.fori_loop(
            0,
            hypers.epoch_count,
            _epoch_step,
            init_val=(optimizer, rollout_state, logs, rngs)
        )

        logs = logs._replace(
            rewards=logs.rewards + rollout.calculate_cumulative_rewards(rollout_state).mean() * hypers.epoch_count * hypers.minibatch_count # this is because it's divided by the minibatch/epoch count and shouldn't
        )

        return optimizer, logs, rngs

    logs = create_training_logs()

    optimizer, logs, rngs = nnx.fori_loop(
        0, config.updates_per_jit, _global_step, init_val=(optimizer, logs, rngs)
    )

    logs = jax.tree_util.tree_map(lambda x: x / (config.updates_per_jit * hypers.epoch_count * hypers.minibatch_count), logs)

    return optimizer, rngs, logs


def replicate_model(optimizer, sharding):
    state = nnx.state(optimizer)
    state = jax.device_put(state, sharding)
    nnx.update(optimizer, state)

def block_all(xs):
  return jax.tree_util.tree_map(lambda x: x.block_until_ready(), xs)


def train_run(
    experiment: Experiment,
    trial: optuna.Trial | None = None,
    profile: bool = False
):
    console = Console()
    # mesh = Mesh(devices=jax.devices(), axis_names=("batch",))
    # replicate_sharding = NamedSharding(mesh, P())
    # batch_sharding = NamedSharding(mesh, P("batch"))

    max_steps = experiment.config.max_env_steps

    logger = experiment.create_logger(console)
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

    optimizer = nnx.ModelAndOptimizer(
        model=model,
        tx=create_optimizer(
            experiment.config.learner.optimizer,
            experiment.config.update_steps
            * experiment.config.learner.trainer.minibatch_count
            * experiment.config.learner.trainer.epoch_count
        ),
        wrt=nnx.Param
    )

    # replicate_model(optimizer, replicate_sharding)

    # rng = jax.random.PRNGKey(experiment.default_seed)
    # device_rngs = jax.random.split(rng, len(jax.devices()))
    # device_rngs = jax.device_put(device_rngs, batch_sharding)
    rngs = nnx.Rngs(default=jax.random.PRNGKey(experiment.default_seed))

    jitted_train = nnx.jit(train, static_argnums=(2, 3, 4))

    env_steps_per_update = (
        batch_size
        * max_steps
        # * device_rngs.shape[0]
        * experiment.config.updates_per_jit
    )
    outer_updates = experiment.config.update_steps // experiment.config.updates_per_jit

    console.print(f"Starting Training: {experiment.unique_token}")
    console.print(f"Parameter Count: {count_parameters(model)}")

    logs = None
    for i in track(range(outer_updates), description="Training", console=console):
        start_time = time.time()

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
        console.print(f"Steps per second: {format_count(int(env_steps_per_update / delta_time))}")
        console.print(f"Updates per second: {experiment.config.updates_per_jit / delta_time}")

        if trial:
            trial.report(logs.rewards.item(), i)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()

        if i % (outer_updates // 5) == (outer_updates // 5) - 1:
            checkpointer.save(optimizer.model, i * experiment.config.updates_per_jit)
            # optimizer.model.preturb(rngs)

    checkpointer.save(optimizer.model, experiment.config.update_steps)

    logger.close()
    checkpointer.close()

    if logs:
        return logs.rewards.item()
    return -1.0
