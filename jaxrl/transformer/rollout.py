from typing import NamedTuple

import jax
from jax import numpy as jnp

from jaxrl.constants import index_type
from jaxrl.envs.specs import ActionSpec, ObservationSpec
from jaxrl.types import TimeStep


class RolloutState(NamedTuple):
    # observation gets plus one because we need to store the next trailing observation
    obs: jax.Array
    action_mask: jax.Array
    actions: jax.Array
    rewards: jax.Array

    log_prob: jax.Array
    values: jax.Array

    # these are calculated in the rollout
    advantages: jax.Array
    targets: jax.Array

    # these are used as part of the observation
    last_actions: jax.Array
    last_rewards: jax.Array


class Rollout:
    def __init__(
        self,
        batch_size: int,
        trajectory_length: int,
        obs_spec: ObservationSpec,
        action_spec: ActionSpec,
    ):
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def create_state(self) -> RolloutState:
        return RolloutState(
            # observation gets plus one because we need to store the next trailing observation
            obs=jnp.zeros(
                (self.batch_size, self.trajectory_length, *self.obs_spec.shape),
                dtype=self.obs_spec.dtype,
            ),
            action_mask=jnp.ones(
                (self.batch_size, self.trajectory_length, self.action_spec.num_actions),
                dtype=jnp.bool,
            ),
            actions=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=index_type
            ),
            rewards=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            log_prob=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            values=jnp.zeros(
                (self.batch_size, self.trajectory_length + 1), dtype=jnp.float32
            ),
            # these are calculated in the rollout
            advantages=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            targets=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
            last_actions=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.int32
            ),
            last_rewards=jnp.zeros(
                (self.batch_size, self.trajectory_length), dtype=jnp.float32
            ),
        )

    def store(
        self,
        state: RolloutState,
        step: jax.Array,
        timestep: TimeStep,
        next_timestep: TimeStep,
        log_prob: jax.Array,
        value: jax.Array,
    ) -> RolloutState:
        return state._replace(
            obs=state.obs.at[:, step].set(timestep.obs),
            action_mask=(
                state.action_mask.at[:, step].set(timestep.action_mask)
                if timestep.action_mask is not None
                else state.action_mask
            ),
            actions=state.actions.at[:, step].set(next_timestep.last_action),
            log_prob=state.log_prob.at[:, step].set(log_prob),
            values=state.values.at[:, step].set(value),
            rewards=state.rewards.at[:, step].set(next_timestep.last_reward),
            last_actions=state.last_actions.at[:, step].set(timestep.last_action),
            last_rewards=state.last_rewards.at[:, step].set(timestep.last_reward),
        )

    def calculate_advantage(
        self, state: RolloutState, discount: float, gae_lambda: float
    ) -> RolloutState:
        def _inner_calc(rewards, values):
            delta_t = rewards + discount * values[1:] - values[:-1]

            # Iterate backwards to calculate advantages.
            def _body(acc, xs):
                deltas = xs
                acc = deltas + discount * gae_lambda * acc
                return acc, acc

            _, advantages = jax.lax.scan(_body, 0.0, delta_t, reverse=True)
            targets = values[:-1] + advantages

            return advantages, targets

        advantages, targets = jax.vmap(_inner_calc)(state.rewards, state.values)

        return state._replace(
            advantages=advantages,
            targets=targets,
        )
