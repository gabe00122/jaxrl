from typing import NamedTuple

from einops import rearrange
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
    terminated: jax.Array
    next_terminated: jax.Array

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
            obs=jnp.zeros((self.batch_size, *self.obs_spec.shape), dtype=self.obs_spec.dtype),
            action_mask=jnp.ones((self.batch_size, self.action_spec.num_actions), dtype=jnp.bool_),
            actions=jnp.zeros((self.batch_size,), dtype=index_type),
            rewards=jnp.zeros((self.batch_size,), dtype=jnp.float32),
            terminated=jnp.zeros((self.batch_size,), dtype=jnp.bool_),
            next_terminated=jnp.zeros((self.batch_size,), dtype=jnp.bool_),
            log_prob=jnp.zeros((self.batch_size,), dtype=jnp.float32),
            values=jnp.zeros((self.batch_size,), dtype=jnp.float32),
            advantages=jnp.zeros((self.batch_size,), dtype=jnp.float32),
            targets=jnp.zeros((self.batch_size,), dtype=jnp.float32),
            last_actions=jnp.zeros((self.batch_size,), dtype=jnp.int32),
            last_rewards=jnp.zeros((self.batch_size,), dtype=jnp.float32),
        )

    def store(
        self,
        timestep: TimeStep,
        next_timestep: TimeStep,
        log_prob: jax.Array,
        value: jax.Array,
    ) -> RolloutState:
        action_mask = (
            timestep.action_mask
            if timestep.action_mask is not None
            else jnp.ones((self.batch_size, self.action_spec.num_actions), dtype=jnp.bool_)
        )
        return RolloutState(
            obs=timestep.obs,
            action_mask=action_mask,
            actions=next_timestep.last_action,
            rewards=next_timestep.last_reward,
            terminated=timestep.terminated,
            next_terminated=next_timestep.terminated,
            log_prob=log_prob,
            values=value,
            advantages=jnp.zeros((self.batch_size,), dtype=jnp.float32),
            targets=jnp.zeros((self.batch_size,), dtype=jnp.float32),
            last_actions=timestep.last_action,
            last_rewards=timestep.last_reward,
        )

    def calculate_advantage(
        self, state: RolloutState, discount: float, gae_lambda: float
    ) -> RolloutState:
        def _body(acc, xs):
            rewards, discount, v_tp1 = xs
            acc = rewards + discount * ((1 - gae_lambda) * v_tp1 + gae_lambda * acc)
            return acc, acc

        # swap to time major
        rewards = jnp.swapaxes(state.rewards, 0, 1)
        terminated = jnp.swapaxes(state.next_terminated, 0, 1)
        values = jnp.swapaxes(state.values, 0, 1)

        _, targets = jax.lax.scan(
            _body,
            values[-1],
            (rewards, jnp.where(terminated, 0.0, discount), values[1:]),
            reverse=True,
        )
        advantage = targets - values[:-1]

        targets = jnp.swapaxes(targets, 0, 1)
        advantages = jnp.swapaxes(advantage, 0, 1)

        return state._replace(
            advantages=advantages,
            targets=targets,
        )

    def _shuffle(self, state: RolloutState, rng_key: jax.Array) -> RolloutState:
        indecies = jax.random.permutation(rng_key, self.batch_size)
        return jax.tree.map(lambda x: x[indecies], state)

    def create_minibatches(
        self, state: RolloutState, minibatches: int, rng_key: jax.Array
    ) -> RolloutState:
        if minibatches > 1:
            state = self._shuffle(state, rng_key)

        return jax.tree.map(
            lambda x: rearrange(x, "(m b) ... -> m b ...", m=minibatches), state
        )
