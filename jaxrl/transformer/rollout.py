from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax.typing import DTypeLike
from flax import nnx
from numpy import dtype

from jaxrl.constants import index_type
from jaxrl.envs.wrapper import TimeStep
from jaxrl.types import Observation

class ObservationSpec(NamedTuple):
    shape: tuple[int, ...]
    dtype: DTypeLike = jnp.float32


class Rollout:
    def __init__(self, batch_size: int, trajectory_length: int, obs_spec: ObservationSpec):
        # observation gets plus one because we need to store the next trailing observation
        self.obs = nnx.Variable(jnp.zeros((batch_size, trajectory_length, *obs_spec.shape), dtype=obs_spec.dtype))
        self.actions = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=index_type))
        self.reward = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=jnp.float32))

        self.log_prob = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=jnp.float32))
        self.values = nnx.Variable(jnp.zeros((batch_size, trajectory_length + 1), dtype=jnp.float32))

        # these are calculated in the rollout
        self.advantage = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=jnp.float32))
        self.target = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=jnp.float32))

    def store(self, step: jax.Array, obs: jax.Array, action: jax.Array, reward: jax.Array, log_prob: jax.Array, value: jax.Array):
        # perhaps the order should be (timestep, batch, obs) for insertion and rotated to (batch, timestep, obs) for training but this could be a premature optimization
        self.obs.value = self.obs.value.at[:, step].set(obs)
        self.actions.value = self.actions.value.at[:, step].set(action)
        self.log_prob.value = self.log_prob.value.at[:, step].set(log_prob)
        self.values.value = self.values.value.at[:, step].set(value)
        self.reward.value = self.reward.value.at[:, step].set(reward)
    
    def calculate_advantage(self, discount: float, gae_lambda: float):
        lambda_ = jnp.ones_like(discount) * gae_lambda

        delta_t = self.reward.value + discount * self.values.value[1:] - self.values.value[:-1]

        # Iterate backwards to calculate advantages.
        def _body(acc, xs):
            deltas, discounts, lambda_ = xs
            acc = deltas + discounts * lambda_ * acc
            return acc, acc

        _, advantage = jax.lax.scan(_body, 0.0, (delta_t, discount, lambda_), reverse=True)

        self.advantage.value = self.advantage.value
        self.target.value = advantage + self.values.value


def main():
    batch_size = 4
    context_size = 32
    action_size = 8
    rngs = nnx.Rngs()

    rollout = Rollout(batch_size, context_size, ObservationSpec(shape=(action_size,)))
    
    for i in range(context_size):
        obs_seed = rngs.default()
        reward_seed = rngs.default()

        step = jnp.array(i, dtype=jnp.int32)
        agents_view = 


if __name__ == '__main__':
    main()
    main()
