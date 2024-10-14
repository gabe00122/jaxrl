from jax import numpy as jnp
from flax import nnx


class CumulativeReward(nnx.Metric):
    def __init__(self, shape):
        self.last_episode_rewards = nnx.Variable(jnp.zeros(shape))
        self.other_episode_rewards = nnx.Variable(jnp.zeros(shape))
        self.counts = nnx.Variable(jnp.ones(shape, dtype=jnp.uint32))
        self.accumulator = nnx.Variable(jnp.zeros(shape))

    def reset(self) -> None:
        self.other_episode_rewards.value = jnp.zeros_like(
            self.other_episode_rewards.value
        )
        self.counts.value = jnp.ones_like(self.counts.value)

    def update(self, *, reward, done, **kwargs) -> None:
        self.accumulator.value += reward
        self.counts.value += done

        self.other_episode_rewards.value += done * self.last_episode_rewards.value
        self.last_episode_rewards.value = jnp.where(
            done, self.accumulator.value, self.last_episode_rewards.value
        )
        self.accumulator.value *= 1.0 - done

    def compute(self):
        average_rewards = (
            self.last_episode_rewards.value + self.other_episode_rewards.value
        ) / self.counts.value
        return jnp.mean(average_rewards)
