import time
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax.typing import DTypeLike
from jax import random
from flax import nnx

from jaxrl.constants import index_type

class ObservationSpec(NamedTuple):
    shape: tuple[int, ...]
    dtype: DTypeLike = jnp.float32


class Rollout(nnx.Module):
    def __init__(self, batch_size: int, trajectory_length: int, obs_spec: ObservationSpec):
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.obs_spec = obs_spec

        # observation gets plus one because we need to store the next trailing observation
        self.obs = nnx.Variable(jnp.zeros((batch_size, trajectory_length, *obs_spec.shape), dtype=obs_spec.dtype))
        self.actions = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=index_type))
        self.rewards = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=jnp.float32))

        self.log_prob = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=jnp.float32))
        self.values = nnx.Variable(jnp.zeros((batch_size, trajectory_length + 1), dtype=jnp.float32))

        # these are calculated in the rollout
        self.advantages = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=jnp.float32))
        self.targets = nnx.Variable(jnp.zeros((batch_size, trajectory_length), dtype=jnp.float32))

    def store(self, step: jax.Array, obs: jax.Array, action: jax.Array, reward: jax.Array, log_prob: jax.Array, value: jax.Array):
        # perhaps the order should be (timestep, batch, obs) for insertion and rotated to (batch, timestep, obs) for training but this could be a premature optimization
        self.obs.value = self.obs.value.at[:, step].set(obs)
        self.actions.value = self.actions.value.at[:, step].set(action)
        self.log_prob.value = self.log_prob.value.at[:, step].set(log_prob)
        self.values.value = self.values.value.at[:, step].set(value)
        self.rewards.value = self.rewards.value.at[:,step].set(reward)
    
    def calculate_advantage(self, discount: float, gae_lambda: float):
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
        
        self.advantages.value, self.targets.value = jax.vmap(_inner_calc)(self.rewards.value, self.values.value)


def bench(batch_size: int, context_size: int, obs_size: int, action_size: int, rollout: Rollout, rngs: nnx.Rngs):
    def _step(i, xs: tuple[Rollout, nnx.Rngs]):
        rollout, rngs = xs
        
        obs_seed = rngs.default()
        reward_seed = rngs.default()
        action_seed = rngs.default()
        log_prob_seed = rngs.default()
        value_seed = rngs.default()

        step = jnp.array(i, dtype=jnp.int32)
        agents_view = random.normal(obs_seed, (batch_size, obs_size))
        actions = random.randint(action_seed, (batch_size,), minval=0, maxval=action_size)
        reward = random.normal(reward_seed, (batch_size,))
        
        log_prob = random.normal(log_prob_seed, (batch_size,))
        value = random.normal(value_seed, (batch_size,))

        rollout.store(step, agents_view, actions, reward, log_prob, value)
        return rollout, rngs

    nnx.fori_loop(0, context_size, _step, (rollout, rngs))
    
    rollout.calculate_advantage(0.99, 0.95)
    # return rollout, rngs

def main():
    batch_size = 1024
    context_size = 128
    obs_size = 8 * 8 * 24
    action_size = 8
    rngs = nnx.Rngs(default=0)

    rollout = Rollout(batch_size, context_size, ObservationSpec(shape=(obs_size,)))

    
    jitted_bench = nnx.cached_partial(nnx.jit(bench, static_argnums=(0, 1, 2, 3), donate_argnums=(4, 5)), batch_size, context_size, obs_size, action_size, rollout, rngs)

    print("warmup")
    for _ in range(10):
        jitted_bench()
    
    rollout.targets.value.block_until_ready()
    
    print("benchmark")
    start = time.time()
    for _ in range(10000):
        jitted_bench()

    rollout.targets.value.block_until_ready()
    end = time.time()
    
    print(f"Execution time: {end - start:.4f} seconds")




if __name__ == '__main__':
    main()
