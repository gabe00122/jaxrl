import jax
from jax import numpy as jnp

from jaxrl.envs.environment import Environment


class TaskIdWrapper:
    def __init__(self, env: Environment, task_id: int):
        self._env = env
        self._task_ids = jnp.full((env.num_agents,), task_id)

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    def reset(self, rng_key: jax.Array):
        state, ts = self._env.reset(rng_key)
        ts = ts._replace(task_ids=self._task_ids)
        return state, ts

    def step(self, state, action, rng_key):
        new_state, ts = self._env.step(state, action, rng_key)
        ts = ts._replace(task_ids=self._task_ids)
        return new_state, ts
