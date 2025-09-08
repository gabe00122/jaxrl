import jax
from jax import numpy as jnp
from functools import cached_property

from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import ObservationSpec, ActionSpec
from jaxrl.types import TimeStep


class TaskIdWrapper(Environment):
    """Environment wrapper that appends a constant task id to timesteps."""

    def __init__(self, env: Environment, task_id: int):
        self._env = env
        self._task_id = jnp.asarray(task_id, dtype=jnp.int32)

    def _add_task_id(self, timestep: TimeStep) -> TimeStep:
        task_ids = jnp.full((self._env.num_agents,), self._task_id, dtype=jnp.int32)
        return timestep._replace(task_id=task_ids)

    def reset(self, rng_key: jax.Array):
        state, timestep = self._env.reset(rng_key)
        return state, self._add_task_id(timestep)

    def step(self, state, action: jax.Array, rng_key: jax.Array):
        state, timestep = self._env.step(state, action, rng_key)
        return state, self._add_task_id(timestep)

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return self._env.observation_spec

    @cached_property
    def action_spec(self) -> ActionSpec:
        return self._env.action_spec

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    @property
    def is_jittable(self) -> bool:
        return self._env.is_jittable

    def create_placeholder_logs(self):
        return self._env.create_placeholder_logs()

    def create_logs(self, state):
        return self._env.create_logs(state)
