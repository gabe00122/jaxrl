from functools import cached_property

import jax
from jax import numpy as jnp

from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import ActionSpec, DiscreteActionSpec, ObservationSpec
from jaxrl.envs.task_id_wrapper import TaskIdWrapper


def _stack_pytree(batch):
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batch)


class MultiTaskWrapper(Environment):
    def __init__(self, envs: tuple[Environment], env_names: tuple[str]) -> None:
        envs = [TaskIdWrapper(env, task_id) for task_id, env in enumerate(envs)]

        self._envs = envs
        self._env_names = env_names

    def reset(self, rng_key: jax.Array):
        rng_keys = jax.random.split(rng_key, len(self._envs))

        states = []
        timesteps = []

        for i, env in enumerate(self._envs):
            s, t = env.reset(rng_keys[i])
            states.append(s)
            timesteps.append(t)

        return tuple(states), _stack_pytree(timesteps)

    def step(self, states, actions: jax.Array, rng_key: jax.Array):
        rng_keys = jax.random.split(rng_key, len(self._envs))

        state_out = []
        timesteps = []

        start = 0
        for i, env in enumerate(self._envs):
            end = start + env.num_agents
            s, t = env.step(states[i], actions[start:end], rng_keys[i])
            start = end

            state_out.append(s)
            timesteps.append(t)

        return tuple(state_out), _stack_pytree(timesteps)

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        # max_channel = max([env.observation_spec.max_value for env in self._envs])
        # dtype = self._envs[0].observation_spec.dtype
        # shape = self._envs[0].observation_spec.shape
        
        return self._envs[0].observation_spec # todo: we shouldn't assume they are all the same


    @cached_property
    def action_spec(self) -> ActionSpec:
        num_actions = max([env.action_spec.num_actions for env in self._envs])

        return DiscreteActionSpec(num_actions)

    @property
    def is_jittable(self) -> bool:
        return all([env.is_jittable for env in self._envs])

    @property
    def num_agents(self) -> int:
        return sum([env.num_agents for env in self._envs])
    
    @property
    def num_tasks(self) -> int:
        return len(self._envs)
    
    @property
    def teams(self) -> jax.Array:
        return jnp.concatenate([env.teams for env in self._envs], axis=0)

    def create_placeholder_logs(self):
        return {
            name: env.create_placeholder_logs()
            for name, env in zip(self._env_names, self._envs)
        }

    def create_logs(self, state):
        return {
            name: env.create_logs(s)
            for name, env, s in zip(self._env_names, self._envs, state)
        }
    
