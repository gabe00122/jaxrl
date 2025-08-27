from functools import cached_property
from typing import Sequence

import jax
from jax import numpy as jnp
from einops import rearrange

from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import ActionSpec, DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep


def _stack_pytree(batch):
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batch)


# def _slice_pytree(tree, _slice):
#     return jax.tree.map(lambda xs: xs[_slice], tree)


class MultiplexWrapper[Environment]:
    def __init__(self, envs: tuple[Environment]) -> None:
        self._envs = envs

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
        max_channel = max([env.observation_spec.max_value for env in self._envs])
        dtype = self._envs[0].observation_spec.dtype
        shape = self._envs[0].observation_spec.shape

        return ObservationSpec(dtype, shape, max_channel)

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
