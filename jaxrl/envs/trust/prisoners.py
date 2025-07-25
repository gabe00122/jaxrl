from functools import cached_property
import jax
from jax import numpy as jnp

from typing import NamedTuple
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep


class PrisonersState(NamedTuple):
    time: jax.Array


class PrisonersEnv(Environment[PrisonersState]):
    def __init__(self) -> None:
        super().__init__()

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            dtype=jnp.float32,
            shape=(3,)
        )

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(2)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return 2

    def reset(self, rng_key: jax.Array) -> tuple[PrisonersState, TimeStep]:
        state = PrisonersState(time=jnp.zeros((), dtype=jnp.int32))
        obs = jnp.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        time = jnp.repeat(state.time[None], 2)

        return state, TimeStep(
            obs,
            time,
            last_action=jnp.zeros((2,), dtype=jnp.int32),
            last_reward=jnp.zeros((2,), dtype=jnp.float32),
            action_mask=None
        )

    def step(self, state: PrisonersState, action: jax.Array, rng_key: jax.Array) -> tuple[PrisonersState, TimeStep]:
        reward_table = jnp.array([
            [-1.0, -3.0],
            [0.0, -2.0],
        ])

        rewards = jnp.array([
            reward_table[action[0], action[1]],
            reward_table[action[1], action[0]],
        ])

        state = state._replace(time=state.time + 1)

        obs = jax.nn.one_hot(jnp.flip(action + 1), num_classes=3)
        time = jnp.repeat(state.time[None], 2)

        return state, TimeStep(
            obs=obs,
            time=time,
            last_action=action,
            last_reward=rewards,
            action_mask=None
        )


class PrisonersRenderer:
    def __init__(self, env: PrisonersEnv):
        pass

    def render(self, state: PrisonersState, timestep: TimeStep):
        print(f"action: {timestep.last_action}, reward: {timestep.last_reward}")
