from functools import cached_property
from typing import Any, NamedTuple

import jax
from jax import numpy as jnp

from craftax.craftax_env import make_craftax_env_from_name

from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep


PREPROCESS_SHAPE = (65, 55, 1)


class CraftaxWrapperState(NamedTuple):
    cstate: Any
    time: jax.Array
    rewards: jax.Array


def rgb2gray(rgb):
    return jnp.dot(rgb[..., :3], jnp.array([0.2989, 0.5870, 0.1140]))[..., None]


class CraftaxEnvironment(Environment[CraftaxWrapperState]):
    def __init__(self) -> None:
        super().__init__()

        self._symbolic = True

        self._env = make_craftax_env_from_name(
            "Craftax-Symbolic-v1" if self._symbolic else "Craftax-Pixels-v1",
            auto_reset=True,
        )
        self._env_params = self._env.default_params

        self._n_actions = self._env.action_space().n
        self._n_obs = self._env.observation_space(self._env_params).shape

    def reset(self, rng_key):
        obs, cstate = self._env.reset(rng_key, self._env_params)

        actions = jnp.zeros(1, dtype=jnp.int32)
        rewards = jnp.zeros(1)
        time = jnp.zeros(1, dtype=jnp.int32)

        state = CraftaxWrapperState(cstate, time, jnp.float32(0.0))

        return state, self._encode_timestep(
            obs, jnp.array(False, dtype=jnp.bool), actions, rewards, time
        )

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            shape=self._n_obs if self._symbolic else PREPROCESS_SHAPE,
            dtype=jnp.float32,
        )

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(num_actions=self._n_actions)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return 1

    def step(
        self, state, action: jax.Array, rng_key: jax.Array
    ) -> tuple[Any, TimeStep]:
        obs, cstate, reward, done, info = self._env.step(
            rng_key, state.cstate, action.squeeze(-1), self._env_params
        )

        state = state._replace(
            cstate=cstate,
            time=state.time + 1,
            rewards=state.rewards + jnp.squeeze(reward),
        )

        return state, self._encode_timestep(obs, done, action, reward[None], state.time)

    def _encode_timestep(self, obs, terminated, actions, rewards, time):
        if not self._symbolic:
            obs = jax.image.resize(obs, (65, 55, 3), jax.image.ResizeMethod.LINEAR)
            obs = rgb2gray(obs)
        obs = obs[None, ...]

        return TimeStep(
            obs=obs,
            time=time,
            terminated=terminated[None],
            last_action=actions,
            last_reward=rewards,
            action_mask=None,
        )

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: CraftaxWrapperState):
        return {"rewards": state.rewards}


def main():
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    _rngs = jax.random.split(_rng, 3)

    env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=True)
    env_params = env.default_params
    print(env.action_space().n)

    # Get an initial state and observation
    obs, state = env.reset(_rngs[0], env_params)
    print(obs.dtype)

    # Pick random action
    action = env.action_space(env_params).sample(_rngs[1])

    # Step environment
    obs, state, reward, done, info = env.step(_rngs[2], state, action, env_params)


if __name__ == "__main__":
    main()
