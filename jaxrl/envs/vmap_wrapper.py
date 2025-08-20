from functools import cached_property
import jax
from einops import rearrange

from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import ActionSpec, ObservationSpec
from jaxrl.types import TimeStep


class VmapWrapper[EnvState](Environment[EnvState]):
    def __init__(self, env: Environment[EnvState], vec_count: int):
        self.base_env = env
        self._vec_count = vec_count

    def reset(self, rng_key: jax.Array) -> tuple[EnvState, TimeStep]:
        rng_keys = jax.random.split(rng_key, self._vec_count)
        state, timestep = jax.vmap(self.base_env.reset)(rng_keys)
        return state, self._flatten_timestep(timestep)

    def step(
        self, state: EnvState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[EnvState, TimeStep]:
        rng_keys = jax.random.split(rng_key, self._vec_count)

        action = action.reshape(self._vec_count, self.base_env.num_agents)
        state, timestep = jax.vmap(
            self.base_env.step, in_axes=(0, 0, 0), out_axes=(0, 0)
        )(state, action, rng_keys)
        return state, self._flatten_timestep(timestep)

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return self.base_env.observation_spec

    @cached_property
    def action_spec(self) -> ActionSpec:
        return self.base_env.action_spec

    @property
    def is_jittable(self) -> bool:
        return self.base_env.is_jittable

    @property
    def num_agents(self) -> int:
        return self._vec_count * self.base_env.num_agents

    def _flatten_timestep(self, timestep: TimeStep) -> TimeStep:
        return jax.tree_util.tree_map(
            lambda x: rearrange(x, "b a ... -> (b a) ...") if x is not None else None,
            timestep,
        )
        # return TimeStep(
        #     action_mask=(
        #         rearrange(timestep.action_mask, "b a ... -> (b a) ...")
        #         if timestep.action_mask is not None
        #         else None
        #     ),
        #     obs=rearrange(timestep.obs, "b a ... -> (b a) ..."),
        #     time=timestep.time.reshape(self.num_agents),
        #     last_action=timestep.last_action.reshape(self.num_agents),
        #     last_reward=timestep.last_reward.reshape(self.num_agents),
        # )
