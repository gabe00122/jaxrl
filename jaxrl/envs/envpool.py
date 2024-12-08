from functools import cached_property
import dm_env.specs
import envpool
import dm_env

from jax import Array
import numpy as np

from jaxrl.envs.specs import (
    ActionSpec,
    ContinuousActionSpec,
    DiscreteActionSpec,
    ObservationSpec,
)
from jaxrl.envs.wrapper import EnvWrapper, TimeStep
from jaxrl.types import Observation


class EnvPoolWrapper(EnvWrapper[None]):
    def __init__(self, env_name: str, num_envs: int, seed: int) -> None:
        self._num_envs = num_envs
        self.env: dm_env.Environment = envpool.make_dm(
            task_id=env_name,
            num_envs=num_envs,
            seed=seed,
        )

    @property
    def is_jittable(self) -> bool:
        return False

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def players(self) -> int:
        return 1

    def reset(self) -> tuple[None, TimeStep]:
        timestep = self.env.reset()

        return None, _convert_timestep(timestep)

    def step(self, state, action: Array) -> tuple[None, TimeStep]:
        timestep = self.env.step(np.asarray(action))

        # gymnasium autoreset
        # done = timestep.step_type == dm_env.StepType.LAST
        # env_ids_to_reset = np.where(done)[0]
        # if len(env_ids_to_reset) > 0:
        #     reset_time_step = self.env.step(np.zeros_like(action), env_ids_to_reset)
        #     timestep.observation.obs[env_ids_to_reset] = reset_time_step.observation.obs

        return None, _convert_timestep(timestep)

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        obs = self.env.observation_spec().obs
        return ObservationSpec(obs.shape)

    @cached_property
    def action_spec(self) -> ActionSpec:
        spec = self.env.action_spec()
        discrete = isinstance(spec, dm_env.specs.DiscreteArray)

        if discrete:
            return DiscreteActionSpec(spec.num_values)
        else:
            return ContinuousActionSpec(spec.shape)


def _convert_timestep(dm_time_step: dm_env.TimeStep) -> TimeStep:
    return TimeStep(
        dm_time_step.step_type,
        Observation(dm_time_step.observation.obs, None),
        dm_time_step.reward,
    )
