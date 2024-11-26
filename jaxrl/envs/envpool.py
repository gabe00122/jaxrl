import envpool
import dm_env

from jax import Array

from jaxrl.envs.wrapper import EnvWrapper, StepType, TimeStep
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
        timestep = self.env.step(action)
        return None, _convert_timestep(timestep)
    
    def observation_spec(self):
        self.env.observation_spec()
        return ObservationSpec(shape=self.env.observation_spec().shape)
    
    def action_spec(self):
        return ActionSpec(
            shape=self.env.action_spec().shape,
            discrete=self.env.action_spec().discrete,
        )


def _convert_timestep(dm_time_step: dm_env.TimeStep) -> TimeStep:
    return TimeStep(
        StepType(dm_time_step.step_type),
        Observation(dm_time_step.observation, None),
        dm_time_step.reward,
    )


def main():
    env = envpool.make_gymnasium("Pong-v5", num_envs=4)
    # obs, info = env.reset()



if __name__ == "__main__":
    main()
    # compare()
