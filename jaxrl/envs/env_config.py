from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jaxrl.envs.gridworld.king_hill import KingHillConfig, KingHillEnv
from jaxrl.envs.gridworld.grid_return import ReturnDiggingConfig, ReturnDiggingEnv
from jaxrl.envs.gridworld.traveling_salesman import TravelingSalesmanConfig, TravelingSalesmanEnv
from jaxrl.envs.gridworld.scouts import ScoutsConfig, ScoutsEnv
from jaxrl.envs.gridworld.renderer import GridworldClient

from jaxrl.envs.sequence.n_back import NBackConfig, NBackMemory

from jaxrl.envs.third_party.craftax_wrapper import CraftaxConfig, CraftaxEnvironment

from jaxrl.envs.client import EnvironmentClient
from jaxrl.envs.environment import Environment
from jaxrl.envs.task_id_wrapper import TaskIdWrapper
from jaxrl.envs.multitask import MultiTaskWrapper
from jaxrl.envs.vector import VectorWrapper



class PrisonersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["prisoners"] = "prisoners"


type EnvironmentConfig = (
    NBackConfig
    | ReturnDiggingConfig
    | TravelingSalesmanConfig
    | ScoutsConfig
    | KingHillConfig
    | PrisonersConfig
    | CraftaxConfig
)


class MultiTaskEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    num: int = 1
    name: str
    env: EnvironmentConfig = Field(discriminator="env_type")


class MultiTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["multi"] = "multi"
    envs: tuple[MultiTaskEnvConfig, ...]

    @field_validator("envs", mode="before")
    @classmethod
    def coerce_envs(cls, v):
        # JSON gives list; accept list and turn into tuple
        return tuple(v) if isinstance(v, list) else v


def create_env(
    env_config: EnvironmentConfig | MultiTaskConfig,
    length: int,
    vec_count: int = 1,
    env_name: str | None = None,
) -> tuple[Environment, int]:
    num_tasks = 1
    if env_config.env_type == "multi" and env_name is not None:
        num_tasks = len(env_config.envs)
        for task_id, env_def in enumerate(env_config.envs):
            if env_def.name == env_name:
                return TaskIdWrapper(create_env(env_def.env, length, vec_count=vec_count)[0], task_id), num_tasks
        raise ValueError("Could not find environment matching env_name")

    match env_config.env_type:
        case "multi":
            out_envs = []
            out_env_names = []
            num_tasks = len(env_config.env_type)
            for env_def in env_config.envs:
                out_envs.append(create_env(env_def.env, length, env_def.num)[0])
                out_env_names.append(env_def.name)

            env = MultiTaskWrapper(tuple(out_envs), tuple(out_env_names))
        case "nback":
            env = NBackMemory(env_config.max_n, env_config.max_value, length)
        case "return_digging":
            env = ReturnDiggingEnv(env_config, length)
        case "scouts":
            env = ScoutsEnv(env_config, length)
        case "traveling_salesman":
            env = TravelingSalesmanEnv(env_config, length)
        case "king_hill":
            env = KingHillEnv(env_config, length)
        case "craftax":
            env = CraftaxEnvironment()
        case _:
            raise ValueError(f"Unknown environment type: {env_config.env_type}")

    if vec_count > 1:
        env = VectorWrapper(env, vec_count)

    return env, num_tasks


def create_client[State](env: Environment[State]) -> EnvironmentClient[State]:
    # use the grid client for all environments for now
    return GridworldClient(env)
