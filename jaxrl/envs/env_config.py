from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jaxrl.envs.gridworld.king_hill import KingHillConfig, KingHillEnv
from jaxrl.envs.sequence.n_back import NBackConfig
from jaxrl.envs.gridworld.grid_return import ReturnDiggingConfig
from jaxrl.envs.gridworld.explore import ExploreConfig
from jaxrl.envs.gridworld.traveling_salesman import TravelingSalesmanConfig
from jaxrl.envs.gridworld.scouts import ScoutsConfig
from jaxrl.envs.third_party.craftax_wrapper import CraftaxConfig
from jaxrl.envs.client import EnvironmentClient
from jaxrl.envs.third_party.craftax_wrapper import CraftaxEnvironment
from jaxrl.envs.environment import Environment
from jaxrl.envs.gridworld.explore import ExploreEnv
from jaxrl.envs.sequence.n_back import NBackMemory
from jaxrl.envs.gridworld.grid_return import ReturnDiggingEnv
from jaxrl.envs.gridworld.scouts import ScoutsEnv
from jaxrl.envs.gridworld.renderer import GridworldClient
from jaxrl.envs.gridworld.traveling_salesman import TravelingSalesmanEnv
from jaxrl.envs.multitask import MultiTaskWrapper
from jaxrl.envs.vector import VectorWrapper
from jaxrl.envs.task_id import TaskIdWrapper


class PrisonersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["prisoners"] = "prisoners"


type EnvironmentConfig = (
    NBackConfig
    | ReturnDiggingConfig
    | ExploreConfig
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
    selector: str | None = None,
) -> Environment:
    if env_config.env_type == "multi" and selector is not None:
        for env_def in env_config.envs:
            if env_def.name == selector:
                return create_env(env_def.env, length, vec_count=vec_count)
        raise ValueError("Could not find environment for selector")

    match env_config.env_type:
        case "multi":
            out_envs = []
            out_env_names = []
            for idx, env_def in enumerate(env_config.envs):
                base_env = create_env(env_def.env, length, vec_count=env_def.num)
                base_env = TaskIdWrapper(base_env, idx)
                out_envs.append(base_env)
                out_env_names.append(env_def.name)

            env = MultiTaskWrapper(tuple(out_envs), tuple(out_env_names))
        case "nback":
            env = NBackMemory(env_config.max_n, env_config.max_value, length)
        case "return_digging":
            env = ReturnDiggingEnv(env_config, length)
        case "scouts":
            env = ScoutsEnv(env_config, length)
        case "explore":
            env = ExploreEnv(env_config, length)
        case "traveling_salesman":
            env = TravelingSalesmanEnv(env_config, length)
        case "king_hill":
            env = KingHillEnv(env_config, length)
        case "craftax":
            env = CraftaxEnvironment()
        case _:
            raise ValueError(f"Unknown environment type: {env_config.env_type}")

    if env_config.env_type != "multi" and vec_count > 1:
        env = VectorWrapper(env, vec_count)

    return env


def create_client[State](env: Environment[State]) -> EnvironmentClient[State]:
    # use the grid client for all environments for now
    return GridworldClient(env)
