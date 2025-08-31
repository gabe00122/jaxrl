from jaxrl.config import EnvironmentConfig, MultiTaskConfig
from jaxrl.envs.client import EnvironmentClient
from jaxrl.envs.craftax_wrapper import CraftaxEnvironment
from jaxrl.envs.environment import Environment
from jaxrl.envs.memory.n_back import NBackMemory
from jaxrl.envs.memory.return_2d_digging import ReturnDiggingEnv
from jaxrl.envs.memory.scouts import ScoutsEnv
from jaxrl.envs.gridworld.renderer import GridworldClient
from jaxrl.envs.multitask import MultiTaskWrapper
from jaxrl.envs.trust.prisoners import PrisonersEnv
from jaxrl.envs.vector import VectorWrapper


def create_env(env_config: EnvironmentConfig | MultiTaskConfig, length: int, vec_count: int = 1, selector: str | None = None) -> Environment:
    if env_config.env_type == "multi" and selector is not None:
        for env_def in env_config.envs:
            if env_def.name == selector:
                return create_env(env_def.env, length, vec_count=vec_count)
        raise ValueError("Could not find environment for selector")


    match env_config.env_type:
        case "multi":
            out_envs = []
            out_env_names = []
            for env_def in env_config.envs:
                out_envs.append(create_env(env_def.env, length, env_def.num))
                out_env_names.append(env_def.name)

            env = MultiTaskWrapper(tuple(out_envs), tuple(out_env_names))
        case "nback":
            env = NBackMemory(env_config.max_n, env_config.max_value, length)
        case "return_digging":
            env = ReturnDiggingEnv(env_config, length)
        case "prisoners":
            env = PrisonersEnv()
        case "scouts":
            env = ScoutsEnv(env_config, length)
        case "craftax":
            env = CraftaxEnvironment()
        case _:
            raise ValueError(f"Unknown environment type: {env_config.env_type}")
    
    if vec_count > 1:
        env = VectorWrapper(env, vec_count)
    
    return env


def create_client[State](env: Environment[State]) -> EnvironmentClient[State]:
        if isinstance(env, (ReturnEnv, ReturnDiggingEnv, ScoutsEnv)):
            # Use the shared gridworld renderer for all gridworld envs
            return GridworldClient(env)
