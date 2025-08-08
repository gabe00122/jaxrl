from jaxrl.config import EnvironmentConfig
from jaxrl.envs.client import EnvironmentClient
from jaxrl.envs.environment import Environment
from jaxrl.envs.memory.n_back import NBackMemory
from jaxrl.envs.memory.return_2d import ReturnClient, ReturnEnv
from jaxrl.envs.memory.return_2d_colors import ReturnColorClient, ReturnColorEnv
from jaxrl.envs.memory.return_2d_digging import ReturnDiggingClient, ReturnDiggingEnv
from jaxrl.envs.memory.scouts import ScoutsClient, ScoutsEnv
from jaxrl.envs.trust.prisoners import PrisonersEnv


def create_env(env_config: EnvironmentConfig, length: int) -> Environment:
    match env_config.env_type:
        case "nback":
            return NBackMemory(env_config.max_n, env_config.max_value, length)
        case "return":
            return ReturnEnv(env_config)
        case "return_color":
            return ReturnColorEnv(env_config)
        case "return_digging":
            return ReturnDiggingEnv(env_config)
        case "prisoners":
            return PrisonersEnv()
        case "scouts":
            return ScoutsEnv(env_config)
        case _:
            raise ValueError(f"Unknown environment type: {env_config.type}")


def create_client[State](env: Environment[State]) -> EnvironmentClient[State]:
        if isinstance(env, ReturnEnv):
            return ReturnClient(env)
        elif isinstance(env, ReturnColorEnv):
            return ReturnColorClient(env)
        elif isinstance(env, ReturnDiggingEnv):
            return ReturnDiggingClient(env)
        elif isinstance(env, ScoutsEnv):
            return ScoutsClient(env)
