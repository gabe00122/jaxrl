from jaxrl.config import EnvironmentConfig
from jaxrl.envs.environment import Environment
from jaxrl.envs.memory.n_back import NBackMemory
from jaxrl.envs.memory.return_2d import ReturnEnv


def create_env(env_config: EnvironmentConfig, length: int) -> Environment:
    match env_config.env_type:
        case "nback":
            return NBackMemory(env_config.max_n, env_config.max_value, length)
        case "return":
            return ReturnEnv(env_config)
        case _:
            raise ValueError(f"Unknown environment type: {env_config.type}")
