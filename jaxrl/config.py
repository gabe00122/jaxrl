import json
from pathlib import Path
import random
from typing import Literal
from pydantic import BaseModel


class LoggerConfig(BaseModel):
    use_console: bool = True
    use_tb: bool = False
    use_neptune: bool = False


class OptimizerConfig(BaseModel):
    type: Literal["adamw"]
    learning_rate: float
    warmup_steps: int
    weight_decay: float
    eps: float
    beta1: float
    beta2: float


class ModelConfig(BaseModel):
    hidden_size: list[int]
    activation: Literal["relu", "tanh", "gelu", "silu", "mish"]
    dtype: Literal["float32", "bfloat16"] = "float32"
    param_dtype: Literal["float32", "bfloat16"] = "float32"


class LearnerConfig(BaseModel):
    optimizer: OptimizerConfig
    model: ModelConfig
    discount: float = 0.99
    actor_coefficient: float = 1.0
    critic_coefficient: float = 1.0
    entropy_coefficient: float = 0.0


class EnvironmentConfig(BaseModel):
    backend: str
    name: str
    num_envs: int
    max_steps: int


class Config(BaseModel):
    seed: int | Literal["random"] = "random"
    learner: LearnerConfig
    environment: EnvironmentConfig
    logger: LoggerConfig = LoggerConfig()


def load_config(file: Path) -> Config:
    with open(file) as f:
        json_config = json.load(f)

    config = Config.model_validate(json_config)
    if config.seed == "random":
        config.seed = random.getrandbits(31)

    return config
