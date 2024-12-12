import json
from pathlib import Path
import random
from typing import Literal
from pydantic import BaseModel, ConfigDict


class LoggerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    log_rate: int = 1000
    use_console: bool = True
    use_tb: bool = False
    use_neptune: bool = False
    use_wandb: bool = False


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal["adamw"]
    learning_rate: float
    weight_decay: float
    eps: float
    beta1: float
    beta2: float


class MlpConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal["mlp"]
    layers: list[int]


class CnnLayerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    features: int
    kernel_size: list[int]
    stride: list[int]
    # padding: Literal["valid", "same"]


class CnnConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal["cnn"]
    layers: list[CnnLayerConfig]
    output_size: int


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    body: MlpConfig | CnnConfig
    activation: Literal["relu", "tanh", "gelu", "silu", "mish"]
    dtype: Literal["float32", "bfloat16"] = "float32"
    param_dtype: Literal["float32", "bfloat16"] = "float32"


class LearnerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    optimizer: OptimizerConfig
    model: ModelConfig
    discount: float = 0.99
    actor_coefficient: float = 1.0
    critic_coefficient: float = 1.0
    entropy_coefficient: float = 0.0


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    backend: str
    name: str
    num_envs: int
    max_steps: int


class Config(BaseModel):
    model_config = ConfigDict(extra='forbid')

    seed: int | Literal["random"] = "random"
    learner: LearnerConfig
    environment: EnvironmentConfig
    logger: LoggerConfig = LoggerConfig()


def load_config(file: Path) -> Config:
    with open(file) as f:
        json_config = json.load(f)

    config = Config.model_validate(json_config, strict=True)
    if config.seed == "random":
        config.seed = random.getrandbits(31)

    return config
