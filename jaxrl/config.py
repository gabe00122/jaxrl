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
    hidden_size: int
    num_layers: int
    num_actions: int
    continuous: bool


class LearnerConfig(BaseModel):
    optimizer: OptimizerConfig


class Config(BaseModel):
    seed: int | Literal["random"] = "random"
    learner: LearnerConfig
    logger: LoggerConfig = LoggerConfig()


def load_config(file: Path) -> Config:
    with open(file) as f:
        json_config = json.load(f)

    config = Config.model_validate(json_config)
    if config.seed == "random":
        config.seed = random.getrandbits(32)

    return config
