import json
from pathlib import Path
import random
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class LinearObsEncoderConfig(BaseModel):
    obs_type: Literal["linear"] = "linear"


class TransformerBlockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")    
    num_heads: int
    ffn_size: int
    rope_max_wavelength: float = 10_000
    glu: bool = True
    gtrxl_gate: bool = True
    gtrxl_bias: float = 0.0
    attention_softcap: Optional[float] = None


class TransformerActorCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obs_encoder: LinearObsEncoderConfig = Field(discriminator="obs_type")
    hidden_features: int
    
    transformer_block: TransformerBlockConfig
    num_layers: int

    activation: Literal["relu","gelu", "silu", "mish"]
    norm: Literal["layer_norm", "rms_norm"]
    kernel_init: Literal["glorot_uniform", "he_uniform", "lecun_uniform", "normal"] = "glorot_uniform"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    param_dtype: Literal["float32", "bfloat16"] = "float32"


class LoggerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    log_rate: int = 1000
    use_console: bool = True
    use_tb: bool = False
    use_neptune: bool = False
    use_wandb: bool = False


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["adamw"]
    learning_rate: float
    weight_decay: float
    eps: float
    beta1: float
    beta2: float
    max_norm: float


class MlpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["mlp"]
    layers: list[int]


class CnnLayerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    features: int
    kernel_size: list[int]
    stride: list[int]
    # padding: Literal["valid", "same"]


class CnnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["cnn"]
    layers: list[CnnLayerConfig]
    output_size: int


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    body: MlpConfig | CnnConfig
    activation: Literal["relu", "tanh", "gelu", "silu", "mish"]
    dtype: Literal["float32", "bfloat16"] = "float32"
    param_dtype: Literal["float32", "bfloat16"] = "float32"


class BasicActorCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    discount: float = 0.99
    actor_coefficient: float = 1.0
    critic_coefficient: float = 1.0
    entropy_coefficient: float = 0.0


class PPOConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    learner_type: Literal["ppo"]

    minibatch_count: int = 1
    vf_coef: float = 1.0
    entropy_coef: float = 0.005
    vf_clip: float = 0.2
    discount: float = 0.95
    gae_lambda: float = 0.95


class LearnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer: OptimizerConfig
    model: TransformerActorCriticConfig
    trainer: PPOConfig


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: str
    name: str
    num_envs: int
    max_steps: int


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int | Literal["random"] = "random"
    num_envs: int
    max_env_steps: int

    update_steps: int

    learner: LearnerConfig
    # environment: NBackMemory
    logger: LoggerConfig = LoggerConfig()


def load_config(file: Path) -> Config:
    with open(file) as f:
        json_config = json.load(f)

    config = Config.model_validate(json_config, strict=True)
    if config.seed == "random":
        config.seed = random.getrandbits(31)

    return config
