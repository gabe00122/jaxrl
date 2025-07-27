import json
import random
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class NBackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["nback"] = "nback"

    max_n: int = 12
    max_value: int = 2


class ReturnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["return"] = "return"

    num_agents: int = 1

    width: int = 40
    height: int = 40
    view_width: int = 5
    view_height: int = 5


class ReturnColorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["return_color"] = "return_color"

    num_agents: int = 1
    num_colors: int = 4

    width: int = 40
    height: int = 40
    view_width: int = 5
    view_height: int = 5


class ReturnDiggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["return_digging"] = "return_digging"

    num_agents: int = 1

    width: int = 40
    height: int = 40
    view_width: int = 5
    view_height: int = 5

    mapgen_threshold: float = 0.3
    digging_timeout: int = 5


class PrisonersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["prisoners"] = "prisoners"


class GridCnnObsEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    obs_type: Literal["grid_cnn"] = "grid_cnn"


class LinearObsEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    obs_type: Literal["linear"] = "linear"


class TransformerBlockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ffn_size: int
    glu: bool = True

    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int | None = None

    attention_impl: str = "xla"

    rope_max_wavelength: float = 10_000
    use_post_attn_norm: bool = False
    use_post_ffw_norm: bool = False
    use_qk_norm: bool = False

    gtrxl_gate: bool = False
    gtrxl_bias: float = 0.0


class TransformerActorCriticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    obs_encoder: LinearObsEncoderConfig | GridCnnObsEncoderConfig = Field(
        discriminator="obs_type"
    )
    hidden_features: int

    transformer_block: TransformerBlockConfig
    num_layers: int

    activation: Literal["relu", "gelu", "silu", "mish"]
    norm: Literal["layer_norm", "rms_norm"]
    kernel_init: Literal["glorot_uniform", "he_uniform", "lecun_uniform", "normal"] = (
        "glorot_uniform"
    )
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

    type: Literal["adamw", "moun"]
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

    trainer_type: Literal["ppo"]

    epoch_count: int = 1
    minibatch_count: int = 1
    vf_coef: float = 1.0
    entropy_coef: float = 0.005
    vf_clip: float = 0.2
    discount: float = 0.95
    gae_lambda: float = 0.95
    normalize_advantage: bool = False


class LearnerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer: OptimizerConfig
    model: TransformerActorCriticConfig
    trainer: PPOConfig


type EnvironmentConfig = NBackConfig | ReturnConfig | ReturnColorConfig | ReturnDiggingConfig | PrisonersConfig


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int | Literal["random"] = "random"
    num_envs: int
    max_env_steps: int

    updates_per_jit: int = 1
    update_steps: int

    learner: LearnerConfig
    environment: EnvironmentConfig = Field(discriminator="env_type")
    logger: LoggerConfig = LoggerConfig()


def load_config(json_config: str) -> Config:
    config = Config.model_validate(json.loads(json_config), strict=True)
    if config.seed == "random":
        config = config.model_copy(update={"seed": random.getrandbits(31)})

    return config
