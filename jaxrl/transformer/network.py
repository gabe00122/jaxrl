import jax
from jax import numpy as jnp
from jax.typing import DTypeLike
from typing import Callable, Literal, Optional, Any

from flax import nnx
import tensorflow_probability.substrates.jax.distributions as tfd
from pydantic import BaseModel, ConfigDict, Field

from jaxrl.networks import DiscreteActionHead, parse_activation_fn
from jaxrl.types import Observation
from jaxrl.transformer.attention import AttentionBlock
from jaxrl.transformer.feed_forward import GLUBlock, FFBlock
from jaxrl.transformer.gate import GatingMechanism


class LinearObsEncoderConfig(BaseModel):
    obs_type: Literal["linear"] = "linear"

class LinearObsEncoder(nnx.Module):
    def __init__(self, config: LinearObsEncoderConfig, obs_dim: int, output_size: int, *, dtype, params_dtype, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(obs_dim, output_size, dtype=dtype, param_dtype=params_dtype, rngs=rngs)

    def __call__(self, x) -> Any:
        return self.linear(x)
    
def create_obs_encoder(config: LinearObsEncoderConfig, obs_dim: int, output_size: int, *, dtype, params_dtype, rngs: nnx.Rngs):
    return LinearObsEncoder(config, obs_dim, output_size, dtype=dtype, params_dtype=params_dtype, rngs=rngs)

class TransformerBlockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")    
    num_heads: int
    ffn_size: int
    max_seq_length: int
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


def get_kernel_init(init_name: str) -> nnx.Initializer:
    """Convert a string kernel initializer name to an actual initializer."""
    match init_name:
        case "glorot_uniform":
            return nnx.initializers.glorot_uniform()
        case "he_uniform":
            return nnx.initializers.he_uniform()
        case "lecun_uniform":
            return nnx.initializers.lecun_uniform()
        case "normal":
            return nnx.initializers.normal()
        case _:
            raise ValueError(f"Unknown kernel initializer: {init_name}")


def get_dtype(dtype_str: str) -> jnp.dtype:
    """Convert a string dtype to an actual dtype."""
    match dtype_str:
        case "float32":
            return jnp.float32
        case "bfloat16":
            return jnp.bfloat16
        case _:
            raise ValueError(f"Unknown dtype: {dtype_str}")

def get_norm(norm_type: str):
    match norm_type:
        case "layer_norm":
            return nnx.LayerNorm
        case "rms_norm":
            return nnx.RMSNorm
        case _:
            raise ValueError(f"Unknown norm type: {norm_type}")


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        config: TransformerBlockConfig,
        hidden_features: int,
        activation: Callable[[jax.Array], jax.Array],
        normalizer,
        kernel_init: nnx.Initializer,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        *,
        rngs: nnx.Rngs,
    ):
        # Extract parameters from config
        activation = activation
        hidden_features = hidden_features

        num_heads = config.num_heads
        ffn_size = config.ffn_size
        max_seq_length = config.max_seq_length
        glu = config.glu
        gtrxl_gate = config.gtrxl_gate
        gtrxl_bias = config.gtrxl_bias
        attention_softcap = config.attention_softcap
        rope_max_wavelength = config.rope_max_wavelength
        
        self.gtrxl_gate = gtrxl_gate

        self.attention_norm = normalizer(
            num_features=hidden_features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention = AttentionBlock(
            num_heads,
            hidden_features,
            max_seq_length=max_seq_length,
            rope_max_wavelength=rope_max_wavelength,
            attention_softcap=attention_softcap,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )

        self.ffn_norm = normalizer(
            num_features=hidden_features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        ff_block = GLUBlock if glu else FFBlock
        self.ffn = ff_block(
            hidden_features,
            ffn_size,
            activation=activation,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if gtrxl_gate:
            self.attention_gate = GatingMechanism(
                hidden_features, gtrxl_bias, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
            self.ffn_gate = GatingMechanism(
                hidden_features, gtrxl_bias, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

    def create_kv_cache(
        self, batch_size: int, context_size: int, *, dtype: DTypeLike | None = None
    ):
        self.attention.create_kv_cache(batch_size, context_size, dtype=dtype)

    def __call__(self, x, time_steps, use_kv_cache: bool):
        attention_input = self.attention_norm(x)
        attention = self.attention(
            attention_input, time_steps, use_kv_cache=use_kv_cache
        )
        x = self.attention_gate(x, attention) if self.gtrxl_gate else x + attention

        feed_forward_input = self.ffn_norm(x)
        feed_forward = self.ffn(feed_forward_input)
        x = self.ffn_gate(x, feed_forward) if self.gtrxl_gate else x + feed_forward

        return x


class TransformerActorCritic(nnx.Module):
    def __init__(
        self,
        config: TransformerActorCriticConfig,
        obs_dim: int,
        action_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        # Extract parameters from config
        transformer_config = config.transformer_block
        num_layers = config.num_layers
        
        hidden_features = config.hidden_features
        
        # Convert string representations to actual objects if needed
        kernel_init = get_kernel_init(config.kernel_init)
        activation = parse_activation_fn(config.activation)
        dtype = get_dtype(config.dtype)
        param_dtype = get_dtype(config.param_dtype)
        norm = get_norm(config.norm)
        
        self.obs_encoder = create_obs_encoder(config=config.obs_encoder, obs_dim=obs_dim, output_size=hidden_features, dtype=dtype, params_dtype=param_dtype, rngs=rngs)

        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerBlock(
                    config=transformer_config,
                    hidden_features=hidden_features,
                    activation=activation,
                    normalizer=norm,
                    kernel_init=kernel_init,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
        self.layers = tuple(layers)
        self.output_norm = norm(num_features=hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        self.value_head = nnx.Linear(
            hidden_features,
            1,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.action_head = DiscreteActionHead(hidden_features, action_dim=action_dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def create_kv_cache(
        self, batch_size: int, context_size: int, *, dtype: DTypeLike | None = None
    ):
        for layer in self.layers:
            layer.create_kv_cache(batch_size, context_size, dtype=dtype)

    def __call__(self, observation: Observation, use_kv_cache: bool) -> tuple[jax.Array, tfd.Distribution]:
        x = self.obs_encoder(observation.agents_view)
        # todo create an action emender
        # todo create a value embedder

        for layer in self.layers:
            x = layer(x, observation.time_steps, use_kv_cache)

        x = self.output_norm(x)

        action_logits = self.action_head(x, observation.action_mask)
        value = self.value_head(x)

        return value, action_logits
