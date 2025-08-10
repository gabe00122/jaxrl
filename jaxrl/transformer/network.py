from einops import rearrange
import jax
from jax import numpy as jnp
from jax.typing import DTypeLike
from typing import Callable, Any

from flax import nnx
import tensorflow_probability.substrates.jax.distributions as tfd

from jaxrl import hl_gauss
from jaxrl.config import (
    GridCnnObsEncoderConfig,
    LinearObsEncoderConfig,
    TransformerActorCriticConfig,
    TransformerBlockConfig,
)
from jaxrl.distributions import IdentityTransformation
from jaxrl.envs.specs import ObservationSpec
from jaxrl.hl_gauss import HlGaussConfig, calculate_supports, transform_from_probs
from jaxrl.transformer.observation import GridCnnObsDecoder, create_obs_encoder
from jaxrl.types import TimeStep
from jaxrl.transformer.attention import AttentionBlock, KVCache
from jaxrl.transformer.feed_forward import GLUBlock, FFBlock
from jaxrl.utils.preturb import preturb


def parse_activation_fn(activation_name: str) -> Callable[[jax.Array], jax.Array]:
    match activation_name:
        case "relu":
            return jax.nn.relu
        case "mish":
            return jax.nn.mish
        case "gelu":
            return jax.nn.gelu
        case "silu":
            return jax.nn.silu
        case "tanh":
            return jax.nn.tanh
        case _:
            raise ValueError(f"Activation function {activation_name} not recognized")


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
        max_seq_length: int,
        kernel_init: nnx.Initializer,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        *,
        rngs: nnx.Rngs,
    ):
        head_dim = config.head_dim
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads

        ffn_size = config.ffn_size
        glu = config.glu
        rope_max_wavelength = config.rope_max_wavelength

        self.use_post_attn_norm = config.use_post_attn_norm
        self.use_post_ffw_norm = config.use_post_ffw_norm

        self.attention_norm = normalizer(
            num_features=hidden_features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention = AttentionBlock(
            hidden_features,
            head_dim,
            num_heads,
            num_kv_heads,
            max_seq_length=max_seq_length if config.sliding_window is None else config.sliding_window,
            rope_max_wavelength=rope_max_wavelength,
            dtype=dtype,
            param_dtype=param_dtype,
            attention_impl=config.attention_impl,
            rngs=rngs,
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

        if self.use_post_attn_norm:
            self.post_attn_norm = normalizer(
                num_features=hidden_features,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        if self.use_post_ffw_norm:
            self.post_ffw_norm = normalizer(
                num_features=hidden_features,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def create_kv_cache(self, batch_size: int) -> KVCache:
        return self.attention.create_kv_cache(batch_size)

    def __call__(
        self, x, time_steps, kv_cache: KVCache | None = None
    ) -> tuple[jax.Array, KVCache | None]:
        attention_input = self.attention_norm(x)
        attention_output, kv_cache = self.attention(attention_input, time_steps, kv_cache)
        if self.use_post_attn_norm:
            attention_output = self.post_attn_norm(attention_output)
        x = x + attention_output

        feed_forward_input = self.ffn_norm(x)
        feed_forward_output = self.ffn(feed_forward_input)
        if self.use_post_ffw_norm:
            feed_forward_output = self.post_ffw_norm(feed_forward_output)
        x = x + feed_forward_output

        return x, kv_cache


class Embedder(nnx.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_features: int,
        *,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ):
        self.dtype = dtype
        self.embedding_features = embedding_features
        self.param_dtype = param_dtype

        key = rngs.param()
        self.embedding_table = nnx.Param(
            jax.random.normal(key, (vocab_size, embedding_features)) * 0.01,
            dtype=param_dtype,
        )

    def encode(self, x: jax.Array):
        x = jnp.take(self.embedding_table.value, x, axis=0, fill_value=0)

        x = jnp.asarray(x, dtype=self.dtype)
        x *= jnp.sqrt(self.embedding_features).astype(self.dtype)
        return x

    def decode(self, x: jax.Array):
        return jnp.dot(x, self.embedding_table.value.T)


class TransformerActorCritic(nnx.Module):
    def __init__(
        self,
        config: TransformerActorCriticConfig,
        obs_spec: ObservationSpec,
        action_dim: int,
        hl_gauss: HlGaussConfig,
        max_seq_length: int,
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

        self.dtype = config.dtype
        self.hl_gauss = hl_gauss

        self.reward_encoder = nnx.Linear(
            1, hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.action_embedder = Embedder(
            action_dim, hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.obs_encoder = create_obs_encoder(
            config=config.obs_encoder,
            obs_spec=obs_spec,
            output_size=hidden_features,
            dtype=dtype,
            params_dtype=param_dtype,
            rngs=rngs,
        )

        ## temp
        self.obs_decoder = GridCnnObsDecoder(
            config=config.obs_encoder,
            obs_spec=obs_spec,
            output_size=hidden_features,
            dtype=dtype,
            params_dtype=param_dtype,
            rngs=rngs,
        )
        self.transition = FFBlock(
            hidden_features,
            hidden_features,
            activation,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs
        )
        ## temp

        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerBlock(
                    config=transformer_config,
                    hidden_features=hidden_features,
                    activation=activation,
                    normalizer=norm,
                    max_seq_length=max_seq_length,
                    kernel_init=kernel_init,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
        self.layers = tuple(layers)
        self.output_norm = norm(
            num_features=hidden_features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if config.value_hidden_dim is not None:
            self.value_mlp = nnx.Linear(
                hidden_features,
                config.value_hidden_dim,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        self.value_head = nnx.Linear(
            hidden_features if config.value_hidden_dim is None else config.value_hidden_dim,
            hl_gauss.n_logits,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def create_kv_cache(self, batch_size: int) -> tuple[KVCache, ...]:
        return tuple(layer.create_kv_cache(batch_size) for layer in self.layers)

    def __call__(
        self, ts: TimeStep, kv_cache: tuple[KVCache, ...] | None = None, actions: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, tfd.Distribution, tuple[KVCache, ...] | None, jax.Array]:
        obs_embedding = self.obs_encoder(ts.obs)
        reward_embedding = self.reward_encoder(ts.last_reward[..., None])
        action_embedding = self.action_embedder.encode(ts.last_action)

        x = obs_embedding + reward_embedding + action_embedding

        if kv_cache is not None:
            out_kv_cache = []
            for layer, _kv_cache in zip(self.layers, kv_cache):
                x, _kv_cache = layer(x, ts.time, _kv_cache)
                out_kv_cache.append(_kv_cache)
            kv_cache = tuple(out_kv_cache)
        else:
            for layer in self.layers:
                x, _ = layer(x, ts.time)

        x = self.output_norm(x)

        action_logits = self.action_embedder.decode(x)
        if ts.action_mask is not None:
            action_logits = jnp.where(
                ts.action_mask,
                action_logits,
                jnp.finfo(action_logits.dtype).min,
            )

        action_logits = action_logits.astype(jnp.float32)

        policy = IdentityTransformation(
            distribution=tfd.Categorical(logits=action_logits)
        )

        prevalue = x
        if hasattr(self, 'value_mlp'):
            prevalue = self.value_mlp(prevalue)
            prevalue = nnx.gelu(prevalue)

        value = self.value_head(prevalue)
        value_logits = value.astype(jnp.float32)

        b, t, _ = value_logits.shape

        value_probs = nnx.softmax(value_logits, axis=-1)
        
        _, centers = calculate_supports(self.hl_gauss, b * t)

        value_probs = rearrange(value_probs, "b t v -> (b t) v")
        values = transform_from_probs(centers, value_probs)
        values = rearrange(values, "(b t) -> b t", b=b, t=t)

        predicted_obs = None
        if actions is not None:
            predicted_obs = self.obs_decoder(self.transition(x + self.action_embedder.encode(actions)))


        return values, value_logits, policy, kv_cache, predicted_obs

