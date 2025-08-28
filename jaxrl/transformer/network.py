import jax
from jax import numpy as jnp
from jax.typing import DTypeLike
from typing import Callable

from flax import nnx
import tensorflow_probability.substrates.jax.distributions as tfd

from jaxrl.config import (
    LayerConfig,
    TransformerActorCriticConfig,
)
from jaxrl.distributions import IdentityTransformation
from jaxrl.envs.specs import ObservationSpec
from jaxrl.transformer.observation import create_obs_encoder
from jaxrl.types import TimeStep
from jaxrl.transformer.attention import AttentionBlock, KVCache
from jaxrl.transformer.rnn import RnnBlock
from jaxrl.transformer.feed_forward import GLUBlock, FFBlock
from jaxrl.values import HlGaussValue, MseValue


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
        config: LayerConfig,
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
        self.use_post_attn_norm = config.use_post_attn_norm
        self.use_post_ffw_norm = config.use_post_ffw_norm

        self.history_norm = normalizer(
            num_features=hidden_features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        if config.history.type == "rnn":
            self.history = RnnBlock(hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        else:
            self.history = AttentionBlock(
                hidden_features,
                config.history.head_dim,
                config.history.num_heads,
                config.history.num_kv_heads,
                max_seq_length=max_seq_length,
                rope_max_wavelength=config.history.rope_max_wavelength,
                attention_impl=config.history.attention_impl,
                use_qk_norm=config.history.use_qk_norm,
                dtype=dtype,
                param_dtype=param_dtype,
                kernel_init=kernel_init,
                rngs=rngs
            )

        self.ffn_norm = normalizer(
            num_features=hidden_features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        ff_block = GLUBlock if config.feed_forward.glu else FFBlock
        self.ffn = ff_block(
            hidden_features,
            config.feed_forward.size,
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

    def initialize_carry(self, batch_size: int, rngs) -> KVCache:
        return self.history.initialize_carry(batch_size, rngs)

    def __call__(self, x, time_steps, carry = None) -> tuple[jax.Array, KVCache | None]:
        history_input = self.history_norm(x)
        history_output, carry = self.history(history_input, time_steps, carry)
        if self.use_post_attn_norm:
            history_output = self.post_attn_norm(history_output)
        x = x + history_output

        feed_forward_input = self.ffn_norm(x)
        feed_forward_output = self.ffn(feed_forward_input)
        if self.use_post_ffw_norm:
            feed_forward_output = self.post_ffw_norm(feed_forward_output)
        x = x + feed_forward_output

        return x, carry


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
        max_seq_length: int,
        *,
        rngs: nnx.Rngs,
    ):
        hidden_features = config.hidden_features

        # Convert string representations to actual objects if needed
        kernel_init = get_kernel_init(config.kernel_init)
        activation = parse_activation_fn(config.activation)
        dtype = get_dtype(config.dtype)
        param_dtype = get_dtype(config.param_dtype)
        norm = get_norm(config.norm)

        self.dtype = config.dtype

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

        layers = []
        for _ in range(config.num_layers):
            layers.append(
                TransformerBlock(
                    config=config.layer,
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

        value_in_dim = hidden_features if config.value_hidden_dim is None else config.value_hidden_dim
        if config.value.type == "hl_gauss":
            self.value_head = HlGaussValue(value_in_dim, config.value, rngs=rngs)
        elif config.value.type == "mse":
            self.value_head = MseValue(value_in_dim, rngs=rngs)

    def initialize_carry(self, batch_size: int, rngs):
        return tuple(layer.initialize_carry(batch_size, rngs) for layer in self.layers)

    def __call__(
        self, ts: TimeStep, carry = None
    ) -> tuple[jax.Array, tfd.Distribution, tuple[KVCache, ...] | None]:
        obs_embedding = self.obs_encoder(ts.obs)
        reward_embedding = self.reward_encoder(ts.last_reward[..., None])
        action_embedding = self.action_embedder.encode(ts.last_action)

        x = obs_embedding + reward_embedding + action_embedding

        if carry is not None:
            out_carry = []
            for layer, _carry in zip(self.layers, carry):
                x, _carry = layer(x, ts.time, _carry)
                out_carry.append(_carry)
            carry = tuple(out_carry)
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
        value_rep = value.astype(jnp.float32)

        return value_rep, policy, carry

    def get_value(self, value_rep: jax.Array) -> jax.Array:
        return self.value_head.get_value(value_rep)
    
    def get_value_loss(self, value_rep: jax.Array, targets: jax.Array) -> jax.Array:
        return self.value_head.get_loss(value_rep, targets)
