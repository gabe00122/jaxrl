import jax
from jax import numpy as jnp
from jax.typing import DTypeLike

from flax import nnx
import tensorflow_probability.substrates.jax.distributions as tfd


from jaxrl.networks import create_norm
from jaxrl.transformer import positional_embeddings
from jaxrl.transformer.attention import position_mask
from jaxrl.types import Observation
from jaxrl.transformer.attention import AttentionBlock
from jaxrl.transformer.feed_forward import GLUBlock, FFBlock
from jaxrl.transformer.gate import GatingMechanism


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_features: int,
        ffn_size: int,
        *,
        activation: str,
        glu: bool = True,
        gtrxl_gate: bool = True,
        gtrxl_bias: float = 0.0,
        attention_softcap: float | None = None,
        kernel_init: nnx.Initializer,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.gtrxl_gate = gtrxl_gate

        self.attention_norm = nnx.LayerNorm(
            num_features=hidden_features,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.attention = AttentionBlock(
            num_heads, hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

        self.ffn_norm = nnx.LayerNorm(
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
                hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )
            self.ffn_gate = GatingMechanism(
                hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs
            )

    def create_kv_cache(
        self, batch_size: int, context_size: int, *, dtype: DTypeLike | None = None
    ):
        self.attention.create_kv_cache(batch_size, context_size, dtype=dtype)

    def __call__(self, x, time_steps, mask, rope_values, use_kv_cache: bool):
        attention_input = self.attention_norm(x)
        attention = self.attention(
            attention_input, time_steps, mask, rope_values, use_kv_cache=use_kv_cache
        )
        x = self.attention_gate(x, attention) if self.gtrxl_gate else x + attention

        feed_forward_input = self.ffn_norm(x)
        feed_forward = self.ffn(feed_forward_input)
        x = self.ffn_gate(x, feed_forward) if self.gtrxl_gate else x + feed_forward

        return x


class TransformerActorCritic(nnx.Module):
    def __init__(
        self,
        obs_encoder: nnx.Module,
        action_head: nnx.Module,
        num_layers: int,
        num_heads: int,
        hidden_features: int,
        ffn_size: int,
        max_seq_length: int,
        *,
        activation: str,
        norm: str,
        use_bias: bool = True,
        glu: bool = True,
        gtrxl_gate: bool = True,
        gtrxl_bias: float = 0.0,
        attention_softcap: float | None = None,
        kernel_init: nnx.Initializer,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.head_size = hidden_features // num_heads
        self.max_seq_length = max_seq_length

        self.obs_encoder = obs_encoder
        self.action_head = action_head

        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerBlock(
                    num_heads,
                    hidden_features,
                    ffn_size,
                    activation=activation,
                    glu=glu,
                    gtrxl_gate=gtrxl_gate,
                    gtrxl_bias=gtrxl_bias,
                    kernel_init=kernel_init,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    rngs=rngs,
                )
            )
        self.layers = tuple(layers)
        self.output_norm = create_norm(norm, hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        self.value_head = nnx.Linear(
            hidden_features,
            1,
            use_bias=use_bias,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def create_kv_cache(
        self, batch_size: int, context_size: int, *, dtype: DTypeLike | None = None
    ):
        for layer in self.layers:
            layer.create_kv_cache(batch_size, context_size, dtype=dtype)

    def __call__(
        self, observation: Observation, use_kv_cache: bool
    ) -> tuple[jax.Array, tfd.Distribution]:
        net_input = jnp.concatenate(
            (
                observation.agents_view,
                observation.last_action[..., None],
                observation.last_reward[..., None],
            ),
            axis=-1,
        )

        x = self.obs_encoder(net_input)

        mask = position_mask(observation.time_steps, 1024)
        position_values = positional_embeddings.calculate_rope_values(observation.time_steps, self.head_size)

        for layer in self.layers:
            x = layer(x, observation.time_steps, mask, position_values, use_kv_cache)

        x = self.output_norm(x)

        action_logits = self.action_head(x)
        value = self.value_head(x)

        return value, tfd.Categorical(action_logits)
