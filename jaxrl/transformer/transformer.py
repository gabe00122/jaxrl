from flax import nnx
from jax import numpy as jnp
from jax.typing import DTypeLike

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
        kernel_init: nnx.Initializer,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.gtrxl_gate = gtrxl_gate

        self.attention_norm = nnx.LayerNorm(num_features=hidden_features, rngs=rngs)
        self.attention = AttentionBlock(num_heads, hidden_features, rngs=rngs)

        self.ffn_norm = nnx.LayerNorm(num_features=hidden_features, rngs=rngs)
        ff_block = GLUBlock if glu else FFBlock
        self.ffn = ff_block(hidden_features, ffn_size, activation=activation, kernel_init=kernel_init, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

        if gtrxl_gate:
            self.attention_gate = GatingMechanism(hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
            self.ffn_gate = GatingMechanism(hidden_features, dtype=dtype, param_dtype=param_dtype, rngs=rngs)

    def create_kv_cache(self, batch_size: int, context_size: int, *, dtype: DTypeLike | None = None):
        self.attention.create_kv_cache(batch_size, context_size, dtype=dtype)

    def __call__(self, x, time_steps, use_kv_cache: bool):
        attention_input = self.attention_norm(x)
        attention = self.attention(attention_input, time_steps, use_kv_cache=use_kv_cache)
        x = self.attention_gate(x, attention) if self.gtrxl_gate else x + attention

        feed_forward_input = self.ffn_norm(x)
        feed_forward = self.ffn(feed_forward_input)
        x = self.ffn_gate(x, feed_forward) if self.gtrxl_gate else x + feed_forward

        return x

