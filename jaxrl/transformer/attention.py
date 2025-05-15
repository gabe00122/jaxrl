import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import nnx

from jaxrl.transformer import positional_embeddings
from jaxrl.constants import index_type


def softcap(x, cap):
    return jnp.tanh(x / cap) * cap


def position_mask(time_steps, max_seq_length: int):
    seq_range = jnp.arange(max_seq_length, dtype=index_type)
    mask = seq_range[None, None, :] <= time_steps[:, None]
    return mask[:, None, :, :]


def einsum_attention(
    query, key, value, mask, *, attention_softcap: float | None = None
):
    dtype = query.dtype

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

    attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key)

    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

    if attention_softcap is not None:
        attn_weights = softcap(attn_weights, attention_softcap)

    attn_weights = jax.nn.softmax(attn_weights)

    x = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
    return x


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        *,
        attention_softcap: float | None = None,
        use_bias: bool = False,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer = nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.attention_softcap = attention_softcap
        self.dtype = dtype
        self.param_dtype = param_dtype

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.d_model // self.num_heads

        self.in_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(self.num_heads, self.head_dim * 3),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            use_bias=use_bias,
            rngs=rngs,
        )

        self.out = nnx.LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.d_model,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            use_bias=use_bias,
            rngs=rngs,
        )

    def create_kv_cache(
        self, batch_size: int, context_size: int, *, dtype: DTypeLike | None = None
    ):
        shape = (batch_size, context_size, self.num_heads, self.head_dim)
        self.key_cache = nnx.Variable(jnp.zeros(shape, dtype=dtype))
        self.value_cache = nnx.Variable(jnp.zeros(shape, dtype=dtype))

    def update_kv_cache(self, time_steps, key, value):
        batch_idx = jnp.arange(self.key_cache.shape[0], dtype=index_type)
        batch_idx = batch_idx[:, None]

        self.key_cache.value = self.key_cache.value.at[batch_idx, time_steps].set(key)
        self.value_cache.value = self.value_cache.value.at[batch_idx, time_steps].set(
            value
        )

        return self.key_cache.value, self.value_cache.value

    def __call__(self, inputs, time_steps, mask, rope_values, use_kv_cache: bool):
        in_proj = self.in_proj(inputs)

        query, key, value = jnp.split(in_proj, 3, -1)

        query = positional_embeddings.apply_rope(query, rope_values)
        key = positional_embeddings.apply_rope(key, rope_values)

        if use_kv_cache:
            key, value = self.update_kv_cache(time_steps, key, value)

        x = einsum_attention(
            query,
            key,
            value,
            mask,
            attention_softcap=self.attention_softcap,
        )
        out = self.out(x)

        return out
