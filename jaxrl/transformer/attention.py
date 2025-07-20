from typing import NamedTuple
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
    mask = seq_range[None, None, :] <= time_steps[:, :, None]
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


class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        *,
        max_seq_length: int,
        rope_max_wavelength: float,
        attention_softcap: float | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer = nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.rope_max_wavelength = rope_max_wavelength
        self.attention_softcap = attention_softcap
        self.dtype = dtype
        self.param_dtype = param_dtype

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.head_dim = self.d_model // self.num_heads

        self.key_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(self.num_heads, self.head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.value_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(self.num_heads, self.head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.query_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(1, self.head_dim),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.out = nnx.LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.d_model,
            axis=(-2, -1),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self._query_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
        self._key_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)

    def create_kv_cache(
        self, batch_size: int, context_size: int, *, dtype: DTypeLike | None = None
    ) -> KVCache:
        shape = (batch_size, context_size, self.num_heads, self.head_dim)
        key = jnp.zeros(shape, dtype=dtype)
        value = jnp.zeros(shape, dtype=dtype)
        return KVCache(key, value)

    def update_kv_cache(self, kv_cache: KVCache, seq_pos, key, value) -> KVCache:
        batch_idx = jnp.arange(kv_cache.key.shape[0], dtype=index_type)
        batch_idx = batch_idx[:, None]
        # pos = seq_pos[0, 0]

        # todo: investigate dynamic slice?
        # key = jax.lax.dynamic_update_slice(kv_cache.key, key, (0, pos, 0, 0))
        # value = jax.lax.dynamic_update_slice(kv_cache.value, value, (0, pos, 0, 0))
        key = kv_cache.key.at[batch_idx, seq_pos].set(key)
        value = kv_cache.value.at[batch_idx, seq_pos].set(value)

        return KVCache(key, value)

    def __call__(self, inputs, seq_pos, kv_cache: KVCache | None = None) -> tuple[jax.Array, KVCache | None]:
        # in_proj = self.in_proj(inputs)
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)

        query = self._query_norm(query)
        key = self._key_norm(key)

        query = positional_embeddings.apply_rope(query, seq_pos, self.head_dim, self.rope_max_wavelength)
        key = positional_embeddings.apply_rope(key, seq_pos, self.head_dim, self.rope_max_wavelength)

        if kv_cache is not None:
            kv_cache = self.update_kv_cache(kv_cache, seq_pos, key, value)
            key = kv_cache.key
            value = kv_cache.value

        mask = position_mask(seq_pos, self.max_seq_length)

        # x = jax.nn.dot_product_attention(query, key, value, implementation='cudnn')
        x = einsum_attention(
            query,
            key,
            value,
            mask,
            attention_softcap=self.attention_softcap,
        )
        out = self.out(x)

        return out, kv_cache
