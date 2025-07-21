from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from flax import nnx

from jaxrl.transformer import positional_embeddings
from jaxrl.constants import index_type


def position_mask(time_steps, max_seq_length: int):
    seq_range = jnp.arange(max_seq_length, dtype=index_type)
    mask = seq_range[None, None, :] <= time_steps[:, :, None]
    return mask[:, None, :, :]


def einsum_attention(
    query, key, value, mask
):
    dtype = query.dtype

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

    attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key)

    big_neg = jnp.finfo(dtype).min
    attn_weights = jnp.where(mask, attn_weights, big_neg)

    attn_weights = jax.nn.softmax(attn_weights, axis=-1).astype(key.dtype)

    x = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
    return x


class KVCache(NamedTuple):
    key: jax.Array
    value: jax.Array


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        use_built_in: bool = True,
        max_seq_length: int,
        rope_max_wavelength: float = 10_000,
        use_qk_norm: bool = False,
        attention_softcap: float | None = None,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        kernel_init: nnx.Initializer = nnx.initializers.normal(),
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_qk_norm = use_qk_norm
        self.max_seq_length = max_seq_length
        self.rope_max_wavelength = rope_max_wavelength
        self.attention_softcap = attention_softcap
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.use_built_in = use_built_in

        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"Memory dimension ({self.d_model}) must be divisible by "
                f"'num_heads' heads ({self.num_heads})."
            )

        self.kv_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(self.num_kv_heads, self.head_dim * 2),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )

        self.query_proj = nnx.LinearGeneral(
            in_features=self.d_model,
            out_features=(self.num_heads, self.head_dim),
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

        if self.use_qk_norm:
            self._query_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)
            self._key_norm = nnx.RMSNorm(self.head_dim, rngs=rngs)

    def create_kv_cache(
        self, batch_size: int
    ) -> KVCache:
        shape = (batch_size, self.max_seq_length, self.num_kv_heads, self.head_dim)
        key = jnp.zeros(shape, dtype=self.dtype)
        value = jnp.zeros(shape, dtype=self.dtype)
        return KVCache(key, value)

    def update_kv_cache(self, kv_cache: KVCache, seq_pos, key, value) -> KVCache:
        pos = seq_pos[0, 0]

        key = jax.lax.dynamic_update_slice(kv_cache.key, key, (0, pos, 0, 0))
        value = jax.lax.dynamic_update_slice(kv_cache.value, value, (0, pos, 0, 0))

        return KVCache(key, value)

    def __call__(self, inputs, seq_pos, kv_cache: KVCache | None = None) -> tuple[jax.Array, KVCache | None]:
        kv_proj = self.kv_proj(inputs)
        key, value = jnp.split(kv_proj, 2, -1)
        query = self.query_proj(inputs)

        if self.use_qk_norm:
            query = self._query_norm(query)
            key = self._key_norm(key)

        query = positional_embeddings.apply_rope(query, seq_pos, self.head_dim, self.rope_max_wavelength)
        key = positional_embeddings.apply_rope(key, seq_pos, self.head_dim, self.rope_max_wavelength)

        if kv_cache is not None:
            kv_cache = self.update_kv_cache(kv_cache, seq_pos, key, value)
            key = kv_cache.key
            value = kv_cache.value

        mask = position_mask(seq_pos, self.max_seq_length)
        # jax.debug.print("- {}", mask)

        if self.use_built_in:
            batch, seq, _ = inputs.shape
            training = seq > 1
            if training:
                x = jax.nn.dot_product_attention(query, key, value, is_causal=True, implementation='cudnn')
            else:
                kv_length = jnp.full((batch,), seq_pos[0, 0]+1)
                x = jax.nn.dot_product_attention(query, key, value, key_value_seq_lengths=kv_length, implementation='cudnn')
        else:
            x = einsum_attention(
                query,
                key,
                value,
                mask,
            )

        out = self.out(x)

        return out, kv_cache
