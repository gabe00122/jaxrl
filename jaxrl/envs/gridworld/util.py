
import math

from jax import numpy as jnp


def concat_one_hot(x: jnp.ndarray, sizes: tuple[int, ...], dtype=jnp.float32):
    *batch, n = x.shape
    flat_batch = math.prod(batch)
    total = sum(sizes)

    sizes = jnp.array(sizes, jnp.int32)
    offsets = jnp.cumsum(sizes) - sizes

    flat_x = x.reshape(flat_batch, n)
    idx = flat_x + offsets

    out = jnp.zeros((flat_batch, total), dtype)
    out = out.at[jnp.arange(flat_batch)[:, None], idx].set(1)
    out = out.reshape(*batch, total)

    return out
