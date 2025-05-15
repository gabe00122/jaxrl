from typing import NamedTuple
import jax
from jax import numpy as jnp

_MAX_WAVELENGTH = 10_000


class RopeValues(NamedTuple):
    sin: jax.Array
    cos: jax.Array


def calculate_rope_values(
    positions: jax.Array, head_dim: int, max_wavelength: int = _MAX_WAVELENGTH
) -> RopeValues:
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = max_wavelength**fraction

    sinusoid_inp = positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    return RopeValues(sin, cos)


def apply_rope(inputs: jax.Array, rope_values: RopeValues) -> jax.Array:
    """Applies RoPE."""
    sin, cos = rope_values

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)
