
from typing import NamedTuple, Self

import jax
from jax import numpy as jnp
import jaxrl.envs.gridworld.constance as GW


def unique_mask(xs: jax.Array) -> jax.Array:
    """
    Returns a mask over values that are unique
    """

    _, inv, counts = jnp.unique(xs, return_inverse=True, return_counts=True, size=xs.shape[0])

    return counts[inv] == 1
