
from typing import NamedTuple, Self

import jax
from jax import numpy as jnp
import jaxrl.envs.gridworld.constance as GW


class Position(NamedTuple):
    data: jax.Array

    def move(self, direction: jax.Array) -> Self:
        direction = GW.DIRECTIONS[direction]
        if len(direction.shape) > 1:
            direction = direction.transpose(1, 0)

        return self.__class__(data=self.data + direction)

    @property
    def x(self):
        return self.data[0]
    
    @property
    def y(self):
        return self.data[1]
    
    @classmethod
    def from_xy(cls, x: jax.Array, y: jax.Array) -> Self:
        data = jnp.stack((x, y))
        return cls(data)


def unique_mask(xs: jax.Array) -> jax.Array:
    """
    Returns a mask over values that are unique
    """

    _, inv, counts = jnp.unique(xs, return_inverse=True, return_counts=True, size=xs.shape[0])

    return counts[inv] == 1
