from flax import nnx
from jax import numpy as jnp, nn


class GatingMechanism(nnx.Module):
    def __init__(
            self,
            d_input,
            bg=0.1,
            *,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            rngs: nnx.Rngs,
        ):
        self.Wr = nnx.Linear(d_input, d_input, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.Ur = nnx.Linear(d_input, d_input, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.Wz = nnx.Linear(d_input, d_input, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.Uz = nnx.Linear(d_input, d_input, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.Wg = nnx.Linear(d_input, d_input, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.Ug = nnx.Linear(d_input, d_input, dtype=dtype, param_dtype=param_dtype, rngs=rngs)
        self.bg = bg

    def __call__(self, x, y):
        r = nn.sigmoid(self.Wr(y) + self.Ur(x))
        z = nn.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = nn.tanh(self.Wg(y) + self.Ug(jnp.multiply(r, x)))
        g = jnp.multiply(1 - z, x) + jnp.multiply(z, h)
        return g
