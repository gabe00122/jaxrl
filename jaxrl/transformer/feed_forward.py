import functools
from typing import Callable

from flax import nnx
import jax
from jax.typing import DTypeLike

from jaxrl.utils.preturb import preturb


class FFBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        hidden_features: int,
        activation: Callable[[jax.Array], jax.Array],
        *,
        kernel_init: nnx.Initializer,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.hidden_features = hidden_features

        linear = functools.partial(
            nnx.Linear,
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.activation = activation
        self.up_proj = linear(d_model, hidden_features)
        self.down_proj = linear(hidden_features, d_model)

    def __call__(self, inputs):
        x = self.up_proj(inputs)
        x = self.activation(x)
        out = self.down_proj(x)
        return out

    def preturb(self, alpha: float, rngs: nnx.Rngs):
        preturb(self.up_proj, alpha, rngs)
        preturb(self.down_proj, alpha, rngs)


class GLUBlock(nnx.Module):
    def __init__(
        self,
        d_model: int,
        hidden_features: int,
        activation: Callable[[jax.Array], jax.Array],
        *,
        kernel_init: nnx.Initializer,
        dtype: DTypeLike,
        param_dtype: DTypeLike,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.hidden_features = hidden_features

        linear = functools.partial(
            nnx.Linear,
            use_bias=False,
            kernel_init=kernel_init,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.activation = activation
        self.up_proj = linear(d_model, hidden_features)
        self.up_gate = linear(d_model, hidden_features)
        self.down_proj = linear(hidden_features, d_model)

    def __call__(self, inputs):
        x = self.up_proj(inputs)
        gate = self.up_gate(inputs)
        x = self.activation(x) * gate
        out = self.down_proj(x)
        return out

    def preturb(self, alpha: float, rngs: nnx.Rngs):
        preturb(self.up_proj, alpha, rngs)
        preturb(self.up_gate, alpha, rngs)
        preturb(self.down_proj, alpha, rngs)
