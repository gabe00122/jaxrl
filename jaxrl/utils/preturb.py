import numpy as np
from flax import nnx
from jax import numpy as jnp


def preturb(layer: nnx.Linear, alpha: float, rngs: nnx.Rngs):
    kernel_key = rngs.params()
    new_params = layer.kernel_init(
        kernel_key, (layer.in_features, layer.out_features), layer.param_dtype
    )

    blended_kernel = (1 - alpha) * layer.kernel + alpha * new_params
    layer.kernel[...] = layer.kernel.replace(value=blended_kernel)


def preturb_genreal(layer: nnx.LinearGeneral, alpha: float, rngs: nnx.Rngs):
    kernel_key = rngs.params()
    in_features = np.prod(layer.in_features)
    out_features = np.prod(layer.out_features)

    new_params = layer.kernel_init(
        kernel_key, (in_features, out_features), layer.param_dtype
    )
    new_params = jnp.reshape(new_params, (*layer.in_features, *layer.out_features))

    blended_kernel = (1 - alpha) * layer.kernel + alpha * new_params
    layer.kernel[...] = layer.kernel.replace(value=blended_kernel)
