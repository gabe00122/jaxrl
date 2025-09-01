from typing import Any
from einops import rearrange
from flax import nnx
import jax.numpy as jnp
from jax.scipy.stats import norm
import optax


from jaxrl.config import HlGaussConfig


def calculate_supports(config: HlGaussConfig):
    support = jnp.linspace(
        config.min, config.max, config.n_logits + 1, dtype=jnp.float32
    )
    centers = (support[:-1] + support[1:]) / 2
    support = support[None, :]

    return support, centers


class HlGaussValue(nnx.Module):
    def __init__(
        self, in_features: int, hl_gauss_config: HlGaussConfig, *, rngs: nnx.Rngs
    ) -> None:
        self._min = hl_gauss_config.min
        self._max = hl_gauss_config.max
        self._sigma = hl_gauss_config.sigma

        self.dense = nnx.Linear(in_features, hl_gauss_config.n_logits, rngs=rngs)
        self._supports, self._centers = calculate_supports(hl_gauss_config)

    def __call__(self, x) -> Any:
        return self.dense(x)

    def get_value(self, logits):
        probs = nnx.softmax(logits, axis=-1)
        return (probs * self._centers).sum(-1)

    def get_loss(self, logits, target_values):
        logits = rearrange(logits, "b t l -> (b t) l")
        target_values = rearrange(target_values, "b t -> (b t)")

        targets = jnp.clip(target_values, self._min, self._max)

        cdf_evals = norm.cdf(self._supports, loc=targets[:, None], scale=self._sigma)

        z = cdf_evals[:, -1] - cdf_evals[:, 0]

        bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]

        target_probs = bin_probs / z[:, None]

        return optax.softmax_cross_entropy(logits, target_probs, axis=-1).mean()


class MseValue(nnx.Module):
    def __init__(self, in_features: int, *, rngs: nnx.Rngs) -> None:
        self.dense = nnx.Linear(in_features, 1, rngs=rngs)

    def __call__(self, x) -> Any:
        x = self.dense(x)
        return x.squeeze(axis=-1)

    def get_value(self, value):
        return value

    def get_loss(self, values, target_values):
        return 0.5 * jnp.square(values - target_values).mean()
