from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

import matplotlib.pyplot as plt

class HlGaussConfig(NamedTuple):
    min_value: float
    max_value: float
    n_logits: int
    sigma: float


def calculate_supports(config: HlGaussConfig, batch_size: int):
    support = jnp.linspace(config.min_value, config.max_value, config.n_logits + 1, dtype=jnp.float32)
    centers = (support[:-1] + support[1:]) / 2
    support = support[None, :].repeat(batch_size, axis=0)

    return support, centers

def transform_to_probs(config: HlGaussConfig, support: jax.Array, target: jax.Array):
    target = jnp.clip(target, config.min_value, config.max_value)

    cdf_evals = norm.cdf(support, loc=target[:, None], scale=config.sigma)

    z = cdf_evals[:, -1] - cdf_evals[:, 0]

    bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]

    return bin_probs / z[:, None]

def transform_from_probs(centers: jax.Array, probs: jax.Array):
    return (probs * centers).sum(-1)


def main():
    config = HlGaussConfig(
        min_value=-0.5,
        max_value=0.5,
        n_logits=51,
        sigma=0.75/20
    )
    target_value = 0.0
    support, centers = calculate_supports(config, 1)

    probs = transform_to_probs(config, support, jnp.array([target_value]))

    plt.bar(centers, probs[0], width=(config.max_value - config.min_value) / config.n_logits, alpha=0.6, color='skyblue', edgecolor='k')
    plt.axvline(target_value, color='red', linestyle='--', label='Target')
    plt.title(f"Gaussian histogram projection\nTarget={target_value}, sigma={config.sigma}")
    plt.xlabel("Bin center")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig("test.png")


if __name__ == "__main__":
    main()
