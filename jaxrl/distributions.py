from typing import Any, Optional

import chex
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd


class TanhTransformedDistribution(tfd.TransformedDistribution):
    """A distribution transformed using the `tanh` function.

    This transformation was adapted to acme's implementation.
    For details, please see: http://tinyurl.com/2x5xea57
    """

    def __init__(
        self,
        distribution: tfd.Distribution,
        threshold: float = 0.999,
        validate_args: bool = False,
    ) -> None:
        """Initialises the TanhTransformedDistribution.

        Args:
        ----
          distribution: The base distribution to be transformed.
          bijector: The bijective transformation applied to the distribution.
          threshold: Clipping value for the action when computing the log_prob.
          validate_args: Whether to validate input with respect to distribution parameters.
        """

        super().__init__(
            distribution=distribution, bijector=tfb.Tanh(), validate_args=validate_args
        )
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = jnp.log(1.0 - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = (
            self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        )
        self._log_prob_right = (
            self.distribution.log_survival_function(inverse_threshold) - log_epsilon
        )

    def log_prob(self, event: chex.Array) -> chex.Array:
        """Computes the log probability of the event under the transformed distribution."""
        # Without this clip, there would be NaNs in the internal tf.where.
        event = jnp.clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return jnp.where(
            event <= -self._threshold,
            self._log_prob_left,
            jnp.where(
                event >= self._threshold, self._log_prob_right, super().log_prob(event)
            ),
        )

    def mode(self) -> chex.Array:
        """Returns the mode of the distribution."""
        return self.bijector.forward(self.distribution.mode())

    def entropy(self, seed: chex.PRNGKey = None) -> chex.Array:
        """Computes an estimation of the entropy using a sample of the log_det_jacobian."""
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(
            self.distribution.sample(seed=seed), event_ndims=0
        )

    @classmethod
    def _parameter_properties(
        cls, dtype: Optional[Any], num_classes: Any = None
    ) -> Any:
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class IdentityTransformation(tfd.TransformedDistribution):
    """A distribution transformed using the `Identity()` bijector.

    We transform this distribution with the `Identity()` bijector to enable us to call
    `pi.entropy(seed)` and keep the API identical to the TanhTransformedDistribution.
    """

    def __init__(self, distribution: tfd.Distribution) -> None:
        """Initialises the IdentityTransformation."""
        super().__init__(distribution=distribution, bijector=tfb.Identity())

    def entropy(self, seed: chex.PRNGKey = None) -> chex.Array:
        """Computes the entropy of the distribution."""
        return self.distribution.entropy()

    @classmethod
    def _parameter_properties(
        cls, dtype: Optional[Any], num_classes: Any = None
    ) -> Any:
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties
