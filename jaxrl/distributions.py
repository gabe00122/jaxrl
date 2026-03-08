from typing import Any, Optional

import chex
import distrax


class IdentityTransformation(distrax.Transformed):
    """A distribution transformed using the `Identity()` bijector.

    We transform this distribution with the `Identity()` bijector to enable us to call
    `pi.entropy(seed)` and keep the API identical to the TanhTransformedDistribution.
    """

    def __init__(self, distribution: distrax.Distribution) -> None:
        """Initialises the IdentityTransformation."""
        super().__init__(
            distribution=distribution, bijector=distrax.Lambda(lambda x: x)
        )

    def entropy(self, seed: chex.PRNGKey = None) -> chex.Array:
        """Computes the entropy of the distribution."""
        return self.distribution.entropy()

    @classmethod
    def _parameter_properties(
        cls, dtype: Optional[Any], num_classes: Any = None
    ) -> Any:
        td_properties = super()._parameter_properties(
            dtype, num_classes=num_classes
        )
        del td_properties["bijector"]
        return td_properties
