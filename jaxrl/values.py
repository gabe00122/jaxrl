from typing import Any
from flax import nnx

from jaxrl.config import HlGaussConfig


class HlGaussValue(nnx.Module):
    def __init__(self, in_features: int, hl_gauss_config: HlGaussConfig, *, rngs: nnx.Rngs) -> None:
        self.dense = nnx.Linear(in_features, hl_gauss_config.n_logits, rngs=rngs)

    def __call__(self) -> Any:
        pass
