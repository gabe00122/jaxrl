from dataclasses import dataclass
from jax.typing import DTypeLike

@dataclass
class ObservationSpec:
    dtype: DTypeLike
    shape: tuple[int, ...]


@dataclass
class ContinuousActionSpec:
    num_actions: int


@dataclass
class DiscreteActionSpec:
    num_actions: int


ActionSpec = ContinuousActionSpec | DiscreteActionSpec
