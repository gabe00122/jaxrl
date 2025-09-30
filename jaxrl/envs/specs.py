from typing import NamedTuple
from jax.typing import DTypeLike


class ObservationSpec(NamedTuple):
    dtype: DTypeLike
    shape: tuple[int, ...]
    max_value: int | tuple[int, ...] | None = None


class ContinuousActionSpec(NamedTuple):
    num_actions: int


class DiscreteActionSpec(NamedTuple):
    num_actions: int


ActionSpec = ContinuousActionSpec | DiscreteActionSpec
