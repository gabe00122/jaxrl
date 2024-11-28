from dataclasses import dataclass

@dataclass
class ObservationSpec:
    shape: tuple[int, ...]


@dataclass
class ContinuousActionSpec:
    shape: tuple[int, ...]


@dataclass
class DiscreteActionSpec:
    num_actions: int


ActionSpec = ContinuousActionSpec | DiscreteActionSpec
