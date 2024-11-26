from dataclasses import dataclass

@dataclass
class ObservationSpec:
    shape: tuple[int, ...]


@dataclass
class ActionSpec:
    shape: tuple[int, ...]
    discrete: bool
