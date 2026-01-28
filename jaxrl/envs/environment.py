from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

import jax
from jax import Array

from mapox import TimeStep
from jaxrl.envs.specs import ObservationSpec, ActionSpec
import enum


class StepType(enum.IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


class Environment[State](ABC):
    @abstractmethod
    def reset(self, rng_key: Array) -> tuple[State, TimeStep]: ...

    @abstractmethod
    def step(
        self, state: State, action: Array, rng_key: Array
    ) -> tuple[State, TimeStep]: ...

    @abstractmethod
    def create_placeholder_logs(self) -> dict[str, Any]: ...

    @abstractmethod
    def create_logs(self, state) -> dict[str, Any]: ...

    @cached_property
    @abstractmethod
    def observation_spec(self) -> ObservationSpec: ...

    @cached_property
    @abstractmethod
    def action_spec(self) -> ActionSpec: ...

    @property
    @abstractmethod
    def num_agents(self) -> int: ...

    @property
    @abstractmethod
    def is_jittable(self) -> bool: ...

    @property
    def teams(self) -> jax.Array | None:
        return None
