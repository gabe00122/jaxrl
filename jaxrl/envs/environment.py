from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

from jax import Array

from jaxrl.types import TimeStep
from jaxrl.envs.specs import ObservationSpec, ActionSpec
import enum


class StepType(enum.IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


State = TypeVar("State")


class Environment(ABC, Generic[State]):
    @abstractmethod
    def reset(self, rng_key: Array) -> tuple[State, TimeStep]: ...

    @abstractmethod
    def step(self, state: State, action: Array, rng_key: Array) -> tuple[State, TimeStep]: ...

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

