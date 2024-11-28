from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic, TypeVar

import chex
from jax import Array

from jaxrl.types import Observation
from jaxrl.envs.specs import ObservationSpec, ActionSpec
import enum


class StepType(enum.IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


@dataclass
class TimeStep:
    step_type: Array
    observation: Observation
    reward: Array


State = TypeVar("State")


class EnvWrapper(ABC, Generic[State]):
    @abstractmethod
    def reset(self) -> tuple[State, TimeStep]: ...

    @abstractmethod
    def step(self, state: State, action: Array) -> tuple[State, TimeStep]: ...

    @cached_property
    @abstractmethod
    def observation_spec(self) -> ObservationSpec: ...

    @cached_property
    @abstractmethod
    def action_spec(self) -> ActionSpec: ...

    @property
    @abstractmethod
    def players(self) -> int: ...

    @property
    @abstractmethod
    def is_jittable(self) -> bool: ...

    @property
    @abstractmethod
    def num_envs(self) -> int: ...
