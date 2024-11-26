from dataclasses import dataclass
from abc import ABC, abstractmethod
from tkinter import FIRST
from typing import Generic, TypeVar

from jax import Array
from flax import nnx

from jaxrl.types import Observation
from jaxrl.envs.specs import ObservationSpec, ActionSpec
import enum


class StepType(enum.IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2

    def first(self) -> bool:
        return self is StepType.FIRST
    
    def mid(self) -> bool:
        return self is StepType.MID
    
    def last(self) -> bool:
        return self is StepType.LAST


@dataclass
class TimeStep:
    step_type: StepType
    observation: Observation
    reward: Array

    def first(self) -> bool:
        return self.step_type.first()
    
    def mid(self) -> bool:
        return self.step_type.mid()
    
    def last(self) -> bool:
        return self.step_type.last()


State = TypeVar("State")


class EnvWrapper(ABC, Generic[State]):
    @abstractmethod
    def reset(self) -> tuple[State, TimeStep]: ...

    @abstractmethod
    def step(self, state: State, action: Array) -> tuple[State, TimeStep]: ...

    @abstractmethod
    def observation_spec(self) -> ObservationSpec: ...

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
