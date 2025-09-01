from abc import ABC, abstractmethod

from jaxrl.types import TimeStep


class EnvironmentClient[State](ABC):
    @abstractmethod
    def render(self, state: State, timestep: TimeStep): ...

    @abstractmethod
    def save_video(self): ...
