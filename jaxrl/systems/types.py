from typing import NamedTuple
import chex

from jaxrl.types import Observation, Done, Action


class Transition(NamedTuple):
    observation: Observation
    action: Action
    reward: chex.Array
    next_observation: Observation
    terminated: Done
    truncated: Done
