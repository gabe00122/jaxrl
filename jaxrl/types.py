import jax
from typing import NamedTuple, TypeAlias
import chex

Action: TypeAlias = jax.Array
Value: TypeAlias = chex.Array
Done: TypeAlias = chex.Array

type Metrics = dict[str, chex.Array]


class Observation(NamedTuple):
    """The observation that the agent sees.

    agents_view: the agent's view of the environment.
    action_mask: boolean array specifying, for each agent, which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    agents_view: chex.Array  # (num_agents, num_obs_features)
    action_mask: chex.Array | None  # (num_agents, num_actions)
    # step_count: chex.Array  # (num_agents, )
