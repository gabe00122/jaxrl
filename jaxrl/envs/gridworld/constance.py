from jax import numpy as jnp

from jaxrl.envs.specs import ObservationSpec

NUM_TYPES = 15

# Unified tile ids across gridworld environments
TILE_EMPTY = 0              # empty space
TILE_WALL = 1               # permanent wall
TILE_DESTRUCTIBLE_WALL = 2  # destructible
TILE_FLAG = 3               # typical goal tile
TILE_FLAG_UNLOCKED = 4      # used for the scouting environment where the the flag gets made available for taking

# TILE_FLAG_BLUE_TEAM = 5
# TILE_FLAG_RED_TEAM = 6

# agents are observed like tiles
AGENT_GENERIC = 5           # typical agent
AGENT_SCOUT = 6             # scout agent (scout env)
AGENT_HARVESTER = 7         # harvester agent (scout env)

AGENT_KNIGHT = 8
AGENT_ARCHER = 9

TILE_DECOR_1 = 10
TILE_DECOR_2 = 11
TILE_DECOR_3 = 12
TILE_DECOR_4 = 13

TILE_ARROW = 14


# Actions
NUM_ACTIONS = 7

MOVE_UP = 0
MOVE_RIGHT = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
STAY = 4
PRIMARY_ACTION = 5
DIG_ACTION = 6


DIRECTIONS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)


def make_obs_spec(width: int, height: int) -> ObservationSpec:
    # CHANNELS:
    # TILE
    # DIRECTION
    # TEAM ID
    # HEALTH
    obs_spec = ObservationSpec(
        dtype=jnp.int8,
        shape=(width, height, 4),
        max_value=(
            NUM_TYPES,
            5, # none, up, right, down, left,
            3, # none, red, blue
            3, # 0, 1, 2
        ),
    )

    return obs_spec


def make_action_mask(actions: list[int], num_agents: int):
    mask = [False] * NUM_ACTIONS

    for action in actions:
        mask[action] = True

    mask_array = jnp.array(mask, jnp.bool)
    return jnp.repeat(mask_array[None, :], num_agents, axis=0)
