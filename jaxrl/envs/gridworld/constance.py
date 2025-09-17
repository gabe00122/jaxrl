from jax import numpy as jnp

from jaxrl.envs.specs import ObservationSpec

NUM_TYPES = 17

# Unified tile ids across gridworld environments
TILE_EMPTY = 0              # empty space
TILE_WALL = 1               # permanent wall
TILE_DESTRUCTIBLE_WALL = 2  # destructible
TILE_FLAG = 3               # typical goal tile
TILE_FLAG_UNLOCKED = 4      # used for the scouting environment where the the flag gets made available for taking

TILE_FLAG_BLUE_TEAM = 5
TILE_FLAG_RED_TEAM = 6

# agents are observed like tiles
AGENT_GENERIC = 7           # typical agent
AGENT_SCOUT = 8             # scout agent (scout env)
AGENT_HARVESTER = 9         # harvester agent (scout env)

AGENT_KNIGHT = 10
AGENT_ARCHER = 11

TILE_DECOR_1 = 12
TILE_DECOR_2 = 13
TILE_DECOR_3 = 14
TILE_DECOR_4 = 15

TILE_ARROW = 16


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


def mask_action_mask(actions: list[int]):
    mask = [False] * NUM_ACTIONS

    for action in actions:
        mask[action] = True

    return jnp.array(mask, jnp.bool)
