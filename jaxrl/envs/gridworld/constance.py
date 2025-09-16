from jax import numpy as jnp

NUM_TYPES = 16

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

# AGENT_RED_KNIGHT_UP    = 10
AGENT_RED_KNIGHT_RIGHT = 10
# AGENT_RED_KNIGHT_DOWN  = 12
# AGENT_RED_KNIGHT_LEFT  = 13

# AGENT_RED_ARCHER_UP    = 14
# AGENT_RED_ARCHER_RIGHT = 15
# AGENT_RED_ARCHER_DOWN  = 16
# AGENT_RED_ARCHER_LEFT  = 17

# AGENT_BLUE_KNIGHT_UP    = 18
AGENT_BLUE_KNIGHT_RIGHT = 11
# AGENT_BLUE_KNIGHT_DOWN  = 20
# AGENT_BLUE_KNIGHT_LEFT  = 21

# AGENT_BLUE_ARCHER_UP    = 22
# AGENT_BLUE_ARCHER_RIGHT = 23
# AGENT_BLUE_ARCHER_DOWN  = 24
# AGENT_BLUE_ARCHER_LEFT  = 25


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
