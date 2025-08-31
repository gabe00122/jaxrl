from jax import numpy as jnp

NUM_TYPES = 7

# Unified tile ids across gridworld environments
TILE_EMPTY = 0
TILE_WALL = 1
TILE_SOFT_WALL = 2
TILE_TREASURE = 3
TILE_TREASURE_OPEN = 4

AGENT_GENERIC = 5
AGENT_SCOUT = 6
AGENT_HARVESTER = 7

# Actions
NUM_ACTIONS = 4

MOVE_UP = 0
MOVE_RIGHT = 1
MOVE_DOWN = 2
MOVE_LEFT = 3


DIRECTIONS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
