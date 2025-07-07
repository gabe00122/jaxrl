
from functools import cached_property
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
import pygame

from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep

NUM_CLASSES = 3

map_template = """
xxxxxxxxxxxxxxxxxxxx
x                  x
x    xxxxxxxxx     x
x    x       x     x
x    x       x     x
x    x       xx xx x
x    x           x x
x    x           x x
x    xxxxxxxxx   x x
x            x   x x
x            x   x x
xxxxxxxxxx   x   x x
x          x x   x x
x          x x   x x
x          x x   x x
x   xxxxxxxx x   x x
x   x        x   x x
x   x        x   x x
x   x        x   x x
xxxxxxxxxxxxxxxxxxxx
"""

TILE_EMPTY = 0
TILE_WALL = 1
TILE_TREASURE = 2


class ReturnState(NamedTuple):
    pos: jax.Array
    tiles: jax.Array
    time: jax.Array


class ReturnEnv(Environment[ReturnState]):
    def __init__(self) -> None:
        super().__init__()

        self.unpadded_width = 20
        self.unpadded_height = 20

        self.view_width = 5
        self.view_height = 5
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.width = self.unpadded_width + self.pad_width
        self.height = self.unpadded_height + self.pad_height

        self.tiles, self.empty_positions = self._load_template(map_template)

    def _load_template(self, text: str):
        tiles = np.zeros((self.unpadded_width, self.unpadded_height), dtype=np.int8)
        empty_positions = []

        x = 0
        y = self.unpadded_height

        for c in text:
            if c == '\n':
                x = 0
                y -= 1
            else:
                if c == ' ':
                    empty_positions.append([x + self.pad_width, y + self.pad_height])

                tile_type = TILE_WALL if c == 'x' else TILE_EMPTY
                tiles[x, y] = tile_type
                x += 1

        tiles = np.pad(
            tiles,
            pad_width=((self.pad_width, self.pad_width), (self.pad_height, self.pad_height)),
            mode="constant",
            constant_values=TILE_WALL
        )

        return jnp.asarray(tiles), jnp.asarray(empty_positions)

    def reset(self, rng_key: jax.Array) -> tuple[ReturnState, TimeStep]:
        actor_pos, treasure_pos = jax.random.choice(rng_key, self.empty_positions, (2,), replace=False)

        tiles = self.tiles.at[treasure_pos[0], treasure_pos[1]].set(TILE_TREASURE)

        state = ReturnState(pos=actor_pos, tiles=tiles, time=jnp.int32(0))

        return state, self.encode_observation(state, jnp.int32(0), jnp.float32(0))

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(shape=(self.view_width, self.view_height), max_value=NUM_CLASSES, dtype=jnp.int8)

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(num_actions=4)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return 1

    def step(self, state: ReturnState, action: jax.Array, rng_key: jax.Array) -> tuple[ReturnState, TimeStep]:
        action = action.squeeze(axis=0)

        directions = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
        new_pos = state.pos + directions[action.squeeze()]

        is_wall = state.tiles[new_pos[0], new_pos[1]] == TILE_WALL

        pos = jax.lax.cond(is_wall, lambda: state.pos, lambda: new_pos)

        is_treasure = state.tiles[pos[0], pos[1]] == TILE_TREASURE
        reward = is_treasure.astype(jnp.float32)

        # randomize location after getting a reward
        pos = jax.lax.cond(is_treasure, lambda: jax.random.choice(rng_key, self.empty_positions, ()), lambda: pos)

        state = state._replace(pos=pos, time=state.time + 1)

        return state, self.encode_observation(state, action, reward)

    def encode_observation(self, state: ReturnState, last_action: jax.Array, last_reward: jax.Array) -> TimeStep:
        view = jax.lax.dynamic_slice(
            state.tiles,
            state.pos - jnp.array([self.view_width // 2, self.view_height // 2]),
            (self.view_width, self.view_height)
        )
        print(view.shape)

        return TimeStep(
            obs=view[None, ...],
            time=state.time[None, ...],
            last_action=last_action[None, ...],
            last_reward=last_reward[None, ...],
            action_mask=None,
        )

class ReturnClient:
    def __init__(self, env: ReturnEnv):
        self.env = env

        self.screen = None
        self.clock = None

        self._init_pygame()

    def _init_pygame(self):
        self.screen = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()

    def render(self, state: ReturnState):
        tile_size = 40

        tiles = state.tiles.tolist()
        colors = ["red", "green", "blue"]

        for x in range(self.env.unpadded_width):
            for y in range(self.env.unpadded_height):
                tx = self.env.pad_width + x
                ty = self.env.pad_height + y

                tile_type = tiles[tx][self.env.height - ty + 1]
                self.screen.fill(colors[tile_type], (x * tile_size, y * tile_size, tile_size, tile_size))

        agent_x = state.pos[0].item() - self.env.pad_width
        agent_y = (self.env.height - state.pos[1].item() + 1) - self.env.pad_height

        self.screen.fill("yellow", (agent_x * tile_size, agent_y * tile_size, tile_size, tile_size))

        self.clock.tick(10)
        pygame.display.flip()

def demo():
    env = ReturnEnv()

    rng_key = jax.random.PRNGKey(10)
    state, timestep = env.reset(rng_key)

    client = ReturnClient(env)

    running = True
    ts = None

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            state, ts = env.step(state, jnp.array([0]), rng_key)
        elif keys[pygame.K_s]:
            state, ts = env.step(state, jnp.array([2]), rng_key)
        elif keys[pygame.K_a]:
            state, ts = env.step(state, jnp.array([3]), rng_key)
        elif keys[pygame.K_d]:
            state, ts = env.step(state, jnp.array([1]), rng_key)

        # print(ts)
        client.render(state)

    pygame.quit()



if __name__ == '__main__':
    demo()
