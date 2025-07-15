
from functools import cached_property, partial
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
import pygame

from jaxrl.config import ReturnConfig
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.utils.video_writter import save_video

NUM_CLASSES = 4

map_template = """
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
x                  xx                  x
x x  xxxx xxxx  x  xx x  xxxx xxxx  x  x
x    x       x     xx    x       x     x
x    x       x     xx    x       x     x
x  x     x         xx  x     x         x
x    x       x  x  xx    x       x  x  x
x    x       x     xx    x       x     x
x    xxxx xxxx     xx    xxxx xxxx     x
x    x        x    xx    x        x    x
x              x   xx              x   x
xxxxxxxxxx      x  xxxxxxxxxxx      x  x
x          x x     xx          x x     x
x x    x   x       xx x    x   x       x
x   x      x   x   xx   x      x   x   x
x   xxxxxxxx       xx   xxxxxxxx       x
x   x        x     xx   x        x     x
x           x   x  xx           x   x  x
x   x      x            x      x       x
xxxxxxxxxxxxxxxxxx    xxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxx    xxxxxxxxxxxxxxxxxx
x                                      x
x x  xxxx xxxx  x  xx x  xxxx xxxx  x  x
x    x       x     xx    x       x     x
x    x       x     xx    x       x     x
x  x     x         xx  x     x         x
x    x       x  x  xx    x       x  x  x
x    x       x     xx    x       x     x
x    xxxx xxxx     xx    xxxx xxxx     x
x    x        x    xx    x        x    x
x              x   xx              x   x
xxxxxxxxxx      x  xxxxxxxxxxx      x  x
x          x x     xx          x x     x
x x    x   x       xx x    x   x       x
x   x      x   x   xx   x      x   x   x
x   xxxxxxxx       xx   xxxxxxxx       x
x   x        x     xx   x        x     x
x           x   x  xx           x   x  x
x   x      x       xx   x      x       x
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
"""

TILE_EMPTY = 0
TILE_WALL = 1
TILE_TREASURE = 2
TILE_AGENT = 3


class ReturnState(NamedTuple):
    agents_pos: jax.Array
    treasure_pos: jax.Array
    time: jax.Array


class ReturnEnv(Environment[ReturnState]):
    def __init__(self, config: ReturnConfig) -> None:
        super().__init__()

        self._num_agents = config.num_agents

        self.unpadded_width = 40
        self.unpadded_height = 40

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
        positions = jax.random.choice(rng_key, self.empty_positions, (1 + self.num_agents,), replace=False)
        treasure_pos = positions[0]
        agents_pos = positions[1:]

        state = ReturnState(treasure_pos=treasure_pos, agents_pos=agents_pos, time=jnp.int32(0))

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        return state, self.encode_observations(state, actions, rewards)

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
        return self._num_agents

    def step(self, state: ReturnState, action: jax.Array, rng_key: jax.Array) -> tuple[ReturnState, TimeStep]:
        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0, 0))
        def _step_agent(local_position, local_action, random_position):
            directions = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
            new_pos = local_position + directions[local_action]

            new_tile = self.tiles[new_pos[0], new_pos[1]]

            # don't move if we are moving into a wall
            new_pos = jnp.where(new_tile == TILE_WALL, local_position, new_pos)

            found_treasure = jnp.all(new_pos == state.treasure_pos) #new_tile == TILE_TREASURE
            reward = found_treasure.astype(jnp.float32)

            # randomize position if the agent finds the reward
            new_pos = jnp.where(found_treasure, random_position, new_pos)

            return new_pos, reward

        random_positions = jax.random.choice(rng_key, self.empty_positions, (self._num_agents,))
        new_position, rewards = _step_agent(state.agents_pos, action, random_positions)

        state = state._replace(agents_pos=new_position, time=state.time + 1)

        return state, self.encode_observations(state, action, rewards)

    def encode_observations(self, state: ReturnState, actions, rewards) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                positions - jnp.array([self.view_width // 2, self.view_height // 2]),
                (self.view_width, self.view_height)
            )

        tiles = self.tiles.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(TILE_AGENT)
        tiles = tiles.at[state.treasure_pos[0], state.treasure_pos[1]].set(TILE_TREASURE)
        view = _encode_view(tiles, state.agents_pos)

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

        return TimeStep(
            obs=view,
            time=time,
            last_action=actions,
            last_reward=rewards,
            action_mask=None,
        )

class ReturnClient:
    def __init__(self, env: ReturnEnv):
        self.env = env

        flags = pygame.SRCALPHA
        self.screen = pygame.display.set_mode((800, 800))
        self.surface = pygame.Surface((800, 800), flags=flags)
        self.clock = pygame.time.Clock()

        self.frames = []

    def render(self, state: ReturnState):
        self.surface.fill(pygame.color.Color(40, 40, 40, 100))

        tile_size = 20

        tiles = self.env.tiles.tolist()
        colors = ["grey", "brown", "blue"]

        for x in range(self.env.unpadded_width):
            for y in range(self.env.unpadded_height):
                tx = self.env.pad_width + x
                ty = self.env.pad_height + y

                tile_type = tiles[tx][self.env.height - ty + 1]
                self.screen.fill(colors[tile_type], (x * tile_size, y * tile_size, tile_size, tile_size))

        agents = state.agents_pos.tolist()

        for x, y in agents:
            agent_x = x - self.env.pad_width
            agent_y = (self.env.height - y + 1) - self.env.pad_height

            self.surface.fill(
                (0, 0, 0, 0),
                (agent_x * tile_size - (tile_size * 2),
                agent_y * tile_size - (tile_size * 2), tile_size * 5, tile_size * 5)
            )
            self.screen.fill("yellow", (agent_x * tile_size, agent_y * tile_size, tile_size, tile_size))

        treasure_x = state.treasure_pos[0] - self.env.pad_width
        treasure_y = (self.env.height - state.treasure_pos[1] + 1) - self.env.pad_height
        self.surface.fill(
            "blue",
            (treasure_x * tile_size,
            treasure_y * tile_size, tile_size, tile_size)
        )

        self.clock.tick(10)
        self.screen.blit(self.surface, (0,0))
        pygame.display.flip()

        self.record_frame()

    def record_frame(self):
        img_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.frames.append(img_data)

    def save_video(self):
        frames = np.array(self.frames)
        save_video(frames, "test.mp4", 10)


def demo():
    env = ReturnEnv()

    rng_key = jax.random.PRNGKey(11)
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
        action = None

        if keys[pygame.K_w]:
            action = 0
            # state, ts = env.step(state, jnp.array([0]), rng_key)
        elif keys[pygame.K_s]:
            action = 2
            # state, ts = env.step(state, jnp.array([2]), rng_key)
        elif keys[pygame.K_a]:
            action = 3
            # state, ts = env.step(state, jnp.array([3]), rng_key)
        elif keys[pygame.K_d]:
            action = 1
            # state, ts = env.step(state, jnp.array([1]), rng_key)

        if action is not None:
            action = jnp.full((env.num_agents, 1), action)
            state, ts = env.step(state, action, rng_key)

        # print(ts)
        client.render(state)

    pygame.quit()



if __name__ == '__main__':
    demo()
