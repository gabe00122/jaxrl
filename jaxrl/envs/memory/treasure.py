from functools import cached_property, partial
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
import pygame

from jaxrl.envs.map_generator import generate_perlin_noise_2d
from jaxrl.config import ReturnConfig, TreasureConfig
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.utils.video_writter import save_video

NUM_CLASSES = 5

TILE_EMPTY = 0
TILE_WALL = 1
TILE_TREASURE = 2
TILE_TREASURE_OPEN = 3
TILE_AGENT = 4


class TreasureState(NamedTuple):
    seer_pos: jax.Array       # n length (x, y)
    opener_pos: jax.Array

    time: jax.Array             # ()

    map: jax.Array              # (w, h) tile type id
    spawn_pos: jax.Array        # n length (x, y) spawnable positions, padded with -1
    spawn_count: jax.Array      # () size of spawn_pos


class TreasureEnv(Environment[TreasureState]):
    def __init__(self, config: TreasureConfig) -> None:
        super().__init__()

        self._num_seers = config.num_seers
        self._num_openers = config.num_openers
        self._num_treasures = config.num_treasures

        self.unpadded_width = config.width
        self.unpadded_height = config.height

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.width = self.unpadded_width + self.pad_width
        self.height = self.unpadded_height + self.pad_height

        # self.tiles, self.empty_positions = self._load_template(map_template)


    def _generate_map(self, rng_key):
        noise = generate_perlin_noise_2d(
            (self.unpadded_width, self.unpadded_height), (5, 5), rng_key=rng_key
        )
        noise = noise + generate_perlin_noise_2d(
            (self.unpadded_width, self.unpadded_height), (10, 10), rng_key=rng_key
        )

        tiles = jnp.where(noise > 0.3, TILE_WALL, TILE_EMPTY)

        # get the empty tiles for spawning
        x_spawns, y_spawns = jnp.where(
            tiles == TILE_EMPTY,
            size=self.unpadded_width * self.unpadded_height,
            fill_value=-1,
        )
        spawn_count = jnp.sum(tiles == TILE_EMPTY)

        # pad the tiles
        tiles = jnp.pad(
            tiles,
            pad_width=(
                (self.pad_width, self.pad_width),
                (self.pad_height, self.pad_height),
            ),
            mode="constant",
            constant_values=TILE_WALL,
        )

        # pad the empty tiles
        y_spawns = y_spawns + self.pad_height
        x_spawns = x_spawns + self.pad_width
        spawn_pos = jnp.stack((x_spawns, y_spawns), axis=1)

        return tiles, spawn_pos, spawn_count

    def reset(self, rng_key: jax.Array) -> tuple[TreasureState, TimeStep]:
        map_key, seer_key, opener_key, treasure_key = jax.random.split(rng_key, 4)

        map, spawn_pos, spawn_count = self._generate_map(map_key)

        seer_pos = jax.random.randint(
            seer_key, (self._num_seers,), minval=0, maxval=spawn_count
        )
        opener_pos = jax.random.randint(
            opener_key, (self._num_seers,), minval=0, maxval=spawn_count
        )
        treasure_pos = jax.random.randint(
            treasure_key, (self._num_treasures,), minval=0, maxval=spawn_count
        )

        map = map.at[treasure_pos[:, 0], treasure_pos[:, 1]].set(TILE_TREASURE)

        state = TreasureState(
            map=map,
            spawn_pos=spawn_pos,
            spawn_count=spawn_count,
            seer_pos=seer_pos,
            opener_pos=opener_pos,
            time=jnp.int32(0),
        )

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        return state, self.encode_observations(state, actions, rewards)

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            shape=(self.view_width, self.view_height),
            max_value=NUM_CLASSES,
            dtype=jnp.int8,
        )

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(num_actions=4)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return self._num_seers + self._num_openers

    def step(
        self, state: TreasureState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[TreasureState, TimeStep]:
        seeker_actions = action[:self._num_seers]
        opener_actions = action[self._num_seers:]

        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0, 0))
        def _step_seeker(local_position, local_action):
            directions = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
            new_pos = local_position + directions[local_action]

            new_tile = state.map[new_pos[0], new_pos[1]]

            # don't move if we are moving into a wall
            new_pos = jnp.where(new_tile == TILE_WALL, local_position, new_pos)

            reward = (new_tile == TILE_TREASURE_OPEN).astype(jnp.float32)

            return new_pos, reward

        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0, 0))
        def _step_opener(local_position, local_action):
            directions = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
            new_pos = local_position + directions[local_action]

            new_tile = state.map[new_pos[0], new_pos[1]]

            # don't move if we are moving into a wall
            new_pos = jnp.where(new_tile == TILE_WALL, local_position, new_pos)

            reward = (new_tile == TILE_TREASURE).astype(jnp.float32)

            return new_pos, reward

        random_positions = state.spawn_pos[
            jax.random.randint(
                rng_key, (self._num_agents,), minval=0, maxval=state.spawn_count
            )
        ]
        new_position, rewards = _step_agent(state.agents_pos, action, random_positions)

        state = state._replace(
            agents_pos=new_position,
            found_reward=jnp.logical_or(state.found_reward, rewards),
            time=state.time + 1,
        )

        return state, self.encode_observations(state, action, rewards)

    def encode_observations(self, state: TreasureState, actions, rewards) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                positions - jnp.array([self.view_width // 2, self.view_height // 2]),
                (self.view_width, self.view_height),
            )

        tiles = state.map.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            TILE_AGENT
        )
        tiles = tiles.at[state.treasure_pos[0], state.treasure_pos[1]].set(
            TILE_TREASURE
        )
        view = _encode_view(tiles, state.agents_pos)

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

        return TimeStep(
            obs=view,
            time=time,
            last_action=actions,
            last_reward=rewards,
            action_mask=None,
        )


class TreasureClient:
    def __init__(self, env: TreasureEnv):
        self.env = env

        self.screen_width = 800
        self.screen_height = 800

        flags = pygame.SRCALPHA
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.surface = pygame.Surface(
            (self.screen_width, self.screen_height), flags=flags
        )
        self.clock = pygame.time.Clock()

        self.frames = []

        self._tile_size = self.screen_width // self.env.unpadded_width

    def render(self, state: TreasureState):
        self.surface.fill(pygame.color.Color(40, 40, 40, 100))

        tiles = state.map.tolist()
        colors = ["grey", "brown", "blue"]

        for x in range(self.env.unpadded_width):
            for y in range(self.env.unpadded_height):
                tx = self.env.pad_width + x
                ty = self.env.pad_height + y

                tile_type = tiles[tx][ty]
                self._draw_tile(self.screen, colors[tile_type], tx, ty)

        agents = state.agents_pos.tolist()
        found_reward = state.found_reward.tolist()

        for (x, y), r in zip(agents, found_reward):
            self._draw_tile(self.screen, "yellow" if not r else "purple", x, y)
            self._draw_tile(self.surface, (0, 0, 0, 0), x, y, 5, 5)

        self._draw_tile(
            self.screen,
            "blue",
            state.treasure_pos[0].item(),
            state.treasure_pos[1].item(),
        )

        self.clock.tick(10)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

        # self.record_frame()

    def _tile_to_screen(self, x: int, y: int):
        return x - self.env.pad_width, (self.env.height - y + 1) - self.env.pad_height

    def _draw_tile(self, surface, color, x, y, width: int = 1, height: int = 1):
        x, y = self._tile_to_screen(x, y)

        half_width = width // 2
        half_height = height // 2

        surface.fill(
            color,
            (
                (x - half_width) * self._tile_size,
                (y - half_height) * self._tile_size,
                width * self._tile_size,
                height * self._tile_size,
            ),
        )

    def record_frame(self):
        img_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.frames.append(img_data)

    def save_video(self):
        frames = np.array(self.frames)
        save_video(frames, "videos/test.mp4", 10)


def demo():
    env = TreasureEnv(TreasureConfig())

    rng_key = jax.random.PRNGKey(11)
    state, timestep = env.reset(rng_key)

    client = TreasureClient(env)

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


if __name__ == "__main__":
    demo()
