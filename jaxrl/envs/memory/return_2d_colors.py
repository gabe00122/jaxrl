from functools import cached_property, partial
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
import pygame

from jaxrl.envs.client import EnvironmentClient
from jaxrl.envs.map_generator import generate_perlin_noise_2d
from jaxrl.config import ReturnColorConfig
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.utils.video_writter import save_video

NUM_CLASSES = 3

TILE_EMPTY = 0
TILE_WALL = 1
TILE_TREASURE = 2
TILE_AGENT = 3

NUM_COLORS = 4

class ReturnColorState(NamedTuple):
    agents_pos: jax.Array
    agent_color: jax.Array
    found_reward: jax.Array

    treasure_pos: jax.Array
    time: jax.Array

    map: jax.Array
    spawn_pos: jax.Array
    spawn_count: jax.Array


class ReturnColorEnv(Environment[ReturnColorState]):
    def __init__(self, config: ReturnColorConfig) -> None:
        super().__init__()

        self._num_agents = config.num_agents

        self.unpadded_width = config.width
        self.unpadded_height = config.height

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.width = self.unpadded_width + self.pad_width
        self.height = self.unpadded_height + self.pad_height

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

    def reset(self, rng_key: jax.Array) -> tuple[ReturnColorState, TimeStep]:
        map_key, pos_key = jax.random.split(rng_key)

        map, spawn_pos, spawn_count = self._generate_map(map_key)

        positions = jax.random.randint(
            pos_key, (1 + self.num_agents,), minval=0, maxval=spawn_count
        )
        treasure_pos = spawn_pos[positions[0]]
        agents_pos = spawn_pos[positions[1:]]
        agent_color = jnp.zeros((self.num_agents,), dtype=jnp.int32)

        state = ReturnColorState(
            map=map,
            spawn_pos=spawn_pos,
            spawn_count=spawn_count,
            treasure_pos=treasure_pos,
            agents_pos=agents_pos,
            agent_color=agent_color,
            found_reward=jnp.zeros((self.num_agents,), dtype=jnp.bool),
            time=jnp.int32(0),
        )

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        action_mask = jnp.repeat((jnp.arange(self.action_spec.num_actions) < 4)[None, :], self.num_agents, axis=0) # move actions

        return state, self.encode_observations(state, actions, rewards, action_mask)

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            shape=(self.view_width, self.view_height),
            max_value=NUM_CLASSES + NUM_COLORS,
            dtype=jnp.int8,
        )

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(num_actions=4 + NUM_COLORS)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return self._num_agents

    def _move_step(self, state: ReturnColorState, action: jax.Array, rng_key: jax.Array):
        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0, 0))
        def _step_agent(local_position, local_action, random_position):
            directions = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
            new_pos = local_position + directions[local_action]

            new_tile = state.map[new_pos[0], new_pos[1]]

            # don't move if we are moving into a wall
            new_pos = jnp.where(new_tile == TILE_WALL, local_position, new_pos)

            found_treasure = jnp.all(
                new_pos == state.treasure_pos
            )  # new_tile == TILE_TREASURE
            reward = found_treasure.astype(jnp.float32)

            # randomize position if the agent finds the reward
            new_pos = jnp.where(found_treasure, random_position, new_pos)

            return new_pos, reward

        random_positions = state.spawn_pos[
            jax.random.randint(
                rng_key, (self._num_agents,), minval=0, maxval=state.spawn_count
            )
        ]
        new_position, rewards = _step_agent(state.agents_pos, action, random_positions)

        action_mask = jnp.repeat((jnp.arange(self.action_spec.num_actions) >= 4)[None, :], self.num_agents, axis=0) # color actions

        state = state._replace(
            agents_pos=new_position,
            found_reward=jnp.logical_or(state.found_reward, rewards),
            time=state.time + 1,
        )

        return state, rewards, action_mask

    def _color_step(self, state: ReturnColorState, action: jax.Array, rng_key: jax.Array):
        action = action - 4 # remove the movement action to get the color id's

        action_mask = jnp.repeat((jnp.arange(self.action_spec.num_actions) < 4)[None, :], self.num_agents, axis=0) # move actions

        state = state._replace(
            agent_color=action,
            time=state.time + 1
        )
        return state, jnp.zeros((self.num_agents,)), action_mask

    def step(
        self, state: ReturnColorState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[ReturnColorState, TimeStep]:
        is_color_step = jnp.equal(state.time % 2, 1)

        state, rewards, action_mask = jax.lax.cond(
            is_color_step,
            self._color_step,
            self._move_step,
            state, action, rng_key
        )

        return state, self.encode_observations(state, action, rewards, action_mask)

    def encode_observations(self, state: ReturnColorState, actions, rewards, action_mask) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                positions - jnp.array([self.view_width // 2, self.view_height // 2]),
                (self.view_width, self.view_height),
            )

        tiles = state.map.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            TILE_AGENT + state.agent_color
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
            action_mask=action_mask,
        )

agent_color_names = ["darkorchid1", "darkorchid2", "darkorchid3", "darkorchid4"]

class ReturnColorClient(EnvironmentClient[ReturnColorState]):
    def __init__(self, env: ReturnColorEnv):
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

    def render(self, state: ReturnColorState):
        if state.time % 2 == 0:
            return

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
        agent_colors = state.agent_color.tolist()

        for (x, y), c in zip(agents, agent_colors):
            self._draw_tile(self.screen, agent_color_names[c], x, y)
            self._draw_tile(self.surface, (0, 0, 0, 0), x, y, self.env.view_width, self.env.view_height)

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
