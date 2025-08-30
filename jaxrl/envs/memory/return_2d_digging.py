from functools import cached_property, partial
from typing import NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
import pygame

from jaxrl.envs.map_generator import generate_perlin_noise_2d
from jaxrl.config import ReturnConfig, ReturnDiggingConfig
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.utils.video_writter import save_video
from jaxrl.envs.gridworld.renderer import (
    GridRenderState,
    TILE_TREASURE as GW_TILE_TREASURE,
)

NUM_CLASSES = 5

TILE_EMPTY = 0
TILE_WALL = 1
TILE_SOFT_WALL = 2
TILE_TREASURE = 3
TILE_AGENT = 4


class ReturnDiggingState(NamedTuple):
    agents_pos: jax.Array
    agents_timeout: jax.Array
    found_reward: jax.Array

    treasure_pos: jax.Array
    time: jax.Array

    map: jax.Array
    spawn_pos: jax.Array
    spawn_count: jax.Array
    
    rewards: jax.Array


class ReturnDiggingEnv(Environment[ReturnDiggingState]):
    def __init__(self, config: ReturnDiggingConfig, length: int) -> None:
        super().__init__()

        self._length = length
        self._num_agents = config.num_agents

        self.unpadded_width = config.width
        self.unpadded_height = config.height

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.width = self.unpadded_width + self.pad_width
        self.height = self.unpadded_height + self.pad_height

        self.mapgen_threshold = config.mapgen_threshold
        self.digging_timeout = config.digging_timeout


    def _generate_map(self, rng_key):
        noise_key, amplitude_key, rng_key = jax.random.split(rng_key, 3)

        res = [4, 5, 8, 10]
        amplitude = jax.random.dirichlet(amplitude_key, jnp.ones((5,)))
        noise = generate_perlin_noise_2d(
            (self.unpadded_width, self.unpadded_height), (2, 2), rng_key=noise_key
        ) * amplitude[0]

        for i, r in enumerate(res):
            noise_key, rng_key = jax.random.split(rng_key)
            noise = noise + generate_perlin_noise_2d(
                (self.unpadded_width, self.unpadded_height), (r, r), rng_key=noise_key
            ) * amplitude[i+1]
        
        tiles = jnp.where(noise > 0.05, TILE_SOFT_WALL, TILE_EMPTY)

        # first_key, second_key = jax.random.split(rng_key)
        # noise = generate_perlin_noise_2d(
        #     (self.unpadded_width, self.unpadded_height), (5, 5), rng_key=first_key
        # )
        # noise = noise + generate_perlin_noise_2d(
        #     (self.unpadded_width, self.unpadded_height), (10, 10), rng_key=second_key
        # )

        # tiles = jnp.where(noise > self.mapgen_threshold, TILE_SOFT_WALL, TILE_EMPTY)

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

    def reset(self, rng_key: jax.Array) -> tuple[ReturnDiggingState, TimeStep]:
        map_key, pos_key = jax.random.split(rng_key)

        map, spawn_pos, spawn_count = self._generate_map(map_key)

        positions = jax.random.randint(
            pos_key, (1 + self.num_agents,), minval=0, maxval=spawn_count
        )
        treasure_pos = spawn_pos[positions[0]]
        agents_pos = spawn_pos[positions[1:]]

        state = ReturnDiggingState(
            map=map,
            spawn_pos=spawn_pos,
            spawn_count=spawn_count,
            treasure_pos=treasure_pos,
            agents_pos=agents_pos,
            agents_timeout=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            found_reward=jnp.zeros((self.num_agents,), dtype=jnp.bool),
            time=jnp.int32(0),
            rewards=jnp.float32(0.0),
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
        return self._num_agents

    def step(
        self, state: ReturnDiggingState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[ReturnDiggingState, TimeStep]:
        @partial(jax.vmap, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0))
        def _step_agent(local_position, timeout, local_action, random_position):
            def _step_timeout(local_position, timeout, local_action, random_position):
                return local_position, local_position, timeout - 1, 0.0

            def _step_move(local_position, timeout, local_action, random_position):
                directions = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
                target_pos = local_position + directions[local_action]

                new_tile = state.map[target_pos[0], target_pos[1]]

                # don't move if we are moving into a wall
                new_pos = jnp.where(jnp.logical_or(new_tile == TILE_WALL, new_tile == TILE_SOFT_WALL), local_position, target_pos)

                found_treasure = jnp.all(new_pos == state.treasure_pos)
                reward = found_treasure.astype(jnp.float32)

                # randomize position if the agent finds the reward
                new_pos = jnp.where(found_treasure, random_position, new_pos)

                # sets a timeout of the tile is dug
                timeout = jnp.where(new_tile == TILE_SOFT_WALL, self.digging_timeout, 0)

                return new_pos, target_pos, timeout, reward

            return jax.lax.cond(timeout > 0, _step_timeout, _step_move, local_position, timeout, local_action, random_position)

        random_positions = state.spawn_pos[
            jax.random.randint(
                rng_key, (self._num_agents,), minval=0, maxval=state.spawn_count
            )
        ]
        new_position, target_pos, timeout, rewards = _step_agent(state.agents_pos, state.agents_timeout, action, random_positions)

        # dig actions
        target_tiles = state.map[target_pos[:, 0], target_pos[:, 1]]
        map = state.map.at[target_pos[:, 0], target_pos[:, 1]].set(jnp.where(target_tiles == TILE_SOFT_WALL, TILE_EMPTY, target_tiles))
        # /dig actions

        state = state._replace(
            agents_pos=new_position,
            agents_timeout=timeout,
            found_reward=jnp.logical_or(state.found_reward, rewards),
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
            map=map,
        )

        return state, self.encode_observations(state, action, rewards)

    def encode_observations(self, state: ReturnDiggingState, actions, rewards) -> TimeStep:
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
            terminated=jnp.equal(time, self._length - 1)
        )

    def create_placeholder_logs(self):
        return {
            "rewards": jnp.float32(0.0)
        }

    def create_logs(self, state: ReturnDiggingState):
        return {
            "rewards": state.rewards
        }

    # Shared renderer adapter
    def get_render_state(self, state: ReturnDiggingState) -> GridRenderState:
        # Map already uses empty/wall/soft-wall; overlay treasure to unified id
        tilemap = state.map
        tilemap = tilemap.at[state.treasure_pos[0], state.treasure_pos[1]].set(GW_TILE_TREASURE)

        return GridRenderState(
            tilemap=tilemap,
            pad_width=self.pad_width,
            pad_height=self.pad_height,
            unpadded_width=self.unpadded_width,
            unpadded_height=self.unpadded_height,
            agent_positions=state.agents_pos,
            agent_types=None,
            agent_colors=None,
            view_width=self.view_width,
            view_height=self.view_height,
        )


class ReturnDiggingClient:
    def __init__(self, env: ReturnDiggingEnv):
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

    def render(self, state: ReturnDiggingState, timestep):
        self.surface.fill(pygame.color.Color(40, 40, 40, 100))

        tiles = state.map.tolist()
        colors = ["grey", "grey", "brown", "blue"]

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

        self.record_frame()

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
    env = ReturnDiggingEnv(ReturnDiggingConfig(
        mapgen_threshold=0.05
    ))

    rng_key = jax.random.PRNGKey(11)
    state, timestep = env.reset(rng_key)

    client = ReturnDiggingClient(env)

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
            action = jnp.full((env.num_agents,), action)
            state, ts = env.step(state, action, rng_key)

        # print(ts)
        client.render(state)

    pygame.quit()


if __name__ == "__main__":
    demo()
