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
from jaxrl.envs.gridworld.renderer import GridRenderState
import jaxrl.envs.gridworld.constance as GW 


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
        
        tiles = jnp.where(noise > 0.05, GW.TILE_SOFT_WALL, GW.TILE_EMPTY)

        # get the empty tiles for spawning
        x_spawns, y_spawns = jnp.where(
            tiles == GW.TILE_EMPTY,
            size=self.unpadded_width * self.unpadded_height,
            fill_value=-1,
        )
        spawn_count = jnp.sum(tiles == GW.TILE_EMPTY)

        # pad the tiles
        tiles = jnp.pad(
            tiles,
            pad_width=(
                (self.pad_width, self.pad_width),
                (self.pad_height, self.pad_height),
            ),
            mode="constant",
            constant_values=GW.TILE_WALL,
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
            max_value=GW.NUM_TYPES,
            dtype=jnp.int8,
        )

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(num_actions=GW.NUM_ACTIONS)

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
                new_pos = jnp.where(jnp.logical_or(new_tile == GW.TILE_WALL, new_tile == GW.TILE_SOFT_WALL), local_position, target_pos)

                found_treasure = jnp.all(new_pos == state.treasure_pos)
                reward = found_treasure.astype(jnp.float32)

                # randomize position if the agent finds the reward
                new_pos = jnp.where(found_treasure, random_position, new_pos)

                # sets a timeout of the tile is dug
                timeout = jnp.where(new_tile == GW.TILE_SOFT_WALL, self.digging_timeout, 0)

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
        map = state.map.at[target_pos[:, 0], target_pos[:, 1]].set(jnp.where(target_tiles == GW.TILE_SOFT_WALL, GW.TILE_EMPTY, target_tiles))
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
            GW.AGENT_GENERIC
        )
        tiles = tiles.at[state.treasure_pos[0], state.treasure_pos[1]].set(
            GW.TILE_TREASURE
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

    def get_render_state(self, state: ReturnDiggingState) -> GridRenderState:
        tilemap = state.map
        tilemap = tilemap.at[state.treasure_pos[0], state.treasure_pos[1]].set(GW.TILE_TREASURE)

        return GridRenderState(
            tilemap=tilemap,
            pad_width=self.pad_width,
            pad_height=self.pad_height,
            unpadded_width=self.unpadded_width,
            unpadded_height=self.unpadded_height,
            agent_positions=state.agents_pos,
            agent_types=None,
            view_width=self.view_width,
            view_height=self.view_height,
        )

