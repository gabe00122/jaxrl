from functools import cached_property, partial
from typing import NamedTuple, Literal

import jax
from jax import numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict

from jaxrl.envs.map_generator import (
    fractal_noise,
    generate_decor_tiles,
    choose_positions,
)
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from mapox import TimeStep
from jaxrl.envs.gridworld.renderer import GridRenderSettings, GridRenderState
import jaxrl.envs.gridworld.constance as GW


class ReturnDiggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["return_digging"] = "return_digging"

    num_agents: int = 1
    num_flags: int = 1

    width: int = 40
    height: int = 40
    view_width: int = 11
    view_height: int = 11

    mapgen_threshold: float = 0.3
    digging_timeout: int = 5
    treasure_reward: float = 1.0

    eval_map: bool = False


class ReturnDiggingState(NamedTuple):
    agents_pos: jax.Array
    agents_timeout: jax.Array
    found_reward: jax.Array

    time: jax.Array

    map: jax.Array
    spawn_pos: jax.Array
    spawn_count: jax.Array

    rewards: jax.Array


class ReturnDiggingEnv(Environment[ReturnDiggingState]):
    def __init__(self, config: ReturnDiggingConfig, length: int) -> None:
        super().__init__()

        self._config = config
        self._length = length
        self._num_agents = config.num_agents
        self.num_flags = config.num_flags

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
        self.treasure_reward = config.treasure_reward

        self._action_mask = GW.make_action_mask(
            [
                GW.MOVE_UP,
                GW.MOVE_RIGHT,
                GW.MOVE_DOWN,
                GW.MOVE_LEFT,
            ],
            self.num_agents,
        )

    def _generate_map(self, rng_key):
        walls_key, decor_key, rng_key = jax.random.split(rng_key, 3)
        noise = fractal_noise(self.unpadded_width, self.unpadded_height, [2, 4, 5, 8, 10], walls_key)

        tiles = generate_decor_tiles(self.unpadded_width, self.unpadded_height, decor_key)
        tiles = jnp.where(noise > 0.05, jnp.int8(GW.TILE_DESTRUCTIBLE_WALL), tiles)

        # get the empty tiles for spawning
        x_spawns, y_spawns = jnp.where(
            tiles == GW.TILE_EMPTY,
            size=self.unpadded_width * self.unpadded_height,
            fill_value=jnp.int8(-1),
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

        unpadded_map = map[
            self.pad_width : -self.pad_width, self.pad_height : -self.pad_height
        ]

        pos_x, pos_y = choose_positions(
            unpadded_map,
            self.num_flags + self.num_agents,
            pos_key,
            replace=False,
        )

        pos_x = pos_x + self.pad_width
        pos_y = pos_y + self.pad_height
        positions = jnp.stack((pos_x, pos_y), axis=1)
        flag_pos = positions[: self.num_flags]
        agents_pos = positions[self.num_flags :]

        if self._config.eval_map:
            o = map
            map = map.at[36:45, 10:30].set(GW.TILE_DESTRUCTIBLE_WALL)
            map = map.at[42:45, 17:23].set(o[42:45, 17:23])
            agents_pos = agents_pos.at[0].set([44, 22])
            map = map.at[43, 18].set(GW.TILE_FLAG)
        else:
            map = map.at[flag_pos[:, 0], flag_pos[:, 1]].set(GW.TILE_FLAG)

        state = ReturnDiggingState(
            map=map,
            spawn_pos=spawn_pos,
            spawn_count=spawn_count,
            agents_pos=agents_pos,
            agents_timeout=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            found_reward=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            time=jnp.int32(0),
            rewards=jnp.float32(0.0),
        )

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        return state, self.encode_observations(state, actions, rewards)

    def load_map(self, map: str):
        tiles = np.zeros((self.unpadded_width, self.unpadded_height), dtype=np.int8)

        x = 0
        y = self.unpadded_height

        agent_positions = []
        spawn_positions = []

        for c in map:
            if c == '\n':
                x = 0
                y -= 1
            else:
                match x:
                    case 'x':
                        tiles[x, y] = GW.TILE_DESTRUCTIBLE_WALL
                    case 'a':
                        agent_positions.append([self.pad_width+x, self.pad_height+y])
                    case 'f':
                        tiles[x, y] = GW.TILE_FLAG
                    case _:
                        spawn_positions.append([self.pad_width+x, self.pad_height+y])

                x += 1

        tiles = jnp.asarray(tiles)
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

        state = ReturnDiggingState(
            map=tiles,
            spawn_pos=jnp.array(spawn_positions, jnp.int32),
            spawn_count=jnp.int32(len(spawn_positions)),
            agents_pos=jnp.arange(agent_positions, jnp.int32),
            agents_timeout=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            found_reward=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            time=jnp.int32(0),
            rewards=jnp.float32(0.0),
        )

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        return state, self.encode_observations(state, actions, rewards)


    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return GW.make_obs_spec(self.view_width, self.view_height)

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
                target_pos = local_position + GW.DIRECTIONS[local_action]

                new_tile = state.map[target_pos[0], target_pos[1]]

                # don't move if we are moving into a wall
                new_pos = jnp.where(
                    jnp.logical_or(
                        new_tile == GW.TILE_WALL, new_tile == GW.TILE_DESTRUCTIBLE_WALL
                    ),
                    local_position,
                    target_pos,
                )

                found_treasure = new_tile == GW.TILE_FLAG
                reward = jnp.where(found_treasure, self.treasure_reward, 0.0)

                # randomize position if the agent finds the reward
                new_pos = jnp.where(found_treasure, random_position, new_pos)

                # sets a timeout of the tile is dug
                timeout = jnp.where(
                    new_tile == GW.TILE_DESTRUCTIBLE_WALL, self.digging_timeout, 0
                )

                return new_pos, target_pos, timeout, reward

            return jax.lax.cond(
                timeout > 0,
                _step_timeout,
                _step_move,
                local_position,
                timeout,
                local_action,
                random_position,
            )

        random_positions = state.spawn_pos[
            jax.random.randint(
                rng_key, (self._num_agents,), minval=0, maxval=state.spawn_count
            )
        ]
        new_position, target_pos, timeout, rewards = _step_agent(
            state.agents_pos, state.agents_timeout, action, random_positions
        )

        # dig actions
        target_tiles = state.map[target_pos[:, 0], target_pos[:, 1]]
        map = state.map.at[target_pos[:, 0], target_pos[:, 1]].set(
            jnp.where(target_tiles == GW.TILE_DESTRUCTIBLE_WALL, GW.TILE_EMPTY, target_tiles)
        )
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

    def _render_tiles(self, state: ReturnDiggingState):
        tiles = state.map
        tiles = tiles.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(GW.AGENT_GENERIC)

        directions = jnp.zeros_like(tiles, dtype=jnp.int8)
        teams = jnp.zeros_like(tiles, dtype=jnp.int8)
        health = jnp.zeros_like(tiles, dtype=jnp.int8)

        return jnp.concatenate(
            (tiles[..., None], directions[..., None], teams[..., None], health[..., None]),
            axis=-1,
        )

    def encode_observations(
        self, state: ReturnDiggingState, actions, rewards
    ) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                (
                    positions[0] - self.view_width // 2,
                    positions[1] - self.view_height // 2,
                    0,
                ),
                (self.view_width, self.view_height, self.observation_spec.shape[-1]),
            )

        tiles = self._render_tiles(state)
        view = _encode_view(tiles, state.agents_pos)

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

        return TimeStep(
            obs=view,
            time=time,
            last_action=actions,
            last_reward=rewards,
            action_mask=self._action_mask,
            terminated=jnp.equal(time, self._length - 1),
        )

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: ReturnDiggingState):
        return {"rewards": state.rewards}

    def get_render_state(self, state: ReturnDiggingState) -> GridRenderState:
        tiles = self._render_tiles(state)

        return GridRenderState(
            tilemap=tiles,
            agent_positions=state.agents_pos,
        )

    def get_render_settings(self) -> GridRenderSettings:
        return GridRenderSettings(
            tile_width=self.unpadded_width,
            tile_height=self.unpadded_height,
            view_width=self.view_width,
            view_height=self.view_height,
        )
