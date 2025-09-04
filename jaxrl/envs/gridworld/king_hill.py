from functools import cached_property, partial
from typing import NamedTuple, Literal

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.envs.gridworld.renderer import GridRenderState
from jaxrl.envs.gridworld.util import Position, unique_mask
import jaxrl.envs.gridworld.constance as GW


class KingHillConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["king_hill"] = "king_hill"

    num_agents: int = 2
    num_flags: int = 8

    width: int = 40
    height: int = 40
    view_width: int = 5
    view_height: int = 5


class KingHillState(NamedTuple):
    agents_pos: Position

    time: jax.Array
    map: jax.Array
    rewards: jax.Array


class KingHillEnv(Environment[KingHillState]):
    def __init__(self, config: KingHillConfig, length: int) -> None:
        super().__init__()

        self._length = length
        self._num_agents = config.num_agents
        self._config = config

        self.width = config.width
        self.height = config.height

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.padded_width = self.width + self.pad_width * 2
        self.padded_height = self.height + self.pad_height * 2

    def _pad_tiles(self, tiles, fill):
        # pads tiles so the observation can just be a slice
        return jnp.pad(
            tiles,
            pad_width=(
                (self.pad_width, self.pad_width),
                (self.pad_height, self.pad_height),
            ),
            mode="constant",
            constant_values=fill,
        )

    def _generate_map(self, rng_key):
        tile_ids = jnp.array([GW.TILE_EMPTY, GW.TILE_DECOR_1, GW.TILE_DECOR_2, GW.TILE_DECOR_3, GW.TILE_DECOR_4])
        tile_probs = jnp.array([0.85, 0.05, 0.05, 0.025, 0.025])
        # tiles = jnp.zeros((self.width, self.height), dtype=jnp.int8)
        tiles = jax.random.choice(rng_key, tile_ids, (self.width, self.height), p=tile_probs)

        tiles = self._pad_tiles(tiles, GW.TILE_WALL)

        return tiles

    def reset(self, rng_key: jax.Array) -> tuple[KingHillState, TimeStep]:
        map_key, pos_key = jax.random.split(rng_key)

        map = self._generate_map(map_key)

        # place objective
        map = map.at[self.padded_width // 2, self.padded_height // 2].set(GW.TILE_FLAG)
        #

        agent_pos = Position(data=jnp.array([
            [0, 1, 2, 3],
            [0, 0, 0, 0]
        ]) + jnp.array([[self.pad_width], [self.pad_height]]))

        state = KingHillState(
            agents_pos=agent_pos,
            map=map,
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

    
    def calculate_movement(self, state: KingHillState, action: jax.Array):
        proposed_position = jnp.where(action < 4, state.agents_pos.move(action).data, state.agents_pos.data)
        proposed_tiles = state.map[proposed_position[0], proposed_position[1]]
        
        # only move to positions with no blocking tile or agent
        proposed_position = jnp.where(proposed_tiles != GW.TILE_WALL, proposed_position, state.agents_pos.data)
        
        position_keys = proposed_position[0] * self.height + proposed_position[1]
        unique_dest_mask = unique_mask(position_keys)

        # only move to destinations that are unique
        proposed_position = jnp.where(unique_dest_mask, proposed_position, state.agents_pos.data)

        return Position(data=proposed_position)


    def step(
        self, state: KingHillState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[KingHillState, TimeStep]:
        new_position = self.calculate_movement(state, action)

        rewards = jnp.zeros((4,))

        state = state._replace(
            agents_pos=new_position,
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
        )

        return state, self.encode_observations(state, action, rewards)

    def encode_observations(
        self, state: KingHillState, actions, rewards
    ) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 1))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                (positions.x - self.view_width // 2, positions.y - self.view_height // 2),
                (self.view_width, self.view_height),
            )

        tiles = state.map.at[state.agents_pos.x, state.agents_pos.y].set(
            GW.AGENT_GENERIC
        )

        view = _encode_view(tiles, state.agents_pos)

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

        return TimeStep(
            obs=view,
            time=time,
            last_action=actions,
            last_reward=rewards,
            action_mask=None,
            terminated=jnp.equal(time, self._length - 1),
        )

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: KingHillState):
        return {"rewards": state.rewards}

    def get_render_state(self, state: KingHillState) -> GridRenderState:
        tilemap = state.map

        tilemap = tilemap.at[state.agents_pos.x, state.agents_pos.y].set(
            GW.AGENT_GENERIC
        )

        x = state.agents_pos.x[:, None]
        y = state.agents_pos.y[:, None]
        agent_pos = jnp.concatenate((x, y), axis=-1)

        return GridRenderState(
            tilemap=tilemap,
            pad_width=self.pad_width,
            pad_height=self.pad_height,
            unpadded_width=self.width,
            unpadded_height=self.height,
            agent_positions=agent_pos,#state.agents_pos,
            agent_types=jnp.array([GW.AGENT_RED_KNIGHT_RIGHT, GW.AGENT_RED_KNIGHT_RIGHT, GW.AGENT_BLUE_KNIGHT_RIGHT, GW.AGENT_BLUE_KNIGHT_RIGHT]),
            view_width=self.view_width,
            view_height=self.view_height,
        )
