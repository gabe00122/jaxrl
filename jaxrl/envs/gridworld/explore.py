from functools import cached_property, partial
from typing import NamedTuple, Literal

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.envs.gridworld.renderer import GridRenderState
import jaxrl.envs.gridworld.constance as GW


class ExploreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["explore"] = "explore"

    num_agents: int = 1

    width: int = 40
    height: int = 40
    view_width: int = 5
    view_height: int = 5


class ExploreState(NamedTuple):
    agents_pos: jax.Array
    time: jax.Array

    map: jax.Array
    reward_map: jax.Array

    rewards: jax.Array


class ExploreEnv(Environment[ExploreState]):
    def __init__(self, config: ExploreConfig, length: int) -> None:
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

    def _generate_map(self, rng_key):
        # pad the tiles
        # tiles = jnp.full((self.width, self.height), GW.TILE_WALL, dtype=jnp.int32)
        reward_map = jax.random.uniform(
            rng_key, (self.unpadded_width, self.unpadded_height)
        )
        reward_map = jnp.where(reward_map < 0.01, reward_map, 0.0) * 20
        reward_map = reward_map * reward_map

        tiles = jnp.where(reward_map == 0, GW.TILE_EMPTY, GW.TILE_FLAG)

        tiles = jnp.pad(
            tiles,
            pad_width=(
                (self.pad_width, self.pad_width),
                (self.pad_height, self.pad_height),
            ),
            mode="constant",
            constant_values=GW.TILE_WALL,
        )

        return tiles, reward_map

    def reset(self, rng_key: jax.Array) -> tuple[ExploreState, TimeStep]:
        map_key, pos_key = jax.random.split(rng_key)

        map, reward_map = self._generate_map(map_key)

        positions = (
            jax.random.randint(
                pos_key, (self.num_agents, 2), minval=0, maxval=self.unpadded_width
            )
            + self.pad_width
        )

        state = ExploreState(
            map=map,
            reward_map=reward_map,
            agents_pos=positions,
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
        self, state: ExploreState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[ExploreState, TimeStep]:
        @partial(jax.vmap, in_axes=(0, 0), out_axes=0)
        def _step_agent(local_position, local_action):
            target_pos = local_position + GW.DIRECTIONS[local_action]

            new_tile = state.map[target_pos[0], target_pos[1]]

            # don't move if we are moving into a wall
            new_pos = jnp.where(new_tile == GW.TILE_WALL, local_position, target_pos)

            return new_pos

        new_position = _step_agent(state.agents_pos, action)

        unpadded_positions = new_position - self.pad_width
        rewards = state.reward_map[unpadded_positions[:, 0], unpadded_positions[:, 1]]
        state = state._replace(
            agents_pos=new_position,
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
        )

        return state, self.encode_observations(state, action, rewards)

    def encode_observations(self, state: ExploreState, actions, rewards) -> TimeStep:
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

    def create_logs(self, state: ExploreState):
        return {"rewards": state.rewards}

    def get_render_state(self, state: ExploreState) -> GridRenderState:
        tilemap = state.map

        tilemap = tilemap.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            GW.AGENT_GENERIC
        )

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
