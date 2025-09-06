from functools import cached_property, partial
from typing import NamedTuple, Literal

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict
from jaxrl.envs.environment import Environment
from jaxrl.envs.map_generator import generate_decor_tiles
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.envs.gridworld.renderer import GridRenderState
import jaxrl.envs.gridworld.constance as GW


class TravelingSalesmanConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["traveling_salesman"] = "traveling_salesman"

    num_agents: int = 2
    num_flags: int = 8

    width: int = 40
    height: int = 40
    view_width: int = 5
    view_height: int = 5


class TravelingSalesmanState(NamedTuple):
    agents_pos: jax.Array  # (n, 2)
    flag_available: jax.Array # [agent, flag]

    time: jax.Array

    map: jax.Array
    flag_index_map: jax.Array

    rewards: jax.Array


class TravelingSalesmanEnv(Environment[TravelingSalesmanState]):
    def __init__(self, config: TravelingSalesmanConfig, length: int) -> None:
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


    def _random_positions(self, rng_key: jax.Array, count: int, replace: bool = True, pad: bool = True) -> jax.Array:
        # This function assumes an empty map and does not account for walls
        indices = jax.random.choice(
            rng_key,
            self.width * self.height,
            (count,),
            replace=replace
        )

        x = indices % self.width
        y = indices // self.width

        if pad:
            x = x + self.pad_width
            y = y + self.pad_height
        # return (count, 2) array of (x, y)
        return jnp.stack((x, y), axis=-1)

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
        decor_key, flag_key = jax.random.split(rng_key)
        tiles = generate_decor_tiles(self.width, self.height, decor_key)

        flag_pos = self._random_positions(flag_key, self._config.num_flags, replace=False, pad=False)

        tiles = tiles.at[flag_pos[:, 0], flag_pos[:, 1]].set(GW.TILE_FLAG)

        flag_index_map = jnp.full_like(tiles, 0, dtype=jnp.int32)
        flag_index_map = flag_index_map.at[flag_pos[:, 0], flag_pos[:, 1]].set(
            jnp.arange(self._config.num_flags, dtype=jnp.int8)
        )

        tiles = self._pad_tiles(tiles, GW.TILE_WALL)
        flag_index_map = self._pad_tiles(flag_index_map, 0)

        return tiles, flag_index_map

    def reset(self, rng_key: jax.Array) -> tuple[TravelingSalesmanState, TimeStep]:
        map_key, pos_key = jax.random.split(rng_key)

        map, flag_index_map = self._generate_map(map_key)

        agent_pos = self._random_positions(pos_key, self._config.num_agents, replace=False, pad=True)

        state = TravelingSalesmanState(
            agents_pos=agent_pos,
            flag_available=jnp.full((self.num_agents, self._config.num_flags), True, dtype=jnp.bool_),
            map=map,
            flag_index_map=flag_index_map,
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
        self, state: TravelingSalesmanState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[TravelingSalesmanState, TimeStep]:
        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0, 0, 0))
        def _step_agent(local_position: jax.Array, local_action: jax.Array, flag_available: jax.Array):
            # local_position: (2,), returns (2,)
            # Move only for movement actions; otherwise stay
            proposed = jnp.where(
                local_action < 4,
                local_position + GW.DIRECTIONS[local_action],
                local_position,
            )

            new_tile = state.map[proposed[0], proposed[1]]

            # don't move if we are moving into a wall
            new_pos = jnp.where(new_tile == GW.TILE_WALL, local_position, proposed)

            flag_index = state.flag_index_map[new_pos[0], new_pos[1]]
            current_flag_available = flag_available[flag_index]

            found_flag = jnp.logical_and(
                state.map[new_pos[0], new_pos[1]] == GW.TILE_FLAG,
                current_flag_available,
            )

            def on_found_flag(flag_available, flag_index):
                flag_available = flag_available.at[flag_index].set(False)

                # reset once all flags taken
                is_all_taken = jnp.all(jnp.logical_not(flag_available))

                flag_available = jax.lax.cond(
                    is_all_taken,
                    lambda _: jnp.full((self._config.num_flags,), True, dtype=jnp.bool_),
                    lambda avail: avail,
                    flag_available,
                )
                reward = jnp.array(1.0 / 3)

                return reward, flag_available

            def on_not_found_flag(flag_available, flag_index):
                return jnp.array(0.0), flag_available

            reward, new_flags = jax.lax.cond(
                found_flag, on_found_flag, on_not_found_flag, flag_available, flag_index
            )

            return new_pos, reward, new_flags

        new_position, rewards, new_flags = _step_agent(state.agents_pos, action, state.flag_available)

        state = state._replace(
            agents_pos=new_position,
            flag_available=new_flags,
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
        )

        return state, self.encode_observations(state, action, rewards)

    def encode_observations(
        self, state: TravelingSalesmanState, actions, rewards
    ) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                (positions[0] - self.view_width // 2, positions[1] - self.view_height // 2),
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

    def create_logs(self, state: TravelingSalesmanState):
        return {"rewards": state.rewards}

    def get_render_state(self, state: TravelingSalesmanState) -> GridRenderState:
        tilemap = state.map

        tilemap = tilemap.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            GW.AGENT_GENERIC
        )

        return GridRenderState(
            tilemap=tilemap,
            pad_width=self.pad_width,
            pad_height=self.pad_height,
            unpadded_width=self.width,
            unpadded_height=self.height,
            agent_positions=state.agents_pos,
            agent_types=None,
            view_width=self.view_width,
            view_height=self.view_height,
        )
