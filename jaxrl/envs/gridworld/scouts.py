from functools import cached_property, partial
from typing import NamedTuple, Literal

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict

from jaxrl.envs.map_generator import generate_perlin_noise_2d
from jaxrl.envs.environment import Environment
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.envs.gridworld.renderer import GridRenderState
import jaxrl.envs.gridworld.constance as GW


class ScoutsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["scouts"] = "scouts"

    num_scouts: int = 1
    num_harvesters: int = 1
    num_treasures: int = 1

    width: int = 40
    height: int = 40
    view_width: int = 5
    view_height: int = 5

    harvesters_move_every: int = 6
    scout_reward: float = 1.0
    harvester_reward: float = 1.0


class ScoutsState(NamedTuple):
    scout_pos: jax.Array  # n length (x, y)
    harvester_pos: jax.Array
    harvester_time: jax.Array  # n length ()

    time: jax.Array  # ()

    map: jax.Array  # (w, h) tile type id
    spawn_pos: jax.Array  # n length (x, y) spawnable positions, padded with -1
    spawn_count: jax.Array  # () size of spawn_pos

    rewards: jax.Array


class ScoutsEnv(Environment[ScoutsState]):
    def __init__(self, config: ScoutsConfig, length: int) -> None:
        super().__init__()

        self._length = length

        self._num_scouts = config.num_scouts
        self._num_harvesters = config.num_harvesters
        self._num_treasures = config.num_treasures

        self.unpadded_width = config.width
        self.unpadded_height = config.height

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.width = self.unpadded_width + self.pad_width
        self.height = self.unpadded_height + self.pad_height

        self.harvesters_move_every = config.harvesters_move_every
        self.scout_reward = config.scout_reward
        self.harvester_reward = config.harvester_reward

    def _generate_map(self, rng_key):
        res = [4, 5, 8, 10]

        noise_key, amplitude_key, rng_key = jax.random.split(rng_key, 3)

        amplitude = jax.random.dirichlet(amplitude_key, jnp.ones((5,)))
        noise = (
            generate_perlin_noise_2d(
                (self.unpadded_width, self.unpadded_height), (2, 2), rng_key=noise_key
            )
            * amplitude[0]
        )

        for i, r in enumerate(res):
            noise_key, rng_key = jax.random.split(rng_key)
            noise = (
                noise
                + generate_perlin_noise_2d(
                    (self.unpadded_width, self.unpadded_height),
                    (r, r),
                    rng_key=noise_key,
                )
                * amplitude[i + 1]
            )

        tiles = jnp.where(noise > 0.05, jnp.int8(GW.TILE_WALL), jnp.int8(GW.TILE_EMPTY))

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

    def reset(self, rng_key: jax.Array) -> tuple[ScoutsState, TimeStep]:
        map_key, scout_key, harvester_key, treasure_key = jax.random.split(rng_key, 4)

        map, spawn_pos, spawn_count = self._generate_map(map_key)

        scout_pos = spawn_pos[
            jax.random.randint(
                scout_key, (self._num_scouts,), minval=0, maxval=spawn_count
            )
        ]
        # harvester_pos = spawn_pos[jax.random.randint(
        #     harvester_key, (self._num_harvesters,), minval=0, maxval=spawn_count
        # )]
        harvester_pos = scout_pos
        treasure_pos = spawn_pos[
            jax.random.randint(
                treasure_key, (self._num_treasures,), minval=0, maxval=spawn_count
            )
        ]

        map = map.at[treasure_pos[:, 0], treasure_pos[:, 1]].set(GW.TILE_FLAG)

        state = ScoutsState(
            map=map,
            spawn_pos=spawn_pos,
            spawn_count=spawn_count,
            scout_pos=scout_pos,
            harvester_pos=harvester_pos,
            harvester_time=jnp.zeros((self._num_harvesters,), dtype=jnp.int32),
            time=jnp.int32(50),
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
        return self._num_scouts + self._num_harvesters

    def step(
        self, state: ScoutsState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[ScoutsState, TimeStep]:
        seeker_actions = action[: self._num_scouts]
        harvester_actions = action[self._num_scouts :]

        @partial(jax.vmap, in_axes=(0, 0), out_axes=(0, 0))
        def _step_scouter(local_position, local_action):
            new_pos = local_position + GW.DIRECTIONS[local_action]

            new_tile = state.map[new_pos[0], new_pos[1]]

            # don't move if we are moving into a wall
            new_pos = jnp.where(new_tile == GW.TILE_WALL, local_position, new_pos)

            reward = jnp.where(new_tile == GW.TILE_FLAG_UNLOCKED, self.scout_reward, 0.0)

            return new_pos, reward

        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0, 0, 0))
        def _step_harvester(local_position, local_action, time):
            def step_time(local_position, local_action, time):
                return local_position, 0.0, time - 1

            def step_move(local_position, local_action, time):
                new_pos = local_position + GW.DIRECTIONS[local_action]

                new_tile = state.map[new_pos[0], new_pos[1]]

                # don't move if we are moving into a wall
                new_pos = jnp.where(new_tile == GW.TILE_WALL, local_position, new_pos)

                reward = jnp.where(new_tile == GW.TILE_FLAG, self.harvester_reward, 0.0)
                time = (
                    self.harvesters_move_every
                )  # (new_tile == TILE_TREASURE).astype(jnp.int32) * 20

                return new_pos, reward, time

            return jax.lax.cond(
                time > 0, step_time, step_move, local_position, local_action, time
            )

        map = state.map
        new_harvester_positions, harvester_rewards, harvester_time = _step_harvester(
            state.harvester_pos, harvester_actions, state.harvester_time
        )

        # update unopened treasure to opened treasure
        new_harvester_tile = map[
            new_harvester_positions[:, 0], new_harvester_positions[:, 1]
        ]
        map = map.at[new_harvester_positions[:, 0], new_harvester_positions[:, 1]].set(
            jnp.where(
                new_harvester_tile == GW.TILE_FLAG,
                GW.TILE_FLAG_UNLOCKED,
                new_harvester_tile,
            )
        )

        # update scounters
        new_scout_positions, scout_rewards = _step_scouter(
            state.scout_pos, seeker_actions
        )
        new_scout_tile = map[new_scout_positions[:, 0], new_scout_positions[:, 1]]
        map = map.at[new_scout_positions[:, 0], new_scout_positions[:, 1]].set(
            jnp.where(
                new_scout_tile == GW.TILE_FLAG_UNLOCKED, GW.TILE_EMPTY, new_scout_tile
            )
        )

        # for each treasure that was found create a new one
        # random_positions = state.spawn_pos[jax.random.randint(
        #     rng_key, (self._num_scouts,), minval=0, maxval=state.spawn_count
        # )]
        # # there is a change two of these positions are the same and be lose a treasure
        # map = map.at[random_positions[:, 0], random_positions[:, 1]].set(jnp.where(
        #     new_scout_tile == TILE_TREASURE_OPEN,
        #     TILE_TREASURE,
        #     map[random_positions[:, 0], random_positions[:, 1]],
        # ))

        rewards = jnp.concatenate((scout_rewards, harvester_rewards))

        state = state._replace(
            map=map,
            scout_pos=new_scout_positions,
            harvester_pos=new_harvester_positions,
            harvester_time=harvester_time,
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
        )

        return state, self.encode_observations(state, action, rewards)

    def encode_observations(self, state: ScoutsState, actions, rewards) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                positions - jnp.array([self.view_width // 2, self.view_height // 2]),
                (self.view_width, self.view_height),
            )

        map = state.map.at[state.scout_pos[:, 0], state.scout_pos[:, 1]].set(
            GW.AGENT_SCOUT
        )
        map = map.at[state.harvester_pos[:, 0], state.harvester_pos[:, 1]].set(
            GW.AGENT_HARVESTER
        )

        agents_pos = jnp.concatenate((state.scout_pos, state.harvester_pos), axis=0)

        view = _encode_view(map, agents_pos)

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

        return TimeStep(
            obs=view,
            time=time,
            last_action=actions,
            last_reward=rewards,
            action_mask=None,
            terminated=jnp.equal(time, self._length - 1),
        )

    # Shared renderer adapter
    def get_render_state(self, state: ScoutsState) -> GridRenderState:
        agent_positions = jnp.concatenate(
            (state.scout_pos, state.harvester_pos), axis=0
        )

        return GridRenderState(
            tilemap=state.map,
            pad_width=self.pad_width,
            pad_height=self.pad_height,
            unpadded_width=self.unpadded_width,
            unpadded_height=self.unpadded_height,
            agent_positions=agent_positions,
            view_width=self.view_width,
            view_height=self.view_height,
        )

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: ScoutsState):
        return {"rewards": state.rewards}
