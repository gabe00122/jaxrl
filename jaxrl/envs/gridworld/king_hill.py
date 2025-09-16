from functools import cached_property, partial
from typing import NamedTuple, Literal, override

import jax
from jax import numpy as jnp
from numpy import where
from pydantic import BaseModel, ConfigDict
from jaxrl.envs.environment import Environment
from jaxrl.envs.map_generator import choose_positions_in_rect, generate_decor_tiles, generate_perlin_noise_2d
from jaxrl.envs.specs import DiscreteActionSpec, ObservationSpec
from jaxrl.types import TimeStep
from jaxrl.envs.gridworld.renderer import GridRenderState
import jaxrl.envs.gridworld.constance as GW


class KingHillConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["king_hill"] = "king_hill"

    num_agents: int = 8
    num_flags: int = 2

    width: int = 40
    height: int = 40
    view_width: int = 5
    view_height: int = 5

    dig_timeout: int = 10
    reward_per_turn: float = 10.0 / 512


class KingHillState(NamedTuple):
    agents_start_pos: jax.Array
    agents_pos: jax.Array # (n, 2)
    agents_direction: jax.Array # (n,) top, right, down, left
    agents_timeouts: jax.Array
    agents_types: jax.Array

    arrows_pos: jax.Array
    arrows_direction: jax.Array
    arrows_timeouts: jax.Array
    arrows_mask: jax.Array

    control_point_pos: jax.Array # (n, 2)
    control_point_team: jax.Array # (n,) neutral, red, blue

    time: jax.Array
    tiles: jax.Array
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

        self._teams = self._repeat_for_team(jnp.int32(0), jnp.int32(1))

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

    def _generate_tiles(self, rng_key):
        decor_key, wall_key = jax.random.split(rng_key)
        tiles = generate_decor_tiles(self.width, self.height, decor_key)

        noise = generate_perlin_noise_2d((self.width, self.height), (10, 10), rng_key=wall_key) > 0.25
        noise = noise.at[:, 0].set(False) # clear the starting edges so agents are not stuck in the walls
        noise = noise.at[:, self.height-1].set(False)

        tiles = jnp.where(noise, GW.TILE_DESTRUCTIBLE_WALL, tiles)

        tiles = self._pad_tiles(tiles, GW.TILE_WALL)

        return tiles

    def reset(self, rng_key: jax.Array) -> tuple[KingHillState, TimeStep]:
        tiles_key, pos_key = jax.random.split(rng_key)

        tiles = self._generate_tiles(tiles_key)

        # place objective
        control_point_x, control_point_y = choose_positions_in_rect(tiles, self._config.num_flags, pos_key, 0, self.padded_height // 2 - 5, self.padded_width, 11) #jnp.array([[self.padded_width // 2, self.padded_height // 2]], jnp.int32)
        control_point_pos = jnp.stack((control_point_x, control_point_y), axis=-1)
        tiles = tiles.at[control_point_x, control_point_y].set(GW.TILE_FLAG)
        

        xs = jnp.arange(self.num_agents // 2, dtype=jnp.int32) + (self.width // 2 - self.num_agents // 4) + self.pad_width
        ys = jnp.zeros((self.num_agents // 2,), dtype=jnp.int32) + self.pad_height
        red_agent_pos = jnp.stack((xs, ys), axis=-1)
        blue_agent_pos = jnp.stack((xs, ys + self.height - 1), axis=-1)
        agent_pos = jnp.concatenate((red_agent_pos, blue_agent_pos), axis=0)

        state = KingHillState(
            agents_start_pos=agent_pos,
            agents_pos=agent_pos,
            agents_direction=jnp.zeros((self.num_agents,), jnp.int32),
            agents_timeouts=jnp.zeros((self.num_agents,), jnp.int32),
            agents_types=jnp.zeros((self._num_agents), jnp.int32),
            arrows_pos=jnp.zeros((self._num_agents, 2), jnp.int32),
            arrows_direction=jnp.zeros((self._num_agents,), jnp.int32),
            arrows_timeouts=jnp.zeros((self._num_agents,), jnp.int32),
            arrows_mask=jnp.zeros((self._num_agents,), jnp.bool_),
            control_point_pos=control_point_pos,
            control_point_team=jnp.zeros((self._config.num_flags,), jnp.int32), # team index that controls the point
            tiles=tiles,
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

    
    def _calculate_movement(self, state: KingHillState, action: jax.Array, rng_key: jax.Array):
        move_order = jax.random.permutation(rng_key, self._num_agents)

        idx = jnp.arange(self._num_agents, dtype=jnp.int32)
        movement_markers = jnp.full_like(state.tiles, -1, jnp.int32)

        proposed_position = jnp.where((action < 4)[:, None], state.agents_pos + GW.DIRECTIONS[action], state.agents_pos)
        target_tile = state.tiles[proposed_position[:, 0], proposed_position[:, 1]]
        not_blocked_by_tile = jnp.logical_and(target_tile != GW.TILE_WALL, target_tile != GW.TILE_DESTRUCTIBLE_WALL)

        ordered_proposed_position = proposed_position[move_order]
        movement_markers = movement_markers.at[ordered_proposed_position[:, 0], ordered_proposed_position[:, 1]].set(move_order)
        movement_markers = movement_markers.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(idx)

        target_move = movement_markers[proposed_position[:, 0], proposed_position[:, 1]]
        proposed_position = jnp.where(
            jnp.logical_and(jnp.logical_or(target_move == idx, target_move == -1), not_blocked_by_tile)[:, None],
            proposed_position,
            state.agents_pos
        )

        return proposed_position

    def _calculate_flag_captures(self, state: KingHillState):
        red_team = state.agents_pos[:self.num_agents//2]
        blue_team = state.agents_pos[self.num_agents//2:]

        # this might be more efficent as a id map with the flag id's, i'm not sure. Right now the compute is agents * flags
        def get_overlaps(flags: jax.Array, team: jax.Array):
            eq = (flags[:, None, :] == team[None, :, :])

            # All coords must match -> (N, M)
            matches = jnp.all(eq, axis=-1)

            # Any match for each a -> (N,)
            return jnp.any(matches, axis=1)

        red_overlaps = get_overlaps(state.control_point_pos, red_team)
        blue_overlaps = get_overlaps(state.control_point_pos, blue_team)


        flag_control = jnp.where(red_overlaps, 1, state.control_point_team)
        flag_control = jnp.where(blue_overlaps, 2, flag_control)

        return flag_control
    
    def _calculate_directions(self, state: KingHillState, action: jax.Array) -> jax.Array:
        return jnp.where(action < 4, action, state.agents_direction)
    
    def _indices_to_mask(self, indices: jax.Array, size: int) -> jax.Array:
        one_hot = jax.nn.one_hot(indices, size, dtype=jnp.bool_)
        mask = jnp.any(one_hot, axis=0)
        return mask

    
    def _calculate_attacks(self, state: KingHillState, action: jax.Array, agent_targets: jax.Array):
        damage_map = jnp.zeros_like(state.tiles)

        target_pos = state.agents_pos + GW.DIRECTIONS[state.agents_direction]

        attack_mask = action == GW.PRIMARY_ACTION

        damage_map = damage_map.at[target_pos[:, 0], target_pos[:, 1]].add(
            jnp.where(attack_mask, 1, 0)
        )

        killed_mask = damage_map[state.agents_pos[:, 0], state.agents_pos[:, 1]] > 0

        # respawn
        state = state._replace(
            agents_pos=jnp.where(killed_mask[:, None], state.agents_start_pos, state.agents_pos)
        )

        # temp
        state.arrows_pos
        state = state._replace(
            arrows_mask=jnp.logical_or(state.arrows_mask, attack_mask),
            arrows_pos=jnp.where(attack_mask[:, None], agent_targets, state.arrows_pos),
            arrows_direction=jnp.where(attack_mask, state.agents_direction, state.arrows_direction),
        )
        # temp

        return state
    
    def _calculate_arrows(self, state: KingHillState) -> KingHillState:
        target_pos = state.arrows_pos + GW.DIRECTIONS[state.arrows_direction]

        return state._replace(
            arrows_pos = target_pos
        )


    def _repeat_for_team(self, red_item, blue_item):
        team_size = self.num_agents // 2

        red_reward = jnp.repeat(red_item[None], team_size)
        blue_reward = jnp.repeat(blue_item[None], team_size)

        return jnp.concatenate((red_reward, blue_reward))
    
    def _calculate_digs(self, state: KingHillState, action: jax.Array, agent_targets: jax.Array) -> KingHillState:
        target_tile = state.tiles[agent_targets[:, 0], agent_targets[:, 1]]
        execute_dig = jnp.logical_and(
            state.tiles[agent_targets[:, 0], agent_targets[:, 1]] == GW.TILE_DESTRUCTIBLE_WALL,
            action == GW.DIG_ACTION,
        )

        tiles = state.tiles.at[agent_targets[:, 0], agent_targets[:, 1]].set(jnp.where(execute_dig, GW.TILE_EMPTY, target_tile))

        state = state._replace(
            tiles=tiles,
            agents_timeouts=jnp.where(execute_dig, self._config.dig_timeout, state.agents_timeouts)
        )
        return state

    def step(
        self, state: KingHillState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[KingHillState, TimeStep]:
        action = jnp.where(state.agents_timeouts > 0, GW.STAY, action) # this does change the next observation, should probably be done with a action mask

        agent_targets = state.agents_pos + GW.DIRECTIONS[state.agents_direction]

        state = self._calculate_digs(state, action, agent_targets)
        state = self._calculate_attacks(state, action, agent_targets)

        state = self._calculate_arrows(state)

        new_position = self._calculate_movement(state, action, rng_key)
        new_directions = self._calculate_directions(state, action)

        # flag control
        state = state._replace(agents_pos=new_position, agents_direction=new_directions)
        flag_control = self._calculate_flag_captures(state)

        rewards = self._config.reward_per_turn * self._repeat_for_team(
            jnp.sum(flag_control == 1),
            jnp.sum(flag_control == 2),
        )

        state = state._replace(
            control_point_team=flag_control,
            time=state.time + 1,
            agents_timeouts=jnp.maximum(0, state.agents_timeouts - 1),
            rewards=state.rewards + jnp.mean(rewards),
        )

        return state, self.encode_observations(state, action, rewards)
    
    def _render_tiles(self, state: KingHillState):
        tiles = state.tiles
        tiles = tiles.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            self._repeat_for_team(jnp.array(GW.AGENT_RED_KNIGHT_RIGHT, jnp.int8), jnp.array(GW.AGENT_BLUE_KNIGHT_RIGHT, jnp.int8))
        )

        flag_tiles = jnp.array([GW.TILE_FLAG, GW.TILE_FLAG_RED_TEAM, GW.TILE_FLAG_BLUE_TEAM], jnp.int8)
        tiles = tiles.at[state.control_point_pos[:, 0], state.control_point_pos[:, 1]].set(
            flag_tiles[state.control_point_team]
        )

        tiles = tiles.at[state.arrows_pos[:, 0], state.arrows_pos[:, 1]].set(
            jnp.where(state.arrows_mask, jnp.int8(GW.TILE_ARROW), tiles[state.arrows_pos[:, 0], state.arrows_pos[:, 1]])
        )

        return tiles

    def encode_observations(
        self, state: KingHillState, actions, rewards
    ) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                (positions[0] - self.view_width // 2, positions[1] - self.view_height // 2),
                (self.view_width, self.view_height),
            )

        tiles = self._render_tiles(state)
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
        tiles = self._render_tiles(state)

        return GridRenderState(
            tilemap=tiles,
            pad_width=self.pad_width,
            pad_height=self.pad_height,
            unpadded_width=self.width,
            unpadded_height=self.height,
            agent_positions=state.agents_pos,
            agent_types=jnp.array(([GW.AGENT_RED_KNIGHT_RIGHT] * (self.num_agents//2)) + ([GW.AGENT_BLUE_KNIGHT_RIGHT] * (self.num_agents//2))),
            view_width=self.view_width,
            view_height=self.view_height,
        )

    @override
    @property
    def teams(self) -> jax.Array:
        return self._teams
