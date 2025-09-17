from functools import cached_property, partial
from typing import NamedTuple, Literal, override

import jax
from jax import numpy as jnp
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

    dig_timeout: int = 5
    reward_per_turn: float = 10.0 / 512

    arrow_timeout: int = 5


class KingHillState(NamedTuple):
    agents_start_pos: jax.Array
    agents_pos: jax.Array # (n, 2)
    agents_direction: jax.Array # (n,) top, right, down, left
    agents_timeouts: jax.Array
    agents_types: jax.Array
    agents_health: jax.Array

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

        self._agent_type_health = jnp.array([2, 1], jnp.int32)


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
        tiles_key, pos_key, agent_type_key = jax.random.split(rng_key, 3)

        tiles = self._generate_tiles(tiles_key)

        # place objective
        control_point_x, control_point_y = choose_positions_in_rect(tiles, self._config.num_flags, pos_key, 0, self.padded_height // 2 - 5, self.padded_width, 11) #jnp.array([[self.padded_width // 2, self.padded_height // 2]], jnp.int32)
        control_point_pos = jnp.stack((control_point_x, control_point_y), axis=-1)
        tiles = tiles.at[control_point_x, control_point_y].set(GW.TILE_FLAG)
        
        team_size = self.num_agents // 2

        xs = jnp.arange(team_size, dtype=jnp.int32) + (self.width // 2 - self.num_agents // 4) + self.pad_width
        ys = jnp.zeros(team_size, dtype=jnp.int32) + self.pad_height
        red_agent_pos = jnp.stack((xs, ys), axis=-1)
        blue_agent_pos = jnp.stack((xs, ys + self.height - 1), axis=-1)
        agent_pos = jnp.concatenate((red_agent_pos, blue_agent_pos), axis=0)

        agent_types = jax.random.randint(agent_type_key, team_size, 0, 2, jnp.int32)
        agent_types = jnp.tile(agent_types, 2)

        state = KingHillState(
            agents_start_pos=agent_pos,
            agents_pos=agent_pos,
            agents_direction=jnp.zeros((self.num_agents,), jnp.int32),
            agents_timeouts=jnp.zeros((self.num_agents,), jnp.int32),
            agents_types=agent_types,
            agents_health=self._agent_type_health[agent_types], # archers 1, melee 2
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

    
    def _calculate_movement(self, state: KingHillState, action: jax.Array, rng_key: jax.Array):
        # warning, with more than 128 agents this will break because of int8
        assert self._num_agents < 128
        move_order = jax.random.permutation(rng_key, self._num_agents).astype(jnp.int8)

        idx = jnp.arange(self._num_agents, dtype=jnp.int8)
        movement_markers = jnp.full_like(state.tiles, -1, jnp.int8)

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

        melee_target_pos = state.agents_pos + GW.DIRECTIONS[state.agents_direction]
        arrow_target_pos = state.arrows_pos + GW.DIRECTIONS[state.arrows_direction]

        melee_attack_mask = jnp.logical_and(action == GW.PRIMARY_ACTION, state.agents_types == 0)

        target_pos = jnp.concatenate((melee_target_pos, arrow_target_pos, state.arrows_pos), axis=0)
        attack_mask = jnp.concatenate((melee_attack_mask, state.arrows_mask, state.arrows_mask), axis=0)

        damage_map = damage_map.at[target_pos[:, 0], target_pos[:, 1]].add(
            jnp.where(attack_mask, 1, 0)
        )

        agents_health = state.agents_health - damage_map[state.agents_pos[:, 0], state.agents_pos[:, 1]]
        killed_mask = agents_health <= 0

        # reset health on death
        agents_health = jnp.where(killed_mask, self._agent_type_health[state.agents_types], agents_health)

        # respawn
        state = state._replace(
            agents_pos=jnp.where(killed_mask[:, None], state.agents_start_pos, state.agents_pos),
            agents_health=agents_health,
            agents_timeouts=jnp.where(melee_attack_mask, 1, state.agents_timeouts)
        )

        # temp
        # todo create constants for the agent types, melee = 0, ranged = 1
        ranged_attack_mask = jnp.logical_and(action == GW.PRIMARY_ACTION, state.agents_types == 1)

        arrow_attack = jnp.logical_and(state.arrows_timeouts == 0, ranged_attack_mask)
        state = state._replace(
            arrows_mask=jnp.logical_or(state.arrows_mask, arrow_attack),
            arrows_pos=jnp.where(arrow_attack[:, None], state.agents_pos, state.arrows_pos),
            arrows_direction=jnp.where(arrow_attack, state.agents_direction, state.arrows_direction),
            arrows_timeouts=jnp.where(arrow_attack, self._config.arrow_timeout, state.arrows_timeouts)
        )
        # temp

        return state
    
    def _calculate_arrows(self, state: KingHillState) -> KingHillState:
        target_pos = state.arrows_pos + GW.DIRECTIONS[state.arrows_direction]

        target_tile = state.tiles[target_pos[:, 0], target_pos[:, 1]]
        arrows_timeouts = jnp.maximum(0, state.arrows_timeouts - 1)

        arrow_reset = target_tile == GW.TILE_WALL
        arrow_reset = jnp.logical_or(arrow_reset, target_tile == GW.TILE_DESTRUCTIBLE_WALL)
        arrow_reset = jnp.logical_or(arrow_reset, arrows_timeouts == 0)

        return state._replace(
            arrows_pos = target_pos,
            arrows_mask = jnp.logical_and(state.arrows_mask, ~arrow_reset),
            arrows_timeouts = arrows_timeouts,
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
    
    def _get_agent_type_tiles(self, state: KingHillState):
        agent_types_map = jnp.array([GW.AGENT_KNIGHT, GW.AGENT_ARCHER], jnp.int8)
        agent_types = agent_types_map[state.agents_types]

        return agent_types

    def _render_tiles(self, state: KingHillState):
        tiles = state.tiles

        flag_tiles = jnp.array([GW.TILE_FLAG, GW.TILE_FLAG_RED_TEAM, GW.TILE_FLAG_BLUE_TEAM], jnp.int8)
        tiles = tiles.at[state.control_point_pos[:, 0], state.control_point_pos[:, 1]].set(
            flag_tiles[state.control_point_team]
        )

        agent_types = self._get_agent_type_tiles(state)
        tiles = tiles.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(agent_types)

        tiles = tiles.at[state.arrows_pos[:, 0], state.arrows_pos[:, 1]].set(
            jnp.where(state.arrows_mask, jnp.int8(GW.TILE_ARROW), tiles[state.arrows_pos[:, 0], state.arrows_pos[:, 1]])
        )

        directions = jnp.zeros_like(tiles)
        directions = directions.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(state.agents_direction)
        directions = directions.at[state.arrows_pos[:, 0], state.arrows_pos[:, 1]].set(state.arrows_direction)

        teams = jnp.zeros_like(tiles)
        teams = teams.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(self.teams+1) # add one to account for none team
        # todo: add flag team

        health = jnp.zeros_like(tiles)
        health = health.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(state.agents_health)

        return jnp.concatenate(
            (tiles[..., None], directions[..., None], teams[..., None], health[..., None]),
            axis=-1,
        )

    def encode_observations(
        self, state: KingHillState, actions, rewards
    ) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                (positions[0] - self.view_width // 2, positions[1] - self.view_height // 2, 0),
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
            tilemap=tiles[..., 0],
            pad_width=self.pad_width,
            pad_height=self.pad_height,
            unpadded_width=self.width,
            unpadded_height=self.height,
            agent_positions=state.agents_pos,
            view_width=self.view_width,
            view_height=self.view_height,
        )

    @override
    @property
    def teams(self) -> jax.Array:
        return self._teams
