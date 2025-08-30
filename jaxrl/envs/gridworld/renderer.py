from typing import NamedTuple

import numpy as np
import pygame
import jax
from jax import numpy as jnp


# Unified tile ids across gridworld environments
TILE_EMPTY = 0
TILE_WALL = 1
TILE_SOFT_WALL = 2
TILE_TREASURE = 3
TILE_TREASURE_OPEN = 4


# Agent type ids for rendering
AGENT_GENERIC = 0
AGENT_SCOUT = 1
AGENT_HARVESTER = 2


class GridRenderState(NamedTuple):
    tilemap: jax.Array  # padded tile map with unified tile ids
    pad_width: int
    pad_height: int
    unpadded_width: int
    unpadded_height: int

    agent_positions: jax.Array  # (N, 2)
    agent_types: jax.Array | None = None  # (N,)
    agent_colors: jax.Array | None = None  # (N,)

    view_width: int = 0
    view_height: int = 0


def _default_tile_color(tile_id: int) -> str:
    # Greys for empty, brown-ish walls, blue/orange for treasures
    palette = {
        TILE_EMPTY: "grey",
        TILE_WALL: "brown",
        TILE_SOFT_WALL: "sienna",
        TILE_TREASURE: "blue",
        TILE_TREASURE_OPEN: "orange",
    }
    return palette.get(tile_id, "grey")


def _agent_color(agent_type: int, color_id: int | None) -> str:
    # Default agent palette
    if color_id is not None:
        # inspired by return_2d_colors
        color_palette = [
            "darkorchid1",
            "darkorchid2",
            "darkorchid3",
            "darkorchid4",
        ]
        idx = int(color_id) % len(color_palette)
        return color_palette[idx]

    if agent_type == AGENT_SCOUT:
        return "yellow"
    if agent_type == AGENT_HARVESTER:
        return "purple"
    return "yellow"


class GridworldRenderer:
    def __init__(self, screen_width: int = 800, screen_height: int = 800, fps: int = 10):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps

        flags = pygame.SRCALPHA
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.surface = pygame.Surface((self.screen_width, self.screen_height), flags=flags)
        self.clock = pygame.time.Clock()

        self.frames: list[np.ndarray] = []
        self._tile_size: int | None = None

    def _ensure_tile_size(self, unpadded_width: int):
        # Compute tile size once per session based on the first env width
        if self._tile_size is None:
            self._tile_size = max(1, self.screen_width // max(1, int(unpadded_width)))

    def _tile_to_screen(self, x: int, y: int, pad_width: int, height: int, pad_height: int):
        return x - pad_width, (height - y + 1) - pad_height

    def _draw_tile(self, surface, color, x, y, width: int, height: int, pad_width: int, pad_height: int, total_height: int):
        x, y = self._tile_to_screen(x, y, pad_width, total_height, pad_height)
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

    def render(self, rs: GridRenderState):
        # Ensure tile size
        self._ensure_tile_size(rs.unpadded_width)

        # Clear translucent overlay
        self.surface.fill(pygame.color.Color(40, 40, 40, 100))

        # Draw base tiles (only unpadded region)
        tiles = rs.tilemap.tolist()
        total_height = rs.unpadded_height + rs.pad_height

        for x in range(rs.unpadded_width):
            for y in range(rs.unpadded_height):
                tx = rs.pad_width + x
                ty = rs.pad_height + y
                tile_type = tiles[tx][ty]
                color = _default_tile_color(tile_type)
                self._draw_tile(
                    self.screen,
                    color,
                    tx,
                    ty,
                    1,
                    1,
                    rs.pad_width,
                    rs.pad_height,
                    total_height,
                )

        # Draw agents
        agent_pos = rs.agent_positions.tolist()
        agent_types = rs.agent_types.tolist() if rs.agent_types is not None else [AGENT_GENERIC] * len(agent_pos)
        agent_colors = rs.agent_colors.tolist() if rs.agent_colors is not None else [None] * len(agent_pos)

        for (x, y), t, c in zip(agent_pos, agent_types, agent_colors):
            self._draw_tile(
                self.screen,
                _agent_color(t, c),
                x,
                y,
                1,
                1,
                rs.pad_width,
                rs.pad_height,
                total_height,
            )
            # Vision highlight
            vw = max(1, int(rs.view_width))
            vh = max(1, int(rs.view_height))
            self._draw_tile(
                self.surface,
                (0, 0, 0, 0),
                x,
                y,
                vw,
                vh,
                rs.pad_width,
                rs.pad_height,
                total_height,
            )

        self.clock.tick(self.fps)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def record_frame(self):
        img_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.frames.append(img_data)

    def save_video(self):
        # Lazy import to avoid dependency if unused
        from jaxrl.utils.video_writter import save_video

        if len(self.frames) == 0:
            return
        frames = np.array(self.frames)
        save_video(frames, "videos/test.mp4", self.fps)


class GridworldClient:
    """EnvironmentClient that renders via GridworldRenderer using per-env adapters."""
    def __init__(self, env):
        from jaxrl.envs.client import EnvironmentClient as _EC  # type: ignore
        assert hasattr(env, "get_render_state"), "Env must implement get_render_state(state)"
        self.env = env
        self.renderer = GridworldRenderer()

    def render(self, state, timestep):
        rs: GridRenderState = self.env.get_render_state(state)
        self.renderer.render(rs)

    def save_video(self):
        self.renderer.save_video()

