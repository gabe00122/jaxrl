from typing import NamedTuple

import numpy as np
import pygame
import jax

import jaxrl.envs.gridworld.constance as GW
from jaxrl.utils.video_writter import save_video


class SpriteSheet:
    def __init__(self, filename) -> None:
        self.sheet = pygame.image.load(filename).convert()
        self.tile_size = 12
        self.tile_pad = 1

    def image_at(self, rectangle, colorkey, target_size):
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size).convert()
        image.blit(self.sheet, (0, 0), rect)
        if colorkey is not None:
            if colorkey == -1:
                colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)

        image = pygame.transform.scale(image, (target_size, target_size))
        return image

    def image_at_tile(self, x: int, y: int, target_size: int):
        px = x * (self.tile_size + self.tile_pad) + self.tile_pad
        py = y * (self.tile_size + self.tile_pad) + self.tile_pad

        return self.image_at(
            (px, py, self.tile_size, self.tile_size), None, target_size
        )


class GridRenderState(NamedTuple):
    tilemap: jax.Array  # padded tile map with unified tile ids
    pad_width: int
    pad_height: int
    unpadded_width: int
    unpadded_height: int

    agent_positions: jax.Array  # (N, 2)
    agent_types: jax.Array | None = None  # (N,)

    view_width: int = 0
    view_height: int = 0


tilemap = {
    GW.TILE_EMPTY: (17, 0),
    GW.TILE_WALL: (0, 0),
    GW.TILE_SOFT_WALL: (0, 0),
    GW.TILE_TREASURE: (29, 23),
    GW.TILE_TREASURE_OPEN: (29, 23),
    GW.AGENT_GENERIC: (104, 0),
    GW.AGENT_HARVESTER: (104, 0),
    GW.AGENT_SCOUT: (104, 0),
}


class GridworldRenderer:
    def __init__(
        self, screen_width: int = 960, screen_height: int = 960, fps: int = 10
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps

        flags = pygame.SRCALPHA
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.vision = pygame.Surface(
            (self.screen_width, self.screen_height), flags=flags
        )
        self.clock = pygame.time.Clock()

        self.frames: list[np.ndarray] = []
        self._tile_size: int | None = None
        self._focused_agent = None

        self._spritesheet = SpriteSheet("./assets/urizen_onebit_tileset__v2d0.png")
        self._tilemap = None  # {name: self._spritesheet.image_at_tile(x, y) for name, (x, y) in tilemap.items()}

    def _ensure_tile_size(self, unpadded_width: int):
        # Compute tile size once per session based on the first env width
        if self._tile_size is None:
            self._tile_size = max(1, self.screen_width // max(1, int(unpadded_width)))
            self._tilemap = {
                name: self._spritesheet.image_at_tile(x, y, self._tile_size)
                for name, (x, y) in tilemap.items()
            }

    def _tile_to_screen(
        self, x: int, y: int, pad_width: int, height: int, pad_height: int
    ):
        return x - pad_width, (height - 1 - y) - pad_height

    def _draw_tile(
        self, image, x, y, pad_width: int, pad_height: int, total_height: int
    ):
        x, y = self._tile_to_screen(x, y, pad_width, total_height, pad_height)
        self.screen.blit(
            image,
            (
                x * self._tile_size,
                y * self._tile_size,
                self._tile_size,
                self._tile_size,
            ),
        )

    def _draw_vision(
        self,
        color,
        x,
        y,
        width: int,
        height: int,
        pad_width: int,
        pad_height: int,
        total_height: int,
    ):
        sx, sy = self._tile_to_screen(x, y, pad_width, total_height, pad_height)

        half_w = width // 2
        half_h = height // 2

        px = (sx - half_w) * self._tile_size
        py = (sy - half_h) * self._tile_size
        pw = width * self._tile_size
        ph = height * self._tile_size

        rect = pygame.Rect(int(px), int(py), int(pw), int(ph))

        clipped = rect.clip(self.vision.get_rect())
        if clipped.width > 0 and clipped.height > 0:
            self.vision.fill(color, clipped)

    def focus_agent(self, agent_id: int | None):
        self._focused_agent = agent_id

    def render(self, rs: GridRenderState):
        # Ensure tile size
        self._ensure_tile_size(rs.unpadded_width)

        # Clear translucent overlay
        self.vision.fill(pygame.color.Color(40, 40, 40, 100))

        # Draw base tiles (only unpadded region)
        tiles = rs.tilemap.tolist()
        total_height = rs.unpadded_height + 2 * rs.pad_height

        for x in range(rs.unpadded_width):
            for y in range(rs.unpadded_height):
                tx = rs.pad_width + x
                ty = rs.pad_height + y
                tile_type = tiles[tx][ty]
                image = self._tilemap[tile_type]
                self._draw_tile(
                    image, tx, ty, rs.pad_width, rs.pad_height, total_height
                )

        # Draw agents
        agent_pos = rs.agent_positions.tolist()
        agent_types = (
            rs.agent_types.tolist()
            if rs.agent_types is not None
            else [GW.AGENT_GENERIC] * len(agent_pos)
        )

        for i, ((x, y), t) in enumerate(zip(agent_pos, agent_types)):
            image = self._tilemap[t]
            self._draw_tile(image, x, y, rs.pad_width, rs.pad_height, total_height)

            # Vision highlight
            if self._focused_agent is None or i == self._focused_agent:
                vw = max(1, int(rs.view_width))
                vh = max(1, int(rs.view_height))
                self._draw_vision(
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
        self.screen.blit(self.vision, (0, 0))
        pygame.display.flip()

    def record_frame(self):
        img_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.frames.append(img_data)

    def save_video(self):
        if len(self.frames) == 0:
            return
        frames = np.array(self.frames)
        save_video(frames, "videos/test.mp4", self.fps)


class GridworldClient:
    """EnvironmentClient that renders via GridworldRenderer using per-env adapters."""

    def __init__(self, env):
        assert hasattr(env, "get_render_state"), (
            "Env must implement get_render_state(state)"
        )
        self.env = env
        self.renderer = GridworldRenderer()

    def render(self, state, timestep):
        rs: GridRenderState = self.env.get_render_state(state)
        self.renderer.render(rs)

    def save_video(self):
        self.renderer.save_video()
