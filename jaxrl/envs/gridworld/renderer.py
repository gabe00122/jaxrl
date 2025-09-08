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
    GW.TILE_WALL: (20, 3),
    GW.TILE_DESTRUCTIBLE_WALL: (20, 3),
    GW.TILE_FLAG: (29, 23),
    GW.TILE_FLAG_UNLOCKED: (29, 23),
    GW.TILE_FLAG_RED_TEAM: (29, 31),
    GW.TILE_FLAG_BLUE_TEAM: (29, 32),
    GW.AGENT_GENERIC: (104, 0),
    GW.AGENT_HARVESTER: (104, 0),
    GW.AGENT_RED_KNIGHT_RIGHT: (153, 5),
    GW.AGENT_BLUE_KNIGHT_RIGHT: (153, 11),
    GW.AGENT_SCOUT: (104, 0),
    GW.TILE_DECOR_1: (15, 5),
    GW.TILE_DECOR_2: (16, 5),
    GW.TILE_DECOR_3: (17, 5),
    GW.TILE_DECOR_4: (14, 5)
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

    def render_agent_view(self, rs: GridRenderState, agent_id: int = 0):
        """Renders only the focused agent's field-of-view, scaled to fill screen.

        - Selects an `agent_id` and crops a (view_width x view_height) window around it.
        - Scales the cropped region to fill the entire screen (may be non-square tiles).
        """
        # Ensure base sprites are loaded (for access to tile images)
        self._ensure_tile_size(max(1, rs.unpadded_width))

        vw = max(1, int(rs.view_width))
        vh = max(1, int(rs.view_height))

        # Determine agent position
        agent_pos = rs.agent_positions.tolist()
        if not (0 <= agent_id < len(agent_pos)):
            agent_id = 0
        ax, ay = agent_pos[agent_id]

        # Compute top-left of view window in world coords
        x0 = ax - vw // 2
        y0 = ay - vh // 2

        # Compute per-tile pixel size (allow non-square to fill screen exactly)
        tw_f = self.screen_width / float(vw)
        th_f = self.screen_height / float(vh)

        tiles = rs.tilemap.tolist()
        total_height = rs.unpadded_height + 2 * rs.pad_height

        # Clear background
        self.screen.fill((0, 0, 0))

        # Draw the cropped tile window
        for dx in range(vw):
            for dy in range(vh):
                tx = x0 + dx
                ty = y0 + dy
                # Guard against out-of-bounds just in case
                if not (0 <= tx < len(tiles) and 0 <= ty < len(tiles[0])):
                    continue
                tile_type = tiles[tx][ty]
                base_img = self._tilemap[tile_type]

                # Compute screen placement; flip y to place origin at bottom-left
                px = int(round(dx * tw_f))
                py = int(round((vh - 1 - dy) * th_f))
                pw = int(round((dx + 1) * tw_f)) - px
                ph = int(round((vh - 0 - dy) * th_f)) - py
                if pw <= 0 or ph <= 0:
                    continue

                scaled = pygame.transform.scale(base_img, (pw, ph))
                self.screen.blit(scaled, (px, py))

        # Draw any agents that fall within the window
        agent_types = (
            rs.agent_types.tolist()
            if rs.agent_types is not None
            else [GW.AGENT_GENERIC] * len(agent_pos)
        )
        for (x, y), t in zip(agent_pos, agent_types):
            if x0 <= x < x0 + vw and y0 <= y < y0 + vh:
                dx = x - x0
                dy = y - y0
                px = int(round(dx * tw_f))
                py = int(round((vh - 1 - dy) * th_f))
                pw = int(round((dx + 1) * tw_f)) - px
                ph = int(round((vh - 0 - dy) * th_f)) - py
                if pw <= 0 or ph <= 0:
                    continue
                agent_img = self._tilemap[t]
                scaled = pygame.transform.scale(agent_img, (pw, ph))
                self.screen.blit(scaled, (px, py))

        self.clock.tick(self.fps)
        pygame.display.flip()

    def record_frame(self):
        img_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.frames.append(img_data)

    def save_video(self, file_name: str):
        if len(self.frames) == 0:
            return
        frames = np.array(self.frames)
        save_video(frames, file_name, self.fps)


class GridworldClient:
    """EnvironmentClient that renders via GridworldRenderer using per-env adapters."""

    def __init__(self, env, screen_width: int = 960, screen_height: int = 960, fps: int = 10):
        assert hasattr(env, "get_render_state"), (
            "Env must implement get_render_state(state)"
        )
        self.env = env
        self.renderer = GridworldRenderer(screen_width=screen_width, screen_height=screen_height, fps=fps)

    def render(self, state, timestep):
        rs: GridRenderState = self.env.get_render_state(state)
        self.renderer.render(rs)

    def render_pov(self, state, timestep, agent_id: int = 0):
        """Render only the focused agent's point-of-view, filling the screen."""
        rs: GridRenderState = self.env.get_render_state(state)
        self.renderer.render_agent_view(rs, agent_id=agent_id)

    def record_frame(self):
        self.renderer.record_frame()

    def save_video(self, file_name: str):
        self.renderer.save_video(file_name)
