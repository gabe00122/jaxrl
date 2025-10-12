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
    agent_positions: jax.Array  # (N, 2)


class GridRenderSettings(NamedTuple):
    tile_width: int
    tile_height: int
    view_width: int
    view_height: int


tilemap = {
    GW.TILE_EMPTY: (17, 0),
    GW.TILE_WALL: (20, 3),
    GW.TILE_DESTRUCTIBLE_WALL: (20, 3),
    GW.TILE_FLAG: [(29, 23), (29, 31), (29, 32)],
    GW.TILE_FLAG_UNLOCKED: (29, 23),
    GW.AGENT_GENERIC: (104, 0),
    GW.AGENT_HARVESTER: (13, 14),
    GW.AGENT_SCOUT: (3, 16),
    GW.AGENT_KNIGHT: [
        None,
        [[(35, 31), (31, 31)], [(36, 31), (32, 31)], [(37, 31), (33, 31)], [(38, 31), (34, 31)]], # red
        [[(35, 32), (31, 32)], [(36, 32), (32, 32)], [(37, 32), (33, 32)], [(38, 32), (34, 32)]]  # blue
    ],
    GW.AGENT_ARCHER: [
        None,
        [(39, 31), (40, 31), (41, 31), (42, 31)], # red
        [(39, 32), (40, 32), (41, 32), (42, 32)]  # blue
    ],
    GW.TILE_DECOR_1: (15, 5),
    GW.TILE_DECOR_2: (16, 5),
    GW.TILE_DECOR_3: (17, 5),
    GW.TILE_DECOR_4: (14, 5),
    GW.TILE_ARROW: (80, 21)
}


class GridworldRenderer:
    def __init__(
        self, screen_width: int = 960, screen_height: int = 960, fps: int = 10
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self._vision_flags = pygame.SRCALPHA
        self.vision = pygame.Surface(
            (self.screen_width, self.screen_height), flags=self._vision_flags
        )
        self.clock = pygame.time.Clock()

        self.frames: list[np.ndarray] = []
        self._tile_size: int | None = None
        self._tilecache = None
        self._agent_tile_size: int | None = None
        self._agent_tilecache = None
        self._view_offset_x = 0
        self._view_offset_y = 0
        self._agent_view_offset_x = 0
        self._agent_view_offset_y = 0
        self._focused_agent = None

        self._spritesheet = SpriteSheet("./assets/urizen_onebit_tileset__v2d0.png")

    def set_env(self, env_settings: GridRenderSettings):
        self._tile_width = env_settings.tile_width
        self._tile_height = env_settings.tile_height
        self._view_width = env_settings.view_width
        self._view_height = env_settings.view_height

        self._pad_width = self._view_width // 2
        self._pad_height = self._view_height // 2

    def _refresh_screen_surface(self):
        current_surface = pygame.display.get_surface()
        self.screen = current_surface

        width, height = current_surface.get_size()
        if width != self.screen_width or height != self.screen_height:
            self.screen_width = width
            self.screen_height = height
            self.vision = pygame.Surface(
                (self.screen_width, self.screen_height), flags=self._vision_flags
            )
            self._tile_size = None

    def _ensure_layout(self):
        self._refresh_screen_surface()

        tile_size = max(
            1,
            min(self.screen_width // self._tile_width, self.screen_height // self._tile_height),
        )

        agent_tile_size = max(
            1,
            min(self.screen_width // self._view_width, self.screen_height // self._view_height),
        )

        if self._tile_size != tile_size or self._tilecache is None:
            self._tile_size = tile_size
            self._tilecache = self._build_tilecache(self._tile_size)

        if self._agent_tile_size != agent_tile_size or self._agent_tilecache is None:
            self._agent_tile_size = agent_tile_size
            self._agent_tilecache = self._build_tilecache(self._agent_tile_size)

        view_pixel_width = self._tile_size * self._tile_width
        view_pixel_height = self._tile_size * self._tile_height
        self._view_offset_x = (self.screen_width - view_pixel_width) // 2
        self._view_offset_y = (self.screen_height - view_pixel_height) // 2

        agent_pixel_width = self._agent_tile_size * self._view_width
        agent_pixel_height = self._agent_tile_size * self._view_height
        self._agent_view_offset_x = (self.screen_width - agent_pixel_width) // 2
        self._agent_view_offset_y = (self.screen_height - agent_pixel_height) // 2

    def _build_tilecache(self, tile_size: int):
        def _load_tiles(item: dict | list | tuple[int, int]):
            if isinstance(item, tuple):
                return self._spritesheet.image_at_tile(item[0], item[1], tile_size)
            if isinstance(item, list):
                return [_load_tiles(elm) for elm in item]
            if isinstance(item, dict):
                return {name: _load_tiles(elm) for name, elm in item.items()}

        return _load_tiles(tilemap)

    def _tile_to_screen(self, x: int, y: int):
        return x - self._pad_width, (self._tile_height - 1 - y) + self._pad_height

    def _draw_tile(self, image, x, y):
        x, y = self._tile_to_screen(x, y)
        px = self._view_offset_x + x * self._tile_size
        py = self._view_offset_y + y * self._tile_size
        dest = pygame.Rect(px, py, self._tile_size, self._tile_size)
        self.screen.blit(image, dest)

    def _draw_vision(
        self,
        color,
        x,
        y,
    ):
        sx, sy = self._tile_to_screen(x, y)

        half_w = self._view_width // 2
        half_h = self._view_height // 2

        px = (sx - half_w) * self._tile_size
        py = (sy - half_h) * self._tile_size
        pw = self._view_width * self._tile_size
        ph = self._view_height * self._tile_size

        rect = pygame.Rect(
            int(self._view_offset_x + px),
            int(self._view_offset_y + py),
            int(pw),
            int(ph),
        )

        clipped = rect.clip(self.vision.get_rect())
        if clipped.width > 0 and clipped.height > 0:
            self.vision.fill(color, clipped)

    def focus_agent(self, agent_id: int | None):
        self._focused_agent = agent_id

    def render(self, rs: GridRenderState):
        self._ensure_layout()

        self.screen.fill((0, 0, 0))
        self.vision.fill(pygame.color.Color(70, 70, 70, 100))

        tiles = rs.tilemap.tolist()

        for x in range(self._tile_width):
            for y in range(self._tile_height):
                tx = self._pad_width + x
                ty = self._pad_height + y
                tile = tiles[tx][ty]
                tile_type_id = tile[0]

                image = self._tilecache[tile_type_id]
                if isinstance(image, list):
                    image = image[tile[2]]
                if isinstance(image, list):
                    image = image[tile[1] - 1]
                if isinstance(image, list):
                    image = image[tile[3] - 1]

                self._draw_tile(image, tx, ty)

        agent_pos = rs.agent_positions.tolist()
        for i, (x, y) in enumerate(agent_pos):
            if self._focused_agent is None or i == self._focused_agent:
                self._draw_vision((0, 0, 0, 0), x, y)

        self.clock.tick(self.fps)
        self.screen.blit(self.vision, (0, 0))
        pygame.display.flip()

    def render_agent_view(self, rs: GridRenderState):
        self._ensure_layout()

        tiles = rs.tilemap.tolist()
        agent_positions = rs.agent_positions.tolist()

        if not agent_positions:
            return

        agent_x, agent_y = agent_positions[self._focused_agent or 0]

        start_x = agent_x - self._pad_width
        start_y = agent_y - self._pad_height

        tilecache = self._agent_tilecache or {}
        tile_size = self._agent_tile_size or 1

        self.screen.fill((0, 0, 0))

        for vx in range(self._view_width):
            tx = start_x + vx
            for vy in range(self._view_height):
                ty = start_y + vy
                tile_data = tiles[tx][ty]
                tile_type_id = tile_data[0]

                image = tilecache.get(tile_type_id)
                if isinstance(image, list):
                    image = image[tile_data[2]]

                if image is None:
                    continue

                px = self._agent_view_offset_x + vx * tile_size
                py = self._agent_view_offset_y + (self._view_height - 1 - vy) * tile_size
                dest = pygame.Rect(px, py, tile_size, tile_size)
                self.screen.blit(image, dest)

        self.clock.tick(self.fps)
        pygame.display.flip()

    def handle_event(self, event: pygame.event.Event) -> bool:
        return False

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
        assert hasattr(env, "get_render_settings"), (
            "Env must implement get_render_settings()"
        )
        self.env = env
        self.renderer = GridworldRenderer(screen_width=screen_width, screen_height=screen_height, fps=fps)
        self.renderer.set_env(env.get_render_settings())

    def render(self, state, timestep):
        rs: GridRenderState = self.env.get_render_state(state)
        self.renderer.render(rs)

    def render_pov(self, state, timestep):
        """Render only the focused agent's point-of-view, filling the screen."""
        rs: GridRenderState = self.env.get_render_state(state)
        self.renderer.render_agent_view(rs)

    def handle_event(self, event: pygame.event.Event) -> bool:
        return self.renderer.handle_event(event)

    def record_frame(self):
        self.renderer.record_frame()

    def save_video(self, file_name: str):
        self.renderer.save_video(file_name)
    
    def focus_agent(self, agent_id: int | None):
        self.renderer.focus_agent(agent_id)
