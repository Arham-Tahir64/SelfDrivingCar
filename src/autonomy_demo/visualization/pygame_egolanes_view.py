from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import LaneLine


class PygameEgoLanesViewer:
    """Standalone front-camera viewer that overlays perceived lane polylines."""

    def __init__(self, *, output_dir: Path | None = None, window_title: str = "autonomy_demo/egolanes_view") -> None:
        self.output_dir = output_dir
        self.window_title = window_title
        self.logger = get_logger(__name__, mode="egolanes_view")
        self._enabled = True
        self._pygame = None
        self._screen = None
        self._font = None
        self._title_font = None
        self._window_size: tuple[int, int] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def flush(self) -> None:
        pygame = self._pygame
        if pygame is not None:
            pygame.quit()

    def render(
        self,
        *,
        bundle,
        lanes: list[LaneLine],
        lane_source: str,
        status_lines: list[str],
    ) -> bool:
        if not self._enabled:
            return False
        camera = getattr(bundle, "front_camera", None)
        if camera is None:
            return True

        image = np.clip(np.asarray(camera.frame, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
        if not self._init_pygame(image.shape[1], image.shape[0]):
            self._enabled = False
            return False

        pygame = self._pygame
        screen = self._screen
        font = self._font
        title_font = self._title_font
        if pygame is None or screen is None or font is None or title_font is None:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._enabled = False
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._enabled = False
                pygame.quit()
                return False

        surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        if self._window_size != (image.shape[1], image.shape[0]):
            surface = pygame.transform.smoothscale(surface, self._window_size)
        screen.blit(surface, (0, 0))

        scale_x = self._window_size[0] / max(image.shape[1], 1)
        scale_y = self._window_size[1] / max(image.shape[0], 1)

        lane_colors = {
            "lane_left": (255, 215, 0),
            "lane_right": (80, 220, 255),
        }
        for lane in lanes:
            points = [
                (int(round(float(point[0]) * scale_x)), int(round(float(point[1]) * scale_y)))
                for point in np.asarray(lane.polyline_image, dtype=np.float32)
            ]
            if len(points) < 2:
                continue
            color = lane_colors.get(lane.lane_id, (255, 255, 255))
            pygame.draw.lines(screen, color, False, points, width=4)

        header_lines = [
            f"EgoLanes View | source: {lane_source}",
            f"tick: {bundle.tick_id}  sim: {bundle.sim_time_s:.2f}s  lanes: {len(lanes)}",
        ] + list(status_lines)
        self._draw_hud(screen, pygame, font, title_font, header_lines)
        pygame.display.flip()
        return True

    def _init_pygame(self, image_width: int, image_height: int) -> bool:
        if self._pygame is not None:
            return True
        try:
            import pygame
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.logger.warning("Pygame EgoLanes viewer unavailable: %s", exc)
            return False
        pygame.init()
        max_width = min(max(image_width, 640), 1440)
        scale = max_width / float(max(image_width, 1))
        height = int(round(image_height * scale))
        self._window_size = (max_width, max(height, 360))
        self._screen = pygame.display.set_mode(self._window_size)
        pygame.display.set_caption(self.window_title)
        self._font = pygame.font.SysFont("Consolas", 18)
        self._title_font = pygame.font.SysFont("Consolas", 24, bold=True)
        self._pygame = pygame
        return True

    def _draw_hud(self, screen, pygame, font, title_font, lines: list[str]) -> None:  # noqa: ANN001
        if not lines:
            return
        title = lines[0]
        body_lines = lines[1:]
        title_surface = title_font.render(title, True, (244, 247, 252))
        body_surfaces = [font.render(line, True, (214, 221, 233)) for line in body_lines]
        width = max(
            [title_surface.get_width(), *[surface.get_width() for surface in body_surfaces], 260]
        )
        height = 18 + title_surface.get_height() + (len(body_surfaces) * 22) + 12
        panel = pygame.Surface((width + 20, height), pygame.SRCALPHA)
        panel.fill((8, 12, 20, 185))
        pygame.draw.rect(panel, (64, 92, 122, 220), panel.get_rect(), width=1)
        panel.blit(title_surface, (10, 8))
        for index, surface in enumerate(body_surfaces):
            panel.blit(surface, (10, 18 + title_surface.get_height() + (index * 22)))
        screen.blit(panel, (14, 14))
