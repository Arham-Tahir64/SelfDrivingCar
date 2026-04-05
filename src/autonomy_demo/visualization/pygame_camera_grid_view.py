from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import ObjectClass


class PygameCameraGridVisualizationService:
    """Four-camera debug viewer with per-camera boxes plus a full-width LiDAR strip."""

    def __init__(self, *, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.logger = get_logger(__name__, mode="camera_grid_view")
        self._enabled = True
        self._pygame = None
        self._screen = None
        self._font = None
        self._title_font = None
        self._window_size = (1200, 900)
        self._top_grid_height = 700
        self._lidar_strip_height = self._window_size[1] - self._top_grid_height
        self._camera_order = (
            ("front_camera", "Front"),
            ("rear_camera", "Rear"),
            ("left_camera", "Left"),
            ("right_camera", "Right"),
        )
        self._lidar_meters_to_pixels = 6.5

    def attach(self, event_bus) -> None:
        if not self._enabled:
            return
        if not self._init_pygame():
            self._enabled = False
            return
        self.logger.info("Pygame camera grid viewer attached")

    def flush(self) -> None:
        pygame = self._pygame
        if pygame is not None:
            pygame.quit()

    def update_bundle(self, bundle) -> None:
        if not self._enabled:
            return
        if self._screen is None or self._pygame is None:
            return
        self._render(bundle)

    def _init_pygame(self) -> bool:
        if self._pygame is not None:
            return True
        try:
            import pygame
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.logger.warning("Pygame camera grid unavailable: %s", exc)
            return False
        pygame.init()
        pygame.display.set_caption("autonomy_demo/camera_grid")
        self._screen = pygame.display.set_mode(self._window_size)
        self._font = pygame.font.SysFont("Consolas", 18)
        self._title_font = pygame.font.SysFont("Consolas", 22, bold=True)
        self._pygame = pygame
        return True

    def _render(self, bundle) -> None:
        pygame = self._pygame
        screen = self._screen
        font = self._font
        title_font = self._title_font
        if pygame is None or screen is None or font is None or title_font is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._enabled = False
                pygame.quit()
                return

        screen.fill((8, 10, 16))
        tile_width = self._window_size[0] // 2
        tile_height = self._top_grid_height // 2
        detections_by_sensor = bundle.metadata.get("perception_camera_detections", {}) or {}
        capture_metadata = bundle.metadata.get("camera_capture", {}) or {}
        seg_maps = bundle.metadata.get("semantic_seg_map", {}) or {}

        for index, (sensor_id, label) in enumerate(self._camera_order):
            camera = getattr(bundle, sensor_id, None)
            if camera is None:
                continue
            tile_x = (index % 2) * tile_width
            tile_y = (index // 2) * tile_height
            tile_rect = pygame.Rect(tile_x, tile_y, tile_width, tile_height)
            self._draw_camera_tile(
                tile_rect=tile_rect,
                camera=camera,
                title=label,
                detections=detections_by_sensor.get(sensor_id, []),
                capture=capture_metadata.get(sensor_id, {}),
                seg_map=seg_maps.get(sensor_id),
            )

        lidar_rect = pygame.Rect(0, self._top_grid_height, self._window_size[0], self._lidar_strip_height)
        self._draw_lidar_strip(
            tile_rect=lidar_rect,
            lidar_points=np.asarray(bundle.lidar.points_xyz, dtype=np.float32),
            detections=bundle.metadata.get("debug_perception_detections", []) or [],
            ego_xyz=np.asarray(bundle.gnss.world_xyz, dtype=np.float32),
            ego_yaw_rad=float(bundle.metadata.get("ego_yaw_rad", 0.0)),
        )

        hud_lines = [
            f"Tick {bundle.tick_id}",
            f"Time {bundle.sim_time_s:.1f}s",
            f"Model {bundle.metadata.get('perception_summary').active_mode if bundle.metadata.get('perception_summary') else '--'}",
        ]
        for index, line in enumerate(hud_lines):
            screen.blit(font.render(line, True, (220, 225, 235)), (18, 14 + (index * 22)))

        pygame.display.flip()

    def _draw_camera_tile(
        self,
        *,
        tile_rect,
        camera,
        title: str,
        detections: list[dict[str, Any]],
        capture: dict[str, Any],
        seg_map=None,
    ) -> None:
        pygame = self._pygame
        screen = self._screen
        font = self._font
        title_font = self._title_font
        if pygame is None or screen is None or font is None or title_font is None:
            return

        margin = 12
        header_height = 34
        content_rect = pygame.Rect(
            tile_rect.x + margin,
            tile_rect.y + header_height + margin,
            tile_rect.width - margin * 2,
            tile_rect.height - header_height - margin * 2,
        )

        pygame.draw.rect(screen, (18, 22, 30), tile_rect)
        pygame.draw.rect(screen, (42, 48, 62), tile_rect, width=1)

        image = np.clip(np.asarray(camera.frame, dtype=np.float32), 0.0, 255.0).astype(np.uint8)
        surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        scaled = pygame.transform.smoothscale(surface, (content_rect.width, content_rect.height))
        screen.blit(scaled, content_rect.topleft)

        # Semantic segmentation overlay
        present_class_ids = self._draw_seg_overlay(screen, pygame, content_rect, seg_map)

        source_width = max(int(image.shape[1]), 1)
        source_height = max(int(image.shape[0]), 1)
        scale_x = content_rect.width / source_width
        scale_y = content_rect.height / source_height
        for detection in detections:
            bbox = detection.get("bbox_xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1 = int(content_rect.x + float(bbox[0]) * scale_x)
            y1 = int(content_rect.y + float(bbox[1]) * scale_y)
            x2 = int(content_rect.x + float(bbox[2]) * scale_x)
            y2 = int(content_rect.y + float(bbox[3]) * scale_y)
            color = self._box_color(str(detection.get("source_modality", "camera")))
            pygame.draw.rect(screen, color, pygame.Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1)), width=2)
            label = f"{detection.get('object_class', 'obj')} {float(detection.get('confidence', 0.0)):.2f}"
            label_surface = font.render(label, True, color)
            screen.blit(label_surface, (x1, max(content_rect.y, y1 - 20)))

        # Segmentation legend (dynamic — only classes present in this frame)
        if present_class_ids is not None:
            self._draw_seg_legend(screen, pygame, font, content_rect, present_class_ids)

        status_text = str(capture.get("status", getattr(camera.status, "value", "OK")))
        meta_text = f"{status_text} | det:{len(detections)}"
        screen.blit(title_font.render(title, True, (236, 240, 247)), (tile_rect.x + 14, tile_rect.y + 10))
        screen.blit(font.render(meta_text, True, (147, 160, 184)), (tile_rect.x + 110, tile_rect.y + 14))

    def _draw_seg_overlay(self, screen, pygame, content_rect, seg_map) -> list[int] | None:
        """Blend colorized semantic segmentation onto the camera tile. Returns present class IDs."""
        if seg_map is None:
            return None
        try:
            from autonomy_demo.perception.cityscapes_palette import CITYSCAPES_PALETTE

            label_map = np.asarray(seg_map.label_map, dtype=np.uint8)
            color_map = CITYSCAPES_PALETTE[label_map]  # (H, W, 3) uint8

            # Build a semi-transparent overlay surface
            overlay_surface = pygame.surfarray.make_surface(np.transpose(color_map, (1, 0, 2)))
            overlay_surface = pygame.transform.smoothscale(
                overlay_surface, (content_rect.width, content_rect.height)
            )
            overlay_surface.set_alpha(120)
            screen.blit(overlay_surface, content_rect.topleft)

            present_ids = sorted(set(np.unique(label_map).tolist()))
            return present_ids
        except Exception:
            return None

    def _draw_seg_legend(self, screen, pygame, font, content_rect, present_class_ids: list[int]) -> None:
        """Draw a compact legend strip at the bottom of the camera tile for visible classes."""
        try:
            from autonomy_demo.perception.cityscapes_palette import (
                CITYSCAPES_LABELS,
                CITYSCAPES_PALETTE,
            )

            legend_x = content_rect.x + 4
            legend_y = content_rect.bottom - 18
            swatch_size = 10
            padding = 6

            # Semi-transparent background bar
            bar_surface = pygame.Surface((content_rect.width, 20), pygame.SRCALPHA)
            bar_surface.fill((0, 0, 0, 160))
            screen.blit(bar_surface, (content_rect.x, legend_y - 2))

            for class_id in present_class_ids:
                if class_id >= len(CITYSCAPES_LABELS):
                    continue
                color = tuple(int(c) for c in CITYSCAPES_PALETTE[class_id])
                label_text = CITYSCAPES_LABELS[class_id]
                text_surface = font.render(label_text, True, (220, 225, 235))
                entry_width = swatch_size + 3 + text_surface.get_width() + padding

                # Stop if we'd overflow the tile width
                if legend_x + entry_width > content_rect.right - 4:
                    break

                pygame.draw.rect(
                    screen, color, pygame.Rect(legend_x, legend_y, swatch_size, swatch_size)
                )
                screen.blit(text_surface, (legend_x + swatch_size + 3, legend_y - 3))
                legend_x += entry_width
        except Exception:
            pass

    def _draw_lidar_strip(
        self,
        *,
        tile_rect,
        lidar_points: np.ndarray,
        detections: list[Any],
        ego_xyz: np.ndarray,
        ego_yaw_rad: float,
    ) -> None:
        pygame = self._pygame
        screen = self._screen
        font = self._font
        title_font = self._title_font
        if pygame is None or screen is None or font is None or title_font is None:
            return

        pygame.draw.rect(screen, (18, 22, 30), tile_rect)
        pygame.draw.rect(screen, (42, 48, 62), tile_rect, width=1)

        margin = 16
        content_rect = pygame.Rect(
            tile_rect.x + margin,
            tile_rect.y + 42,
            tile_rect.width - margin * 2,
            tile_rect.height - 54,
        )
        pygame.draw.rect(screen, (10, 12, 18), content_rect)

        anchor = np.array(
            [content_rect.x + content_rect.width * 0.5, content_rect.y + content_rect.height * 0.82],
            dtype=np.float32,
        )
        self._draw_lidar_grid(content_rect, anchor)

        for point in lidar_points:
            x, y = self._ego_xy_to_screen(float(point[0]), float(point[1]), anchor)
            if content_rect.collidepoint(x, y):
                pygame.draw.circle(screen, (120, 200, 255), (x, y), 2)

        for detection in detections:
            corners = self._bbox_bottom_corners_in_ego_frame(detection, ego_xyz, ego_yaw_rad, anchor)
            if corners is None:
                continue
            pygame.draw.polygon(screen, self._detection_color(getattr(detection, "object_class", "")), corners, width=2)

        ego_width = 1.9 * self._lidar_meters_to_pixels * 0.5
        ego_length = 4.6 * self._lidar_meters_to_pixels
        ego_points = [
            (int(anchor[0]), int(anchor[1] - ego_length * 0.7)),
            (int(anchor[0] - ego_width), int(anchor[1] + ego_length * 0.3)),
            (int(anchor[0] + ego_width), int(anchor[1] + ego_length * 0.3)),
        ]
        pygame.draw.polygon(screen, (240, 245, 250), ego_points)
        pygame.draw.polygon(screen, (60, 220, 255), ego_points, width=2)

        screen.blit(title_font.render("LiDAR", True, (236, 240, 247)), (tile_rect.x + 14, tile_rect.y + 10))
        meta = f"points:{len(lidar_points)} | det:{len(detections)}"
        screen.blit(font.render(meta, True, (147, 160, 184)), (tile_rect.x + 110, tile_rect.y + 14))

    def _draw_lidar_grid(self, content_rect, anchor: np.ndarray) -> None:
        pygame = self._pygame
        screen = self._screen
        if pygame is None or screen is None:
            return
        for meters in range(-80, 85, 5):
            x = int(anchor[0] + meters * self._lidar_meters_to_pixels)
            pygame.draw.line(screen, (28, 34, 48), (x, content_rect.y), (x, content_rect.bottom), 1)
        for meters in range(0, 55, 5):
            y = int(anchor[1] - meters * self._lidar_meters_to_pixels)
            pygame.draw.line(screen, (28, 34, 48), (content_rect.x, y), (content_rect.right, y), 1)

    def _bbox_bottom_corners_in_ego_frame(
        self,
        detection: Any,
        ego_xyz: np.ndarray,
        ego_yaw_rad: float,
        anchor: np.ndarray,
    ) -> list[tuple[int, int]] | None:
        world_bbox = getattr(detection, "world_bbox_3d", None)
        if world_bbox is None or len(world_bbox) < 4:
            return None
        corners: list[tuple[int, int]] = []
        for point in np.asarray(world_bbox, dtype=np.float32)[:4]:
            ego_xy = self._world_to_ego_xy(point, ego_xyz, ego_yaw_rad)
            corners.append(self._ego_xy_to_screen(float(ego_xy[0]), float(ego_xy[1]), anchor))
        return corners

    def _world_to_ego_xy(self, world_xyz: np.ndarray, ego_xyz: np.ndarray, ego_yaw_rad: float) -> np.ndarray:
        delta = np.asarray(world_xyz, dtype=np.float32)[:2] - np.asarray(ego_xyz, dtype=np.float32)[:2]
        rotation = np.array(
            [
                [math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)],
                [-math.sin(ego_yaw_rad), math.cos(ego_yaw_rad)],
            ],
            dtype=np.float32,
        )
        return rotation @ delta

    def _ego_xy_to_screen(self, forward_m: float, lateral_m: float, anchor: np.ndarray) -> tuple[int, int]:
        x = int(anchor[0] + lateral_m * self._lidar_meters_to_pixels)
        y = int(anchor[1] - forward_m * self._lidar_meters_to_pixels)
        return x, y

    def _box_color(self, modality: str) -> tuple[int, int, int]:
        if modality == "camera":
            return (77, 208, 225)
        if modality == "bootstrap":
            return (255, 92, 138)
        return (255, 196, 0)

    def _detection_color(self, object_class: ObjectClass | str) -> tuple[int, int, int]:
        object_value = object_class.value if isinstance(object_class, ObjectClass) else str(object_class)
        if object_value == ObjectClass.VEHICLE.value:
            return (0, 229, 255)
        if object_value in {ObjectClass.PEDESTRIAN.value, ObjectClass.CYCLIST.value}:
            return (255, 196, 0)
        return (255, 255, 255)
