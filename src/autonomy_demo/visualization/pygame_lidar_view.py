from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import ObjectClass, TopicName
from autonomy_demo.interfaces.types import EgoPose, LidarFrame, ObjectDetection


class PygameLidarVisualizationService:
    """Top-down LiDAR point cloud viewer for quick conceptual debugging."""

    def __init__(self, *, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.logger = get_logger(__name__, mode="lidar_view")
        self._enabled = True
        self._pygame = None
        self._screen = None
        self._font = None
        self._latest_lidar: LidarFrame | None = None
        self._latest_detections: list[ObjectDetection] = []
        self._latest_ego_pose: EgoPose | None = None
        self._latest_tick_id = -1
        self._latest_sim_time_s = 0.0
        self._size = (780, 780)
        self._meters_to_pixels = 7.0

    def attach(self, event_bus) -> None:
        if not self._enabled:
            return
        if not self._init_pygame():
            self._enabled = False
            return
        event_bus.subscribe(TopicName.SENSOR_LIDAR.value, self._handle)
        event_bus.subscribe(TopicName.PERCEPTION_DETECTIONS.value, self._handle)
        event_bus.subscribe(TopicName.LOCALIZATION_EGO_POSE.value, self._handle)
        event_bus.subscribe(TopicName.TICK_COMPLETE.value, self._handle)
        self.logger.info("Pygame LiDAR viewer attached")

    def flush(self) -> None:
        pygame = self._pygame
        if pygame is not None:
            pygame.quit()

    def _init_pygame(self) -> bool:
        if self._pygame is not None:
            return True
        try:
            import pygame
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.logger.warning("Pygame LiDAR view unavailable: %s", exc)
            return False
        pygame.init()
        pygame.display.set_caption("autonomy_demo/lidar_view")
        self._screen = pygame.display.set_mode(self._size)
        self._font = pygame.font.SysFont("Consolas", 18)
        self._pygame = pygame
        return True

    def _handle(self, topic: str, payload: Any) -> None:
        if not self._enabled:
            return
        if topic == TopicName.SENSOR_LIDAR.value and isinstance(payload, LidarFrame):
            self._latest_lidar = payload
        elif topic == TopicName.PERCEPTION_DETECTIONS.value and isinstance(payload, list):
            self._latest_detections = [item for item in payload if isinstance(item, ObjectDetection)]
        elif topic == TopicName.LOCALIZATION_EGO_POSE.value and isinstance(payload, EgoPose):
            self._latest_ego_pose = payload
        elif topic == TopicName.TICK_COMPLETE.value and isinstance(payload, dict):
            self._latest_tick_id = int(payload.get("tick_id", -1))
            self._latest_sim_time_s = float(payload.get("sim_time_s", 0.0))
            self._render()

    def _render(self) -> None:
        pygame = self._pygame
        screen = self._screen
        font = self._font
        if pygame is None or screen is None or font is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._enabled = False
                pygame.quit()
                return

        width, height = self._size
        anchor = np.array([width * 0.5, height * 0.82], dtype=np.float32)
        screen.fill((10, 12, 18))
        self._draw_grid(anchor)
        self._draw_axes(anchor)
        self._draw_lidar_points(anchor)
        self._draw_detections(anchor)
        self._draw_ego(anchor)

        info_lines = [
            f"tick: {self._latest_tick_id}",
            f"time: {self._latest_sim_time_s:.1f}s",
            f"points: {0 if self._latest_lidar is None else len(self._latest_lidar.points_xyz)}",
            f"detections: {len(self._latest_detections)}",
        ]
        for index, line in enumerate(info_lines):
            surface = font.render(line, True, (220, 225, 235))
            screen.blit(surface, (18, 16 + (index * 22)))

        legend = [
            ("LiDAR points", (120, 200, 255)),
            ("Vehicle", (0, 229, 255)),
            ("Ped/Cyclist", (255, 196, 0)),
        ]
        for index, (label, color) in enumerate(legend):
            y = 16 + (index * 22)
            pygame.draw.circle(screen, color, (width - 170, y + 9), 5)
            surface = font.render(label, True, (220, 225, 235))
            screen.blit(surface, (width - 155, y))

        pygame.display.flip()

    def _draw_grid(self, anchor: np.ndarray) -> None:
        pygame = self._pygame
        screen = self._screen
        if pygame is None or screen is None:
            return
        for meters in range(-50, 55, 5):
            x = int(anchor[0] + meters * self._meters_to_pixels)
            pygame.draw.line(screen, (28, 34, 48), (x, 0), (x, self._size[1]), 1)
        for meters in range(0, 85, 5):
            y = int(anchor[1] - meters * self._meters_to_pixels)
            pygame.draw.line(screen, (28, 34, 48), (0, y), (self._size[0], y), 1)

    def _draw_axes(self, anchor: np.ndarray) -> None:
        pygame = self._pygame
        screen = self._screen
        if pygame is None or screen is None:
            return
        pygame.draw.line(screen, (80, 96, 120), (int(anchor[0]), 0), (int(anchor[0]), self._size[1]), 1)
        pygame.draw.line(screen, (80, 96, 120), (0, int(anchor[1])), (self._size[0], int(anchor[1])), 1)

    def _draw_lidar_points(self, anchor: np.ndarray) -> None:
        pygame = self._pygame
        screen = self._screen
        if pygame is None or screen is None or self._latest_lidar is None:
            return
        for point in np.asarray(self._latest_lidar.points_xyz, dtype=np.float32):
            x, y = self._ego_xy_to_screen(float(point[0]), float(point[1]), anchor)
            if 0 <= x < self._size[0] and 0 <= y < self._size[1]:
                pygame.draw.circle(screen, (120, 200, 255), (x, y), 2)

    def _draw_detections(self, anchor: np.ndarray) -> None:
        pygame = self._pygame
        screen = self._screen
        if pygame is None or screen is None:
            return
        for detection in self._latest_detections:
            corners = self._bbox_bottom_corners_in_ego_frame(detection)
            if corners is None:
                continue
            color = self._detection_color(detection.object_class)
            pygame.draw.polygon(screen, color, corners, width=2)

    def _draw_ego(self, anchor: np.ndarray) -> None:
        pygame = self._pygame
        screen = self._screen
        if pygame is None or screen is None:
            return
        ego_width = 1.9 * self._meters_to_pixels * 0.5
        ego_length = 4.6 * self._meters_to_pixels
        points = [
            (int(anchor[0]), int(anchor[1] - ego_length * 0.7)),
            (int(anchor[0] - ego_width), int(anchor[1] + ego_length * 0.3)),
            (int(anchor[0] + ego_width), int(anchor[1] + ego_length * 0.3)),
        ]
        pygame.draw.polygon(screen, (240, 245, 250), points)
        pygame.draw.polygon(screen, (60, 220, 255), points, width=2)

    def _bbox_bottom_corners_in_ego_frame(self, detection: ObjectDetection) -> list[tuple[int, int]] | None:
        if detection.world_bbox_3d is None or len(detection.world_bbox_3d) < 4:
            return None
        anchor = np.array([self._size[0] * 0.5, self._size[1] * 0.82], dtype=np.float32)
        corners: list[tuple[int, int]] = []
        for point in np.asarray(detection.world_bbox_3d, dtype=np.float32)[:4]:
            ego_xy = self._world_to_ego_xy(point)
            if ego_xy is None:
                return None
            corners.append(self._ego_xy_to_screen(float(ego_xy[0]), float(ego_xy[1]), anchor))
        return corners

    def _world_to_ego_xy(self, world_xyz: np.ndarray) -> np.ndarray | None:
        if self._latest_ego_pose is None:
            return None
        ego_xyz = np.asarray(self._latest_ego_pose.world_xyz, dtype=np.float32)
        delta = np.asarray(world_xyz, dtype=np.float32)[:2] - ego_xyz[:2]
        yaw = float(self._latest_ego_pose.yaw_rad)
        rotation = np.array(
            [
                [math.cos(yaw), math.sin(yaw)],
                [-math.sin(yaw), math.cos(yaw)],
            ],
            dtype=np.float32,
        )
        return rotation @ delta

    def _ego_xy_to_screen(self, forward_m: float, lateral_m: float, anchor: np.ndarray) -> tuple[int, int]:
        x = int(anchor[0] + lateral_m * self._meters_to_pixels)
        y = int(anchor[1] - forward_m * self._meters_to_pixels)
        return x, y

    def _detection_color(self, object_class: ObjectClass | str) -> tuple[int, int, int]:
        object_value = object_class.value if isinstance(object_class, ObjectClass) else str(object_class)
        if object_value == ObjectClass.VEHICLE.value:
            return (0, 229, 255)
        if object_value in {ObjectClass.PEDESTRIAN.value, ObjectClass.CYCLIST.value}:
            return (255, 196, 0)
        return (255, 255, 255)
