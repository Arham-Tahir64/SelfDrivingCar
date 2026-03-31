from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.types import (
    CameraFrame,
    GnssReading,
    ImuReading,
    LidarFrame,
    RadarFrame,
    SensorFrameBundle,
)


class SensorManager:
    """Synthetic sensor manager for the initial no-CARLA vertical slice."""

    def __init__(self, sensor_config: dict) -> None:
        self.sensor_config = sensor_config

    def setup(self) -> None:
        return None

    def warmup(self, simulation, warmup_ticks: int = 2) -> None:
        return None

    def capture(self, tick_id: int, sim_time_s: float) -> SensorFrameBundle:
        image = np.full((32, 32, 3), fill_value=tick_id, dtype=np.float32)
        lidar_points = np.array(
            [
                [7.5, 1.2, 0.1],
                [7.8, 1.0, 0.5],
                [8.1, 1.4, 1.0],
                [8.5, 1.1, 1.4],
                [8.7, 1.5, 0.9],
                [8.9, 0.9, 0.2],
                [12.0, -1.5, 0.1],
                [12.1, -1.4, 0.4],
                [12.2, -1.6, 0.7],
                [0.0, 0.0, -2.2],
            ],
            dtype=np.float32,
        )
        radar = np.array([[12.0, 0.1, 0.0, 1.2]], dtype=np.float32)
        return SensorFrameBundle(
            tick_id=tick_id,
            sim_time_s=sim_time_s,
            front_camera=CameraFrame("front_camera", image, sim_time_s, frame_id=tick_id),
            rear_camera=CameraFrame("rear_camera", image, sim_time_s, frame_id=tick_id),
            left_camera=CameraFrame("left_camera", image, sim_time_s, frame_id=tick_id),
            right_camera=CameraFrame("right_camera", image, sim_time_s, frame_id=tick_id),
            lidar=LidarFrame(points_xyz=lidar_points, timestamp_s=sim_time_s, frame_id=tick_id),
            radar=RadarFrame(detections=radar, timestamp_s=sim_time_s, frame_id=tick_id),
            gnss=GnssReading(
                world_xyz=np.array([tick_id * 2.0, 50.0, 0.0], dtype=np.float32),
                timestamp_s=sim_time_s,
                frame_id=tick_id,
            ),
            imu=ImuReading(
                acceleration_xyz=np.array([0.1, 0.0, 0.0], dtype=np.float32),
                gyro_xyz=np.array([0.0, 0.0, 0.01], dtype=np.float32),
                timestamp_s=sim_time_s,
                frame_id=tick_id,
            ),
            semantic_camera=CameraFrame("semantic_camera", image, sim_time_s, frame_id=tick_id),
            metadata={"synthetic": True},
        )
