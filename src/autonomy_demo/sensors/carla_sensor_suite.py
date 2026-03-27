from __future__ import annotations

from queue import Empty, Queue
from typing import Any

import numpy as np

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.types import (
    CameraFrame,
    GnssReading,
    ImuReading,
    LidarFrame,
    RadarFrame,
    SensorFrameBundle,
)


class CarlaSensorSuite:
    """Attach CARLA sensors and assemble synchronized typed frame bundles."""

    def __init__(self, sensor_config: dict, backend) -> None:
        self.sensor_config = sensor_config.get("sensors", {})
        self.backend = backend
        self.logger = get_logger(__name__, backend="carla")
        self._queues: dict[str, Queue[Any]] = {}
        self._latest_optional: dict[str, Any] = {}
        self._required_names = ("front_camera", "rear_camera", "left_camera", "right_camera", "lidar", "radar", "gnss", "imu")
        self._optional_names = ("semantic_camera",)

    def setup(self) -> None:
        state = self.backend.state
        if state.world is None or state.ego_actor is None or state.carla is None:
            raise CarlaRuntimeError("CARLA backend must be bootstrapped before sensor setup.")
        for sensor_name, config in self.sensor_config.items():
            actor = self._spawn_sensor(sensor_name, config)
            self.backend.state.sensor_actors[sensor_name] = actor
            queue: Queue[Any] = Queue()
            self._queues[sensor_name] = queue
            actor.listen(queue.put)
        self.logger.info("Attached %s CARLA sensors", len(self._queues))

    def warmup(self, simulation, warmup_ticks: int = 2) -> None:
        for index in range(warmup_ticks):
            simulation.tick(-(index + 1))
        self.logger.info("Completed %s CARLA sensor warm-up ticks", warmup_ticks)

    def _spawn_sensor(self, sensor_name: str, config: dict[str, Any]):
        state = self.backend.state
        bp_id = self._blueprint_id_for_sensor(config["type"])
        blueprint = state.blueprint_library.find(bp_id)
        self._apply_sensor_attributes(blueprint, config)
        transform = self._transform_for_sensor(sensor_name)
        return state.world.spawn_actor(blueprint, transform, attach_to=state.ego_actor)

    def _blueprint_id_for_sensor(self, sensor_type: str) -> str:
        mapping = {
            "rgb_camera": "sensor.camera.rgb",
            "semantic_camera": "sensor.camera.semantic_segmentation",
            "lidar": "sensor.lidar.ray_cast",
            "radar": "sensor.other.radar",
            "gnss": "sensor.other.gnss",
            "imu": "sensor.other.imu",
        }
        if sensor_type not in mapping:
            raise CarlaRuntimeError(f"Unsupported sensor type: {sensor_type}")
        return mapping[sensor_type]

    def _apply_sensor_attributes(self, blueprint, config: dict[str, Any]) -> None:
        sensor_type = config["type"]
        sync_fps = self.backend.runtime_config.carla_sync_fps
        # TODO(PRD 3.2.2): honor lower-frequency sensor configs with multi-rate synchronization.
        blueprint.set_attribute("sensor_tick", f"{1.0 / float(sync_fps):.6f}")
        if sensor_type in {"rgb_camera", "semantic_camera"}:
            blueprint.set_attribute("image_size_x", str(config.get("width", 1280)))
            blueprint.set_attribute("image_size_y", str(config.get("height", 720)))
            blueprint.set_attribute("fov", str(config.get("fov_deg", 90)))
        elif sensor_type == "lidar":
            blueprint.set_attribute("channels", str(config.get("channels", 64)))
            blueprint.set_attribute("range", str(config.get("range_m", 80)))
            blueprint.set_attribute("points_per_second", str(config.get("points_per_second", 1000000)))
            blueprint.set_attribute("rotation_frequency", str(config.get("rotation_hz", 10)))
        elif sensor_type == "radar":
            blueprint.set_attribute("range", str(config.get("range_m", 50)))
            blueprint.set_attribute("horizontal_fov", str(config.get("azimuth_deg", 30)))

    def _transform_for_sensor(self, sensor_name: str):
        carla = self.backend.state.carla
        if sensor_name == "front_camera":
            return carla.Transform(carla.Location(x=2.3, z=0.8))
        if sensor_name == "rear_camera":
            return carla.Transform(carla.Location(x=-2.0, z=1.0), carla.Rotation(yaw=180.0))
        if sensor_name == "left_camera":
            return carla.Transform(carla.Location(y=-0.8, z=1.0), carla.Rotation(yaw=-90.0))
        if sensor_name == "right_camera":
            return carla.Transform(carla.Location(y=0.8, z=1.0), carla.Rotation(yaw=90.0))
        if sensor_name == "semantic_camera":
            return carla.Transform(carla.Location(x=2.3, z=0.8))
        if sensor_name == "lidar":
            return carla.Transform(carla.Location(z=2.2))
        if sensor_name == "radar":
            return carla.Transform(carla.Location(x=2.0, z=0.5))
        if sensor_name == "gnss":
            return carla.Transform(carla.Location(z=1.5))
        if sensor_name == "imu":
            return carla.Transform(carla.Location(z=1.5))
        return carla.Transform()

    def capture(self, tick_id: int, sim_time_s: float) -> SensorFrameBundle:
        frame_id = self.backend.state.current_frame
        if frame_id is None:
            raise CarlaRuntimeError("Cannot capture sensors before the backend has ticked.")
        payloads: dict[str, Any] = {}
        for name in self._required_names:
            payloads[name] = self._await_frame(name, frame_id, required=True)
        for name in self._optional_names:
            payload = self._await_frame(name, frame_id, required=False)
            if payload is not None:
                payloads[name] = payload
        ego_transform = self.backend.state.ego_actor.get_transform()
        snapshot = self.backend.state.current_snapshot
        snapshot_time = sim_time_s
        if snapshot is not None:
            snapshot_time = float(snapshot.timestamp.elapsed_seconds)
        return SensorFrameBundle(
            tick_id=tick_id,
            sim_time_s=snapshot_time,
            front_camera=self._camera_frame("front_camera", payloads["front_camera"]),
            rear_camera=self._camera_frame("rear_camera", payloads["rear_camera"]),
            left_camera=self._camera_frame("left_camera", payloads["left_camera"]),
            right_camera=self._camera_frame("right_camera", payloads["right_camera"]),
            lidar=self._lidar_frame(payloads["lidar"]),
            radar=self._radar_frame(payloads["radar"]),
            gnss=self._gnss_reading(payloads["gnss"], ego_transform.location),
            imu=self._imu_reading(payloads["imu"]),
            semantic_camera=self._camera_frame("semantic_camera", payloads["semantic_camera"])
            if "semantic_camera" in payloads
            else None,
            metadata={"synthetic": False, "carla_frame": frame_id},
        )

    def _await_frame(self, sensor_name: str, frame_id: int, required: bool):
        queue = self._queues.get(sensor_name)
        if queue is None:
            if required:
                raise CarlaRuntimeError(f"Sensor queue missing for {sensor_name}")
            return self._latest_optional.get(sensor_name)
        while True:
            try:
                data = queue.get(timeout=self.backend.runtime_config.carla_timeout_s)
            except Empty as exc:
                if required:
                    raise CarlaRuntimeError(
                        f"Timed out waiting for sensor '{sensor_name}' on frame {frame_id}"
                    ) from exc
                return self._latest_optional.get(sensor_name)
            data_frame = getattr(data, "frame", None)
            if data_frame is None:
                if required:
                    raise CarlaRuntimeError(f"Sensor '{sensor_name}' produced data without a frame id")
                return self._latest_optional.get(sensor_name)
            if data_frame < frame_id:
                continue
            if data_frame > frame_id:
                self.logger.debug(
                    "Using sensor '%s' frame %s for requested frame %s during startup sync",
                    sensor_name,
                    data_frame,
                    frame_id,
                )
                self._latest_optional[sensor_name] = data
                return data
            self._latest_optional[sensor_name] = data
            return data

    def _camera_frame(self, sensor_id: str, image) -> CameraFrame:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1].astype(np.float32)
        return CameraFrame(
            sensor_id=sensor_id,
            frame=array,
            timestamp_s=float(image.timestamp),
            frame_id=int(image.frame),
        )

    def _lidar_frame(self, lidar_data) -> LidarFrame:
        raw = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
        raw = np.reshape(raw, (-1, 4))
        return LidarFrame(
            points_xyz=raw[:, :3],
            intensity=raw[:, 3],
            timestamp_s=float(lidar_data.timestamp),
            frame_id=int(lidar_data.frame),
        )

    def _radar_frame(self, radar_data) -> RadarFrame:
        detections = np.array(
            [
                [float(det.depth), float(det.velocity), float(det.azimuth), float(det.altitude)]
                for det in radar_data
            ],
            dtype=np.float32,
        )
        return RadarFrame(
            detections=detections,
            timestamp_s=float(radar_data.timestamp),
            frame_id=int(radar_data.frame),
        )

    def _gnss_reading(self, gnss_data, location) -> GnssReading:
        return GnssReading(
            world_xyz=np.array([location.x, location.y, location.z], dtype=np.float32),
            timestamp_s=float(gnss_data.timestamp),
            frame_id=int(gnss_data.frame),
        )

    def _imu_reading(self, imu_data) -> ImuReading:
        return ImuReading(
            acceleration_xyz=np.array(
                [
                    imu_data.accelerometer.x,
                    imu_data.accelerometer.y,
                    imu_data.accelerometer.z,
                ],
                dtype=np.float32,
            ),
            gyro_xyz=np.array(
                [imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z],
                dtype=np.float32,
            ),
            timestamp_s=float(imu_data.timestamp),
            frame_id=int(imu_data.frame),
        )
