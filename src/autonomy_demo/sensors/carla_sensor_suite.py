from __future__ import annotations

import math
from queue import Empty, Queue
from typing import Any

import numpy as np

from autonomy_demo.common.exceptions import CarlaRuntimeError
from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import ObjectClass, SensorStatus, TrafficLightState
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
        self._latest_payloads: dict[str, Any] = {}
        self._required_names = (
            "front_camera",
            "rear_camera",
            "left_camera",
            "right_camera",
            "lidar",
            "radar",
            "gnss",
            "imu",
        )
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
        self._apply_sensor_attributes(sensor_name, blueprint, config)
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

    def _apply_sensor_attributes(self, sensor_name: str, blueprint, config: dict[str, Any]) -> None:
        sensor_type = config["type"]
        blueprint.set_attribute("sensor_tick", f"{self._sensor_tick_seconds(sensor_name):.6f}")
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
            return carla.Transform(
                carla.Location(x=-2.6, z=1.35),
                carla.Rotation(yaw=180.0, pitch=-4.0),
            )
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
        ego_velocity = self.backend.state.ego_actor.get_velocity()
        ego_acceleration = self.backend.state.ego_actor.get_acceleration()
        ego_speed_mps = math.sqrt(
            (ego_velocity.x ** 2) + (ego_velocity.y ** 2) + (ego_velocity.z ** 2)
        )
        ego_acceleration_mps2 = math.sqrt(
            (ego_acceleration.x ** 2) + (ego_acceleration.y ** 2) + (ego_acceleration.z ** 2)
        )
        lane_waypoint = self.backend.state.world.get_map().get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=self.backend.state.carla.LaneType.Driving,
        )
        lane_id = "lane_001"
        if lane_waypoint is not None:
            lane_id = (
                f"road_{lane_waypoint.road_id}:section_{lane_waypoint.section_id}:lane_{lane_waypoint.lane_id}"
            )
        snapshot = self.backend.state.current_snapshot
        snapshot_time = sim_time_s
        if snapshot is not None:
            snapshot_time = float(snapshot.timestamp.elapsed_seconds)

        front_camera = self._camera_frame("front_camera", payloads["front_camera"], frame_id)
        rear_camera = self._camera_frame("rear_camera", payloads["rear_camera"], frame_id)
        left_camera = self._camera_frame("left_camera", payloads["left_camera"], frame_id)
        right_camera = self._camera_frame("right_camera", payloads["right_camera"], frame_id)
        semantic_camera = (
            self._camera_frame("semantic_camera", payloads["semantic_camera"], frame_id)
            if "semantic_camera" in payloads
            else None
        )
        camera_annotations = {
            sensor_id: self._actor_annotations(sensor_id, payloads[sensor_id])
            for sensor_id in ("front_camera", "rear_camera", "left_camera", "right_camera")
        }

        return SensorFrameBundle(
            tick_id=tick_id,
            sim_time_s=snapshot_time,
            front_camera=front_camera,
            rear_camera=rear_camera,
            left_camera=left_camera,
            right_camera=right_camera,
            lidar=self._lidar_frame(payloads["lidar"]),
            radar=self._radar_frame(payloads["radar"]),
            gnss=self._gnss_reading(payloads["gnss"], ego_transform.location),
            imu=self._imu_reading(payloads["imu"]),
            semantic_camera=semantic_camera,
            metadata={
                "synthetic": False,
                "carla_frame": frame_id,
                "ego_yaw_rad": math.radians(float(ego_transform.rotation.yaw)),
                "ego_speed_mps": float(ego_speed_mps),
                "ego_acceleration_mps2": float(ego_acceleration_mps2),
                "ego_lane_id": lane_id,
                "carla_actor_annotations": camera_annotations["front_camera"],
                "carla_camera_annotations": camera_annotations,
                "camera_capture": {
                    camera.sensor_id: self._camera_capture_metadata(camera, frame_id)
                    for camera in [front_camera, rear_camera, left_camera, right_camera]
                    + ([semantic_camera] if semantic_camera is not None else [])
                },
            },
        )

    def _sensor_tick_seconds(self, sensor_name: str) -> float:
        rate_hz = self._sensor_rate_hz(sensor_name)
        return 1.0 / max(rate_hz, 1.0)

    def _sensor_rate_hz(self, sensor_name: str) -> float:
        config = self.sensor_config.get(sensor_name, {})
        sensor_type = str(config.get("type", ""))
        if sensor_type == "lidar":
            return float(config.get("rotation_hz", self.backend.runtime_config.carla_sync_fps))
        return float(config.get("fps", self.backend.runtime_config.carla_sync_fps))

    def _sensor_max_age_frames(self, sensor_name: str) -> int:
        sensor_rate_hz = self._sensor_rate_hz(sensor_name)
        sync_fps = max(float(self.backend.runtime_config.carla_sync_fps), 1.0)
        return max(1, int(math.ceil(sync_fps / max(sensor_rate_hz, 1.0))))

    def _camera_capture_metadata(self, camera: CameraFrame, current_frame: int) -> dict[str, Any]:
        age_frames = max(int(current_frame - int(camera.frame_id or current_frame)), 0)
        return {
            "frame_id": camera.frame_id,
            "timestamp_s": float(camera.timestamp_s),
            "status": camera.status.value,
            "age_frames": age_frames,
            "configured_fps": self._sensor_rate_hz(camera.sensor_id),
            "image_shape": list(camera.frame.shape),
        }

    def _actor_annotations(self, sensor_name: str, camera_data) -> list[dict[str, Any]]:
        annotations: list[dict[str, Any]] = []
        image_width = int(camera_data.width)
        image_height = int(camera_data.height)
        fov_deg = float(self.sensor_config.get(sensor_name, {}).get("fov_deg", 90.0))
        sensor_actor = self.backend.state.sensor_actors.get(sensor_name)
        if sensor_actor is None:
            return annotations
        camera_transform = sensor_actor.get_transform()
        camera_location = camera_transform.location
        camera_yaw_rad = math.radians(float(camera_transform.rotation.yaw))
        for actor in self.backend.state.world.get_actors():
            if actor.id == self.backend.state.ego_actor.id:
                continue
            object_class = self._object_class_for_actor(actor)
            if object_class is None:
                continue
            actor_location = actor.get_location()
            dx = float(actor_location.x - camera_location.x)
            dy = float(actor_location.y - camera_location.y)
            forward_distance = math.cos(camera_yaw_rad) * dx + math.sin(camera_yaw_rad) * dy
            lateral_distance = -math.sin(camera_yaw_rad) * dx + math.cos(camera_yaw_rad) * dy
            if forward_distance <= 1.0 or forward_distance > 80.0:
                continue
            max_lateral = max(2.0, forward_distance * math.tan(math.radians(fov_deg * 0.5)))
            if abs(lateral_distance) > max_lateral * 1.25:
                continue
            bbox_xyxy = self._approximate_image_bbox(
                object_class=object_class,
                forward_distance=forward_distance,
                lateral_distance=lateral_distance,
                image_width=image_width,
                image_height=image_height,
                max_lateral=max_lateral,
            )
            velocity = actor.get_velocity()
            velocity_xyz = np.array([velocity.x, velocity.y, velocity.z], dtype=np.float32)
            try:
                bounding_box = actor.bounding_box
                world_vertices = bounding_box.get_world_vertices(actor.get_transform())
                world_bbox_3d = np.array(
                    [[vertex.x, vertex.y, vertex.z] for vertex in world_vertices],
                    dtype=np.float32,
                )
            except Exception:
                self.logger.debug(
                    "Skipping actor %s because its world bounding box could not be resolved",
                    actor.id,
                )
                continue
            annotation: dict[str, Any] = {
                "track_id": int(actor.id),
                "object_class": object_class.value,
                "confidence": 1.0,
                "image_bbox_xyxy": bbox_xyxy.tolist(),
                "world_bbox_3d": world_bbox_3d.tolist(),
                "velocity_xyz": velocity_xyz.tolist(),
                "world_xyz": [actor_location.x, actor_location.y, actor_location.z],
            }
            if object_class == ObjectClass.TRAFFIC_LIGHT:
                annotation["traffic_light_state"] = self._traffic_light_state(actor).value
            annotations.append(annotation)
        return annotations

    def _object_class_for_actor(self, actor) -> ObjectClass | None:
        type_id = str(getattr(actor, "type_id", ""))
        if type_id.startswith("vehicle."):
            return ObjectClass.VEHICLE
        if type_id.startswith("walker."):
            return ObjectClass.PEDESTRIAN
        if type_id.startswith("traffic.traffic_light"):
            return ObjectClass.TRAFFIC_LIGHT
        return None

    def _traffic_light_state(self, actor) -> TrafficLightState:
        raw_state = getattr(actor, "state", None)
        if hasattr(actor, "get_state"):
            try:
                raw_state = actor.get_state()
            except Exception:
                pass
        state = str(raw_state or "Unknown").upper()
        if "RED" in state:
            return TrafficLightState.RED
        if "YELLOW" in state or "AMBER" in state:
            return TrafficLightState.AMBER
        if "GREEN" in state:
            return TrafficLightState.GREEN
        return TrafficLightState.UNKNOWN

    def _approximate_image_bbox(
        self,
        *,
        object_class: ObjectClass,
        forward_distance: float,
        lateral_distance: float,
        image_width: int,
        image_height: int,
        max_lateral: float,
    ) -> np.ndarray:
        normalized_x = np.clip(lateral_distance / max(max_lateral, 1.0), -1.0, 1.0)
        center_x = (image_width * 0.5) + normalized_x * (image_width * 0.45)
        base_height = {
            ObjectClass.VEHICLE: 1200.0,
            ObjectClass.PEDESTRIAN: 850.0,
            ObjectClass.TRAFFIC_LIGHT: 500.0,
        }.get(object_class, 900.0)
        box_height = float(np.clip(base_height / max(forward_distance, 1.0), 18.0, image_height * 0.55))
        aspect_ratio = {
            ObjectClass.VEHICLE: 1.4,
            ObjectClass.PEDESTRIAN: 0.45,
            ObjectClass.TRAFFIC_LIGHT: 0.35,
        }.get(object_class, 1.0)
        box_width = max(14.0, box_height * aspect_ratio)
        center_y = image_height * 0.90 - min(forward_distance, 60.0) * 4.0
        x1 = float(np.clip(center_x - box_width * 0.5, 0.0, image_width - 1.0))
        x2 = float(np.clip(center_x + box_width * 0.5, 1.0, image_width))
        y1 = float(np.clip(center_y - box_height, 0.0, image_height - 1.0))
        y2 = float(np.clip(center_y, 1.0, image_height))
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _await_frame(self, sensor_name: str, frame_id: int, required: bool):
        queue = self._queues.get(sensor_name)
        if queue is None:
            if required:
                raise CarlaRuntimeError(f"Sensor queue missing for {sensor_name}")
            return self._latest_payloads.get(sensor_name)

        payload = self._latest_payloads.get(sensor_name)
        best = payload if self._payload_frame(payload) <= frame_id else None

        while True:
            try:
                candidate = queue.get_nowait()
            except Empty:
                break
            self._latest_payloads[sensor_name] = candidate
            if self._payload_frame(candidate) <= frame_id:
                best = candidate

        if self._is_payload_fresh(sensor_name, best, frame_id):
            return best

        while True:
            try:
                candidate = queue.get(timeout=self.backend.runtime_config.carla_timeout_s)
            except Empty as exc:
                if self._is_payload_fresh(sensor_name, best, frame_id):
                    return best
                if required:
                    raise CarlaRuntimeError(
                        f"Timed out waiting for sensor '{sensor_name}' near frame {frame_id}"
                    ) from exc
                return best
            self._latest_payloads[sensor_name] = candidate
            if self._payload_frame(candidate) <= frame_id:
                best = candidate
                if self._is_payload_fresh(sensor_name, best, frame_id):
                    return best
            elif self._is_payload_fresh(sensor_name, best, frame_id):
                return best

    def _payload_frame(self, payload: Any) -> int:
        if payload is None:
            return -1
        return int(getattr(payload, "frame", -1))

    def _is_payload_fresh(self, sensor_name: str, payload: Any, current_frame: int) -> bool:
        if payload is None:
            return False
        payload_frame = self._payload_frame(payload)
        return payload_frame >= 0 and (current_frame - payload_frame) <= self._sensor_max_age_frames(sensor_name)

    def _camera_frame(self, sensor_id: str, image, current_frame: int) -> CameraFrame:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = np.ascontiguousarray(array[:, :, :3][:, :, ::-1].astype(np.float32))
        age_frames = max(current_frame - int(image.frame), 0)
        status = SensorStatus.OK if age_frames == 0 else SensorStatus.DEGRADED
        return CameraFrame(
            sensor_id=sensor_id,
            frame=array,
            timestamp_s=float(image.timestamp),
            frame_id=int(image.frame),
            status=status,
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
