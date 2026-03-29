from __future__ import annotations

from collections import Counter

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import LaneLineType, ObjectClass, TrackState, TrafficLightState
from autonomy_demo.interfaces.types import (
    ConeDetection,
    DrivableSpaceMask,
    LaneLine,
    ObjectDetection,
    PerceptionStatus,
    SensorFrameBundle,
    TrafficLightDetection,
)
from autonomy_demo.perception.drivable_space import DrivableSpaceExtractor
from autonomy_demo.perception.fusion import fuse_detections
from autonomy_demo.perception.internal_types import (
    FrameDetection2D,
    TrackedDetection2D,
    TrackedLidarClusterDetection,
)
from autonomy_demo.perception.lane_extraction import LaneExtractor
from autonomy_demo.perception.lidar_detection import LidarObstacleDetector
from autonomy_demo.perception.lidar_tracking import SimpleCentroidTracker3D
from autonomy_demo.perception.object_detection import (
    YoloObjectDetector,
    bootstrap_annotations_from_metadata,
)
from autonomy_demo.perception.tracking import SimpleSortTracker


def _count_by_modality(
    detections: list[ObjectDetection],
    cones: list[ConeDetection],
    traffic_lights: list[TrafficLightDetection],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for detection in detections:
        counts[str(detection.source_modality)] += 1
    for cone in cones:
        counts[str(cone.source_modality)] += 1
    for traffic_light in traffic_lights:
        counts[str(traffic_light.source_modality)] += 1
    return dict(sorted(counts.items()))


def _build_perception_status(
    *,
    active_mode: str,
    fallback_state: str,
    detections: list[ObjectDetection],
    cones: list[ConeDetection],
    traffic_lights: list[TrafficLightDetection],
    active_camera_sensors: list[str],
) -> PerceptionStatus:
    return PerceptionStatus(
        active_mode=active_mode,
        fallback_state=fallback_state,
        counts_by_modality=_count_by_modality(detections, cones, traffic_lights),
        active_camera_sensors=active_camera_sensors,
        detection_count=len(detections),
        cone_count=len(cones),
        traffic_light_count=len(traffic_lights),
    )


def _record_perception_metadata(
    bundle: SensorFrameBundle,
    *,
    detections: list[ObjectDetection],
    lanes: list[LaneLine],
    drivable_space: DrivableSpaceMask,
    traffic_lights: list[TrafficLightDetection],
    cones: list[ConeDetection],
    status_summary: PerceptionStatus,
) -> None:
    bundle.metadata["perception_status"] = "ok"
    bundle.metadata["perception_detection_count"] = len(detections)
    bundle.metadata["perception_lane_count"] = len(lanes)
    bundle.metadata["perception_drivable_pixels"] = int(np.count_nonzero(drivable_space.mask))
    bundle.metadata["perception_active_cameras"] = list(status_summary.active_camera_sensors)
    bundle.metadata["perception_summary"] = status_summary


def _empty_outputs(
    bundle: SensorFrameBundle,
) -> tuple[
    list[ObjectDetection],
    list[LaneLine],
    DrivableSpaceMask,
    list[TrafficLightDetection],
    list[ConeDetection],
]:
    height, width = bundle.front_camera.frame.shape[:2]
    drivable = DrivableSpaceMask(
        mask=np.zeros((height, width), dtype=np.bool_),
        class_probabilities=np.zeros((height, width, 2), dtype=np.float32),
        source_sensor_id=bundle.front_camera.sensor_id,
    )
    return [], [], drivable, [], []


class StubPerceptionModule:
    """TODO(PRD 3.2.3): replace with model adapters and tracking."""

    def run(self, bundle: SensorFrameBundle):
        detection = ObjectDetection(
            track_id=1,
            object_class=ObjectClass.VEHICLE,
            world_bbox_3d=np.array(
                [
                    [10.0, 1.0, 0.0],
                    [11.0, 1.0, 0.0],
                    [11.0, 2.0, 0.0],
                    [10.0, 2.0, 0.0],
                    [10.0, 1.0, 1.5],
                    [11.0, 1.0, 1.5],
                    [11.0, 2.0, 1.5],
                    [10.0, 2.0, 1.5],
                ],
                dtype=np.float32,
            ),
            velocity=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            confidence=0.95,
            track_state=TrackState.CONFIRMED,
            image_bbox_xyxy=np.array([500.0, 300.0, 700.0, 600.0], dtype=np.float32),
            source_modality="bootstrap",
            source_sensor_ids=["front_camera"],
            position_estimate_kind="truth_fallback",
        )
        lane = LaneLine(
            lane_id="lane_001",
            polyline_image=np.array([[0, 5], [10, 5], [20, 5]], dtype=np.float32),
            polyline_world=np.array([[0, 50, 0], [10, 50, 0], [20, 50, 0]], dtype=np.float32),
            line_type=LaneLineType.SOLID,
            confidence=0.9,
        )
        drivable = DrivableSpaceMask(
            mask=np.ones((32, 32), dtype=np.bool_),
            class_probabilities=np.ones((32, 32, 2), dtype=np.float32) * 0.5,
            source_sensor_id=bundle.front_camera.sensor_id,
        )
        traffic_light = TrafficLightDetection(
            world_xyz=np.array([30.0, 50.0, 5.0], dtype=np.float32),
            state=TrafficLightState.GREEN,
            stop_line_distance_m=20.0,
            confidence=0.8,
            image_bbox_xyxy=np.array([800.0, 120.0, 860.0, 260.0], dtype=np.float32),
            source_modality="bootstrap",
            source_sensor_ids=["front_camera"],
            position_estimate_kind="truth_fallback",
        )
        cone = ConeDetection(
            world_xyz=np.array([25.0, 48.0, 0.0], dtype=np.float32),
            confidence=0.88,
            source_modality="bootstrap",
        )
        return [detection], [lane], drivable, [traffic_light], [cone]


class _CameraSceneContextMixin:
    def _scene_context(
        self,
        bundle: SensorFrameBundle,
        *,
        lane_extractor: LaneExtractor,
        drivable_extractor: DrivableSpaceExtractor,
    ) -> tuple[list[LaneLine], DrivableSpaceMask]:
        lanes = lane_extractor.extract(bundle.front_camera.frame)
        drivable_space = drivable_extractor.extract(
            bundle.front_camera.frame,
            bundle.front_camera.sensor_id,
        )
        return lanes, drivable_space


class PerceptionStack(_CameraSceneContextMixin):
    """Camera-first perception v1 with YOLO-primary detections and explicit bootstrap fallback."""

    def __init__(self, *, device: str, model_variant: str) -> None:
        self.detector = YoloObjectDetector(model_variant=model_variant, device=device)
        self.tracker = SimpleSortTracker()
        self.lane_extractor = LaneExtractor()
        self.drivable_extractor = DrivableSpaceExtractor()
        self.logger = get_logger(__name__, perception_mode="camera_v1")
        self._camera_order = ("front_camera", "left_camera", "right_camera", "rear_camera")

    def run(
        self, bundle: SensorFrameBundle
    ) -> tuple[
        list[ObjectDetection],
        list[LaneLine],
        DrivableSpaceMask,
        list[TrafficLightDetection],
        list[ConeDetection],
    ]:
        try:
            object_detections, traffic_lights, fallback_state, active_camera_sensors = self.detect_dynamic(bundle)
            lanes, drivable_space = self._scene_context(
                bundle,
                lane_extractor=self.lane_extractor,
                drivable_extractor=self.drivable_extractor,
            )
            status_summary = _build_perception_status(
                active_mode="camera_v1",
                fallback_state=fallback_state,
                detections=object_detections,
                cones=[],
                traffic_lights=traffic_lights,
                active_camera_sensors=active_camera_sensors,
            )
            _record_perception_metadata(
                bundle,
                detections=object_detections,
                lanes=lanes,
                drivable_space=drivable_space,
                traffic_lights=traffic_lights,
                cones=[],
                status_summary=status_summary,
            )
            bundle.metadata["perception_camera_detection_counts"] = self._camera_detection_counts(
                object_detections,
                traffic_lights,
            )
            return object_detections, lanes, drivable_space, traffic_lights, []
        except Exception as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            self.logger.warning("Perception degraded for tick %s: %s", bundle.tick_id, exc)
            return _empty_outputs(bundle)

    def detect_dynamic(
        self,
        bundle: SensorFrameBundle,
    ) -> tuple[list[ObjectDetection], list[TrafficLightDetection], str, list[str]]:
        detections_2d, detector_modes, active_camera_sensors = self._camera_detections(bundle)
        tracked_detections = self.tracker.update(detections_2d)
        object_detections, traffic_lights = self._convert_tracked(tracked_detections)
        return (
            object_detections,
            traffic_lights,
            self._camera_fallback_state(detector_modes),
            active_camera_sensors,
        )

    def _camera_detections(
        self,
        bundle: SensorFrameBundle,
    ) -> tuple[list[FrameDetection2D], dict[str, str], list[str]]:
        detections_by_camera: list[FrameDetection2D] = []
        detector_modes: dict[str, str] = {}
        active_camera_sensors: list[str] = []
        for sensor_id in self._camera_order:
            camera = getattr(bundle, sensor_id)
            if camera.status.value == "OFFLINE":
                continue
            active_camera_sensors.append(sensor_id)
            bootstrap_annotations = bootstrap_annotations_from_metadata(
                bundle.metadata,
                sensor_id=sensor_id,
            )
            detections, detector_mode = self.detector.detect(
                camera.frame,
                bootstrap_annotations,
                sensor_id=sensor_id,
                ego_world_xyz=np.asarray(bundle.gnss.world_xyz, dtype=np.float32),
                ego_yaw_rad=float(bundle.metadata.get("ego_yaw_rad", 0.0)),
            )
            detector_modes[sensor_id] = detector_mode
            detections_by_camera.extend(detections)
        return self._merge_camera_detections(detections_by_camera), detector_modes, active_camera_sensors

    def _merge_camera_detections(
        self,
        detections: list[FrameDetection2D],
    ) -> list[FrameDetection2D]:
        merged_by_track: dict[tuple[int, ObjectClass], FrameDetection2D] = {}
        passthrough: list[FrameDetection2D] = []
        for detection in detections:
            if detection.preferred_track_id is None:
                passthrough.append(detection)
                continue
            key = (int(detection.preferred_track_id), detection.object_class)
            existing = merged_by_track.get(key)
            if existing is None or self._camera_rank(detection) > self._camera_rank(existing) or (
                self._camera_rank(detection) == self._camera_rank(existing)
                and detection.confidence > existing.confidence
            ):
                merged_by_track[key] = detection
        return list(merged_by_track.values()) + passthrough

    def _camera_rank(self, detection: FrameDetection2D) -> int:
        try:
            return len(self._camera_order) - self._camera_order.index(detection.source_sensor_id)
        except ValueError:
            return 0

    def _camera_detection_counts(
        self,
        detections: list[ObjectDetection],
        traffic_lights: list[TrafficLightDetection],
    ) -> dict[str, int]:
        counts = {sensor_id: 0 for sensor_id in self._camera_order}
        for detection in detections:
            for sensor_id in detection.source_sensor_ids:
                if sensor_id in counts:
                    counts[sensor_id] += 1
        for traffic_light in traffic_lights:
            for sensor_id in traffic_light.source_sensor_ids:
                if sensor_id in counts:
                    counts[sensor_id] += 1
        return counts

    def _camera_fallback_state(self, detector_modes: dict[str, str]) -> str:
        if any(mode == "camera" for mode in detector_modes.values()):
            return "camera_only"
        return "bootstrap"

    def _convert_tracked(
        self, detections: list[TrackedDetection2D]
    ) -> tuple[list[ObjectDetection], list[TrafficLightDetection]]:
        object_detections: list[ObjectDetection] = []
        traffic_lights: list[TrafficLightDetection] = []
        for detection in detections:
            world_bbox = (
                np.asarray(detection.world_bbox_3d, dtype=np.float32)
                if detection.world_bbox_3d is not None
                else np.zeros((8, 3), dtype=np.float32)
            )
            velocity = (
                np.asarray(detection.velocity_xyz, dtype=np.float32)
                if detection.velocity_xyz is not None
                else np.zeros(3, dtype=np.float32)
            )
            world_xyz = (
                np.asarray(detection.world_xyz, dtype=np.float32)
                if detection.world_xyz is not None
                else np.mean(world_bbox, axis=0).astype(np.float32)
            )
            if detection.object_class == ObjectClass.TRAFFIC_LIGHT:
                traffic_lights.append(
                    TrafficLightDetection(
                        world_xyz=world_xyz,
                        state=detection.traffic_light_state or TrafficLightState.UNKNOWN,
                        stop_line_distance_m=max(0.0, float(world_xyz[0])),
                        confidence=float(detection.confidence),
                        image_bbox_xyxy=np.asarray(detection.bbox_xyxy, dtype=np.float32),
                        source_modality=detection.source_modality,
                        source_sensor_ids=list(detection.source_sensor_ids or [detection.source_sensor_id]),
                        position_estimate_kind=detection.position_estimate_kind,
                    )
                )
                continue
            object_detections.append(
                ObjectDetection(
                    track_id=int(detection.track_id),
                    object_class=detection.object_class,
                    world_bbox_3d=world_bbox,
                    velocity=velocity,
                    confidence=float(detection.confidence),
                    track_state=detection.track_state,
                    image_bbox_xyxy=np.asarray(detection.bbox_xyxy, dtype=np.float32),
                    source_modality=detection.source_modality,
                    source_sensor_ids=list(detection.source_sensor_ids or [detection.source_sensor_id]),
                    position_estimate_kind=detection.position_estimate_kind,
                )
            )
        return object_detections, traffic_lights


class LidarPerceptionStack(_CameraSceneContextMixin):
    """LiDAR-first perception v1 with deterministic clustering and centroid tracking."""

    def __init__(self) -> None:
        self.detector = LidarObstacleDetector()
        self.tracker = SimpleCentroidTracker3D()
        self.lane_extractor = LaneExtractor()
        self.drivable_extractor = DrivableSpaceExtractor()
        self.logger = get_logger(__name__, perception_mode="lidar_v1")

    def run(
        self, bundle: SensorFrameBundle
    ) -> tuple[
        list[ObjectDetection],
        list[LaneLine],
        DrivableSpaceMask,
        list[TrafficLightDetection],
        list[ConeDetection],
    ]:
        try:
            object_detections, cones = self.detect_dynamic(bundle)
            lanes, drivable_space = self._scene_context(
                bundle,
                lane_extractor=self.lane_extractor,
                drivable_extractor=self.drivable_extractor,
            )
            active_camera_sensors = [
                sensor_id
                for sensor_id in ("front_camera", "left_camera", "right_camera", "rear_camera")
                if getattr(bundle, sensor_id).status.value in {"OK", "DEGRADED"}
            ]
            status_summary = _build_perception_status(
                active_mode="lidar_v1",
                fallback_state="lidar_only",
                detections=object_detections,
                cones=cones,
                traffic_lights=[],
                active_camera_sensors=active_camera_sensors,
            )
            _record_perception_metadata(
                bundle,
                detections=object_detections,
                lanes=lanes,
                drivable_space=drivable_space,
                traffic_lights=[],
                cones=cones,
                status_summary=status_summary,
            )
            bundle.metadata["perception_lidar_cluster_count"] = len(object_detections)
            bundle.metadata["perception_cone_count"] = len(cones)
            return object_detections, lanes, drivable_space, [], cones
        except Exception as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            self.logger.warning("LiDAR perception degraded for tick %s: %s", bundle.tick_id, exc)
            return _empty_outputs(bundle)

    def detect_dynamic(
        self,
        bundle: SensorFrameBundle,
    ) -> tuple[list[ObjectDetection], list[ConeDetection]]:
        detections_3d, cones = self.detector.detect(bundle)
        tracked_detections = self.tracker.update(detections_3d, timestamp_s=float(bundle.sim_time_s))
        return self._convert_tracked(tracked_detections), cones

    def _convert_tracked(
        self,
        detections: list[TrackedLidarClusterDetection],
    ) -> list[ObjectDetection]:
        outputs: list[ObjectDetection] = []
        for detection in detections:
            outputs.append(
                ObjectDetection(
                    track_id=int(detection.track_id),
                    object_class=detection.object_class,
                    world_bbox_3d=np.asarray(detection.world_bbox_3d, dtype=np.float32),
                    velocity=(
                        np.asarray(detection.velocity_xyz, dtype=np.float32)
                        if detection.velocity_xyz is not None
                        else np.zeros(3, dtype=np.float32)
                    ),
                    confidence=float(detection.confidence),
                    track_state=detection.track_state,
                    image_bbox_xyxy=None,
                    source_modality=detection.source_modality,
                    source_sensor_ids=list(detection.source_sensor_ids),
                    position_estimate_kind=detection.position_estimate_kind,
                )
            )
        return outputs


class FusedPerceptionStack(_CameraSceneContextMixin):
    """Object-level camera/LiDAR fusion with camera lanes and LiDAR geometry preference."""

    def __init__(self, *, device: str, model_variant: str) -> None:
        self.camera_stack = PerceptionStack(device=device, model_variant=model_variant)
        self.lidar_stack = LidarPerceptionStack()
        self.lane_extractor = self.camera_stack.lane_extractor
        self.drivable_extractor = self.camera_stack.drivable_extractor
        self.logger = get_logger(__name__, perception_mode="fused_v1")

    def run(
        self, bundle: SensorFrameBundle
    ) -> tuple[
        list[ObjectDetection],
        list[LaneLine],
        DrivableSpaceMask,
        list[TrafficLightDetection],
        list[ConeDetection],
    ]:
        try:
            camera_detections, traffic_lights, camera_fallback_state, active_camera_sensors = (
                self.camera_stack.detect_dynamic(bundle)
            )
            lidar_detections, cones = self.lidar_stack.detect_dynamic(bundle)
            fused_objects = fuse_detections(camera_detections, lidar_detections)
            lanes, drivable_space = self._scene_context(
                bundle,
                lane_extractor=self.lane_extractor,
                drivable_extractor=self.drivable_extractor,
            )
            status_summary = _build_perception_status(
                active_mode="fused_v1",
                fallback_state=self._fused_fallback_state(
                    detections=fused_objects,
                    camera_fallback_state=camera_fallback_state,
                    cones=cones,
                ),
                detections=fused_objects,
                cones=cones,
                traffic_lights=traffic_lights,
                active_camera_sensors=active_camera_sensors,
            )
            _record_perception_metadata(
                bundle,
                detections=fused_objects,
                lanes=lanes,
                drivable_space=drivable_space,
                traffic_lights=traffic_lights,
                cones=cones,
                status_summary=status_summary,
            )
            return fused_objects, lanes, drivable_space, traffic_lights, cones
        except Exception as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            self.logger.warning("Fused perception degraded for tick %s: %s", bundle.tick_id, exc)
            return _empty_outputs(bundle)

    def _fused_fallback_state(
        self,
        *,
        detections: list[ObjectDetection],
        camera_fallback_state: str,
        cones: list[ConeDetection],
    ) -> str:
        modalities = {detection.source_modality for detection in detections}
        if "fused" in modalities:
            return "fused"
        if "lidar" in modalities or cones:
            if any(modality in {"camera", "bootstrap"} for modality in modalities):
                return "fused"
            return "lidar_only"
        if camera_fallback_state == "bootstrap":
            return "bootstrap"
        return "camera_only"


def build_perception_module(runtime_config):
    if runtime_config.perception_mode == "camera_v1":
        return PerceptionStack(
            device=runtime_config.perception_device,
            model_variant=runtime_config.perception_model_variant,
        )
    if runtime_config.perception_mode == "lidar_v1":
        return LidarPerceptionStack()
    if runtime_config.perception_mode == "fused_v1":
        return FusedPerceptionStack(
            device=runtime_config.perception_device,
            model_variant=runtime_config.perception_model_variant,
        )
    return StubPerceptionModule()
