from __future__ import annotations

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import LaneLineType, ObjectClass, TrackState, TrafficLightState
from autonomy_demo.interfaces.types import (
    ConeDetection,
    DrivableSpaceMask,
    LaneLine,
    ObjectDetection,
    SensorFrameBundle,
    TrafficLightDetection,
)
from autonomy_demo.perception.drivable_space import DrivableSpaceExtractor
from autonomy_demo.perception.internal_types import TrackedDetection2D
from autonomy_demo.perception.lane_extraction import LaneExtractor
from autonomy_demo.perception.object_detection import (
    YoloObjectDetector,
    bootstrap_annotations_from_metadata,
)
from autonomy_demo.perception.tracking import SimpleSortTracker


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
        )
        cone = ConeDetection(world_xyz=np.array([25.0, 48.0, 0.0], dtype=np.float32), confidence=0.88)
        return [detection], [lane], drivable, [traffic_light], [cone]


class PerceptionStack:
    """Camera-first perception v1 with YOLO integration point and degraded fallbacks."""

    def __init__(self, *, device: str, model_variant: str) -> None:
        self.detector = YoloObjectDetector(model_variant=model_variant, device=device)
        self.tracker = SimpleSortTracker()
        self.lane_extractor = LaneExtractor()
        self.drivable_extractor = DrivableSpaceExtractor()
        self.logger = get_logger(__name__, perception_mode="camera_v1")

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
            bootstrap_annotations = bootstrap_annotations_from_metadata(bundle.metadata)
            detections_2d = self.detector.detect(bundle.front_camera.frame, bootstrap_annotations)
            tracked_detections = self.tracker.update(detections_2d)
            lanes = self.lane_extractor.extract(bundle.front_camera.frame)
            drivable_space = self.drivable_extractor.extract(
                bundle.front_camera.frame,
                bundle.front_camera.sensor_id,
            )
            object_detections, traffic_lights = self._convert_tracked(tracked_detections)
            bundle.metadata["perception_status"] = "ok"
            bundle.metadata["perception_detection_count"] = len(object_detections)
            bundle.metadata["perception_lane_count"] = len(lanes)
            bundle.metadata["perception_drivable_pixels"] = int(np.count_nonzero(drivable_space.mask))
            return object_detections, lanes, drivable_space, traffic_lights, []
        except Exception as exc:
            bundle.metadata["perception_status"] = "degraded"
            bundle.metadata["perception_error"] = str(exc)
            self.logger.warning("Perception degraded for tick %s: %s", bundle.tick_id, exc)
            return self._empty_outputs(bundle)

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
                )
            )
        return object_detections, traffic_lights

    def _empty_outputs(
        self, bundle: SensorFrameBundle
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


def build_perception_module(runtime_config):
    if runtime_config.perception_mode == "camera_v1":
        return PerceptionStack(
            device=runtime_config.perception_device,
            model_variant=runtime_config.perception_model_variant,
        )
    return StubPerceptionModule()
