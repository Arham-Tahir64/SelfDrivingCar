from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.enums import TrackState
from autonomy_demo.interfaces.types import ObjectDetection


def _bbox_center(detection: ObjectDetection) -> np.ndarray:
    return np.mean(np.asarray(detection.world_bbox_3d, dtype=np.float32), axis=0)


def _class_compatible(camera_detection: ObjectDetection, lidar_detection: ObjectDetection) -> bool:
    if camera_detection.object_class == lidar_detection.object_class:
        return True
    soft_pair = {"pedestrian", "cyclist"}
    return (
        camera_detection.object_class.value in soft_pair
        and lidar_detection.object_class.value in soft_pair
    )


def fuse_detections(
    camera_detections: list[ObjectDetection],
    lidar_detections: list[ObjectDetection],
    *,
    match_distance_m: float = 5.0,
    camera_class_confidence_threshold: float = 0.75,
) -> list[ObjectDetection]:
    fused: list[ObjectDetection] = []
    used_camera_indices: set[int] = set()
    used_lidar_indices: set[int] = set()
    candidate_pairs: list[tuple[float, int, int]] = []

    for camera_index, camera_detection in enumerate(camera_detections):
        camera_center = _bbox_center(camera_detection)
        for lidar_index, lidar_detection in enumerate(lidar_detections):
            if not _class_compatible(camera_detection, lidar_detection):
                continue
            lidar_center = _bbox_center(lidar_detection)
            distance_m = float(np.linalg.norm(camera_center[:2] - lidar_center[:2]))
            if distance_m <= match_distance_m:
                candidate_pairs.append((distance_m, camera_index, lidar_index))

    candidate_pairs.sort(key=lambda item: item[0])
    for _, camera_index, lidar_index in candidate_pairs:
        if camera_index in used_camera_indices or lidar_index in used_lidar_indices:
            continue
        camera_detection = camera_detections[camera_index]
        lidar_detection = lidar_detections[lidar_index]
        preferred_class = (
            camera_detection.object_class
            if camera_detection.confidence >= camera_class_confidence_threshold
            else lidar_detection.object_class
        )
        preferred_velocity = (
            lidar_detection.velocity
            if float(np.linalg.norm(np.asarray(lidar_detection.velocity, dtype=np.float32))) > 0.1
            else camera_detection.velocity
        )
        fused.append(
            ObjectDetection(
                track_id=int(lidar_detection.track_id),
                object_class=preferred_class,
                world_bbox_3d=np.asarray(lidar_detection.world_bbox_3d, dtype=np.float32),
                velocity=np.asarray(preferred_velocity, dtype=np.float32),
                confidence=float(max(camera_detection.confidence, lidar_detection.confidence)),
                track_state=(
                    TrackState.CONFIRMED
                    if TrackState.CONFIRMED in {camera_detection.track_state, lidar_detection.track_state}
                    else TrackState.TENTATIVE
                ),
                image_bbox_xyxy=(
                    None
                    if camera_detection.image_bbox_xyxy is None
                    else np.asarray(camera_detection.image_bbox_xyxy, dtype=np.float32)
                ),
                source_modality="fused",
                source_sensor_ids=sorted(
                    set(camera_detection.source_sensor_ids + lidar_detection.source_sensor_ids)
                ),
                position_estimate_kind="fusion",
            )
        )
        used_camera_indices.add(camera_index)
        used_lidar_indices.add(lidar_index)

    fused.extend(
        detection for index, detection in enumerate(camera_detections) if index not in used_camera_indices
    )
    fused.extend(
        detection for index, detection in enumerate(lidar_detections) if index not in used_lidar_indices
    )
    return fused
