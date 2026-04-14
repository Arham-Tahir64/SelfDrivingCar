from __future__ import annotations

import asyncio
import base64
import json
import math
import threading
import time
from typing import Any

import numpy as np

from autonomy_demo.common.logging import get_logger
from autonomy_demo.common.serialization import serialize
from autonomy_demo.interfaces.enums import TopicName

logger = get_logger(__name__)

_WS_TOPICS = frozenset(
    {
        TopicName.LOCALIZATION_EGO_POSE.value,
        TopicName.PERCEPTION_DETECTIONS.value,
        TopicName.PERCEPTION_LANES.value,
        TopicName.PERCEPTION_TRAFFIC_LIGHTS.value,
        TopicName.PERCEPTION_STATUS.value,
        TopicName.MAP_LOCAL_MAP.value,
        TopicName.PREDICTION_AGENTS.value,
        TopicName.PLANNING_EGO_TRAJECTORY.value,
        TopicName.PLANNING_CANDIDATES.value,
        TopicName.CONTROL_VEHICLE_COMMAND.value,
        TopicName.SCENARIO_INFO.value,
        TopicName.VISUALIZATION_CAMERA_OVERLAY.value,
        TopicName.VISUALIZATION_BEV_DRIVABLE.value,
        TopicName.VISUALIZATION_ROAD_CORRIDOR.value,
        TopicName.VISUALIZATION_WORLD_LAYER.value,
        TopicName.VISUALIZATION_PRIOR_MAP.value,
        TopicName.SENSOR_LIDAR.value,
        TopicName.PIPELINE_LATENCY.value,
        TopicName.PERCEPTION_DEPTH.value,
    }
)

_SKIP_TOPICS = frozenset(
    {
        TopicName.SENSOR_CAMERA_FRONT.value,
        TopicName.SENSOR_LIDAR.value,
        TopicName.PERCEPTION_DRIVABLE_SPACE.value,
        TopicName.VISUALIZATION_BEV_DRIVABLE.value,
    }
)

_RETAINED_TOPICS = frozenset(
    {
        TopicName.SCENARIO_INFO.value,
        TopicName.MAP_LOCAL_MAP.value,
        TopicName.VISUALIZATION_WORLD_LAYER.value,
        TopicName.VISUALIZATION_PRIOR_MAP.value,
    }
)

_FAST_DYNAMIC_TOPICS = frozenset(
    {
        TopicName.LOCALIZATION_EGO_POSE.value,
        TopicName.PERCEPTION_DETECTIONS.value,
        TopicName.PERCEPTION_TRAFFIC_LIGHTS.value,
        TopicName.PERCEPTION_STATUS.value,
        TopicName.CONTROL_VEHICLE_COMMAND.value,
        TopicName.PLANNING_EGO_TRAJECTORY.value,
        TopicName.PIPELINE_LATENCY.value,
        TopicName.VISUALIZATION_ROAD_CORRIDOR.value,
        TopicName.VISUALIZATION_WORLD_LAYER.value,
    }
)

_HEAVY_DYNAMIC_TOPICS = frozenset(
    {
        TopicName.VISUALIZATION_CAMERA_OVERLAY.value,
        TopicName.VISUALIZATION_LIDAR_PREVIEW.value,
        TopicName.VISUALIZATION_BEV_DRIVABLE.value,
        TopicName.PREDICTION_AGENTS.value,
        TopicName.PLANNING_CANDIDATES.value,
        TopicName.PERCEPTION_DEPTH.value,
    }
)

_WORLD_LAYER_STATIC_KEYS = ("signature", "roads", "lane_markers", "sidewalks")
_WORLD_LAYER_DYNAMIC_KEYS = ("traffic_lights",)

_LIDAR_MAX_POINTS = 1500
_JPEG_QUALITY = 80
_OVERLAY_MAX_WIDTH = 960
_FAST_DYNAMIC_FPS = 8.0
_HEAVY_DYNAMIC_FPS = 6.0
_WAYPOINT_STRIDE = 2
_WS_STATS_LOG_INTERVAL_S = 5.0
_LIDAR_PANEL_RANGE_M = 50.0
_LIDAR_PANEL_REAR_RANGE_M = 8.0
_LIDAR_PANEL_LATERAL_LIMIT_M = _LIDAR_PANEL_RANGE_M
_LIDAR_FORWARD_CONE_LENGTH_M = 28.0
_LIDAR_FORWARD_CONE_HALF_ANGLE_DEG = 18.0
_LIDAR_THREAT_COUNT = 3
_LIDAR_PRIORITY_POINT_RATIO = 0.65
_LIDAR_PRIORITY_FORWARD_M = 35.0
_LIDAR_PRIORITY_LATERAL_M = 12.0


def _stable_json(value: Any) -> str:
    return json.dumps(serialize(value), default=str, sort_keys=True, separators=(",", ":"))


def _json_size_bytes(value: Any) -> int:
    return len(_stable_json(value).encode("utf-8"))


def _hash_payload(value: Any) -> str:
    return _stable_json(value)


def _merge_topic(topics: dict[str, Any], topic: str, payload: Any) -> None:
    if payload is None:
        return
    existing = topics.get(topic)
    if isinstance(existing, dict) and isinstance(payload, dict):
        topics[topic] = {**existing, **payload}
        return
    topics[topic] = payload


def _serialize_payload(payload: Any) -> Any:
    return serialize({"payload": payload}).get("payload")


def _field(payload: Any, name: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(name, default)
    return getattr(payload, name, default)


def _encode_overlay_jpeg(frame_rgb: np.ndarray) -> str | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    height, width = frame_rgb.shape[:2]
    if width > _OVERLAY_MAX_WIDTH:
        scale = _OVERLAY_MAX_WIDTH / width
        frame_rgb = cv2.resize(
            frame_rgb,
            (_OVERLAY_MAX_WIDTH, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    success, buf = cv2.imencode(".jpg", frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not success:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_depth_jpeg(depth_map) -> str | None:
    """Colorize a DepthMap with Turbo colormap and encode as JPEG base64."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    depth = np.asarray(depth_map.depth, dtype=np.float32)
    depth_u8 = np.clip(depth * 255, 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    height, width = colored_bgr.shape[:2]
    max_width = _OVERLAY_MAX_WIDTH
    if width > max_width:
        scale = max_width / width
        colored_bgr = cv2.resize(
            colored_bgr,
            (max_width, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    success, buf = cv2.imencode(".jpg", colored_bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    if not success:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _lidar_points_in_panel_roi(points_xyz: np.ndarray) -> np.ndarray:
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        return np.empty((0, 3), dtype=np.float32)
    mask = (
        (points_xyz[:, 0] >= -_LIDAR_PANEL_REAR_RANGE_M)
        & (points_xyz[:, 0] <= _LIDAR_PANEL_RANGE_M)
        & (np.abs(points_xyz[:, 1]) <= _LIDAR_PANEL_LATERAL_LIMIT_M)
    )
    return points_xyz[mask]


def _evenly_sample_points(points_xyz: np.ndarray, sample_count: int) -> np.ndarray:
    count = int(points_xyz.shape[0])
    if count == 0 or sample_count <= 0:
        return np.empty((0, points_xyz.shape[1] if points_xyz.ndim == 2 else 3), dtype=np.float32)
    if count <= sample_count:
        return points_xyz
    indices = np.linspace(0, count - 1, sample_count, dtype=np.int32)
    return points_xyz[indices]


def _downsample_lidar(points_xyz: np.ndarray) -> list[list[float]]:
    roi_points = _lidar_points_in_panel_roi(np.asarray(points_xyz, dtype=np.float32))
    count = int(roi_points.shape[0])
    if count == 0:
        return []
    if count <= _LIDAR_MAX_POINTS:
        return roi_points.tolist()

    priority_mask = (
        (roi_points[:, 0] >= 0.0)
        & (roi_points[:, 0] <= _LIDAR_PRIORITY_FORWARD_M)
        & (np.abs(roi_points[:, 1]) <= _LIDAR_PRIORITY_LATERAL_M)
    )
    priority_points = roi_points[priority_mask]
    secondary_points = roi_points[~priority_mask]

    priority_budget = min(
        int(math.ceil(_LIDAR_MAX_POINTS * _LIDAR_PRIORITY_POINT_RATIO)),
        int(priority_points.shape[0]),
    )
    secondary_budget = _LIDAR_MAX_POINTS - priority_budget
    secondary_sample_count = min(secondary_budget, int(secondary_points.shape[0]))
    priority_sample_count = min(int(priority_points.shape[0]), _LIDAR_MAX_POINTS - secondary_sample_count)

    sampled_priority = _evenly_sample_points(priority_points, priority_sample_count)
    sampled_secondary = _evenly_sample_points(secondary_points, secondary_sample_count)
    sampled_points = np.vstack([sampled_priority, sampled_secondary])

    if int(sampled_points.shape[0]) > _LIDAR_MAX_POINTS:
        sampled_points = sampled_points[:_LIDAR_MAX_POINTS]
    return sampled_points.tolist()


def _world_to_ego_xy(world_xyz: np.ndarray | list[float], ego_pose: Any) -> np.ndarray | None:
    if ego_pose is None:
        return None
    ego_xyz = np.asarray(getattr(ego_pose, "world_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
    yaw = float(getattr(ego_pose, "yaw_rad", 0.0))
    delta_xy = np.asarray(world_xyz, dtype=np.float32)[:2] - ego_xyz[:2]
    rotation = np.array(
        [
            [math.cos(yaw), math.sin(yaw)],
            [-math.sin(yaw), math.cos(yaw)],
        ],
        dtype=np.float32,
    )
    return rotation @ delta_xy


def _distance_point_to_segment(point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> float:
    segment = end_xy - start_xy
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-6:
        return float(np.linalg.norm(point_xy - start_xy))
    t = float(np.clip(np.dot(point_xy - start_xy, segment) / length_sq, 0.0, 1.0))
    projection = start_xy + (segment * t)
    return float(np.linalg.norm(point_xy - projection))


def _distance_to_polyline(point_xy: np.ndarray, polyline_xy: list[list[float]]) -> float:
    if len(polyline_xy) < 2:
        return abs(float(point_xy[1]))
    best = float("inf")
    for start, end in zip(polyline_xy[:-1], polyline_xy[1:]):
        best = min(
            best,
            _distance_point_to_segment(
                point_xy,
                np.asarray(start, dtype=np.float32),
                np.asarray(end, dtype=np.float32),
            ),
        )
    return best


def _centroid_xy_from_bbox(world_bbox_3d: Any) -> np.ndarray:
    bbox = np.asarray(world_bbox_3d, dtype=np.float32)
    if bbox.size == 0:
        return np.zeros(2, dtype=np.float32)
    return np.mean(bbox[:, :2], axis=0).astype(np.float32)


def _ego_frame_footprint(world_bbox_3d: Any, ego_pose: Any) -> list[list[float]]:
    bbox = np.asarray(world_bbox_3d, dtype=np.float32)
    if bbox.shape[0] < 4:
        return []
    footprint: list[list[float]] = []
    for corner in bbox[:4]:
        ego_xy = _world_to_ego_xy(corner[:2], ego_pose)
        if ego_xy is None:
            continue
        footprint.append([float(ego_xy[0]), float(ego_xy[1])])
    return footprint


def _ego_frame_velocity_xy(world_velocity: Any, ego_pose: Any) -> list[float]:
    if ego_pose is None:
        return [0.0, 0.0]
    velocity = np.asarray(world_velocity, dtype=np.float32)
    if velocity.size < 2:
        return [0.0, 0.0]
    yaw = float(getattr(ego_pose, "yaw_rad", 0.0))
    rotation = np.array(
        [
            [math.cos(yaw), math.sin(yaw)],
            [-math.sin(yaw), math.cos(yaw)],
        ],
        dtype=np.float32,
    )
    ego_velocity = rotation @ velocity[:2]
    return [float(ego_velocity[0]), float(ego_velocity[1])]


def _ego_path_polyline(trajectory: Any, ego_pose: Any) -> list[list[float]]:
    if trajectory is None or ego_pose is None:
        return []
    polyline: list[list[float]] = []
    for waypoint in _sample_waypoints(list(getattr(trajectory, "waypoints", []) or [])):
        ego_xy = _world_to_ego_xy([getattr(waypoint, "x", 0.0), getattr(waypoint, "y", 0.0)], ego_pose)
        if ego_xy is None:
            continue
        if float(ego_xy[0]) < -8.0 or abs(float(ego_xy[1])) > (_LIDAR_PANEL_RANGE_M * 0.75):
            continue
        polyline.append([float(ego_xy[0]), float(ego_xy[1])])
    return polyline


def _prediction_ghost_xy(prediction: Any, ego_pose: Any) -> list[float] | None:
    if prediction is None or ego_pose is None:
        return None
    waypoints = list(getattr(prediction, "predicted_trajectory", []) or [])
    if not waypoints:
        return None
    target_waypoint = waypoints[-1]
    for waypoint in waypoints:
        if float(getattr(waypoint, "timestamp", 0.0)) >= 0.6:
            target_waypoint = waypoint
            break
    ego_xy = _world_to_ego_xy(
        [getattr(target_waypoint, "x", 0.0), getattr(target_waypoint, "y", 0.0)],
        ego_pose,
    )
    if ego_xy is None:
        return None
    return [float(ego_xy[0]), float(ego_xy[1])]


def _lidar_panel_status(perception_status: Any, object_count: int, confirmed_count: int, point_count: int) -> dict[str, Any]:
    active_mode = str(_field(perception_status, "active_mode", "unknown"))
    fallback_state = str(_field(perception_status, "fallback_state", "unknown"))
    degraded = fallback_state in {"camera_only", "bootstrap"}
    return {
        "mode": active_mode,
        "degraded": degraded,
        "lidar_track_count": int(object_count),
        "confirmed_track_count": int(confirmed_count),
        "point_count": int(point_count),
    }


def _serialize_lidar_preview(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    lidar_frame = snapshot.get(TopicName.SENSOR_LIDAR.value)
    if lidar_frame is None or not hasattr(lidar_frame, "points_xyz"):
        return None

    ego_pose = snapshot.get(TopicName.LOCALIZATION_EGO_POSE.value)
    detections = list(snapshot.get(TopicName.PERCEPTION_DETECTIONS.value) or [])
    predictions = list(snapshot.get(TopicName.PREDICTION_AGENTS.value) or [])
    prediction_by_track = {
        int(getattr(prediction, "track_id", -1)): prediction
        for prediction in predictions
    }
    path_polyline_xy = _ego_path_polyline(snapshot.get(TopicName.PLANNING_EGO_TRAJECTORY.value), ego_pose)

    candidates: list[dict[str, Any]] = []
    for detection in detections:
        source_modality = str(getattr(detection, "source_modality", ""))
        if source_modality not in {"lidar", "fused"}:
            continue
        track_state = str(getattr(getattr(detection, "track_state", None), "value", getattr(detection, "track_state", "")))
        if track_state in {"LOST", "DELETED"}:
            continue
        footprint_xy = _ego_frame_footprint(getattr(detection, "world_bbox_3d", None), ego_pose)
        if len(footprint_xy) < 4:
            continue
        centroid_world_xy = _centroid_xy_from_bbox(getattr(detection, "world_bbox_3d", None))
        centroid_ego_xy = _world_to_ego_xy([float(centroid_world_xy[0]), float(centroid_world_xy[1])], ego_pose)
        if centroid_ego_xy is None:
            continue
        forward_m = float(centroid_ego_xy[0])
        lateral_m = float(centroid_ego_xy[1])
        if forward_m < -10.0 or forward_m > (_LIDAR_PANEL_RANGE_M + 8.0) or abs(lateral_m) > (_LIDAR_PANEL_RANGE_M * 0.75):
            continue
        velocity_xy = _ego_frame_velocity_xy(getattr(detection, "velocity", [0.0, 0.0, 0.0]), ego_pose)
        speed_mps = float(np.linalg.norm(np.asarray(velocity_xy, dtype=np.float32)))
        path_distance_m = _distance_to_polyline(np.asarray([forward_m, lateral_m], dtype=np.float32), path_polyline_xy)
        is_path_relevant = path_distance_m <= 3.0 and forward_m >= -3.0
        forward_term = max(0.0, 1.0 - min(max(forward_m, 0.0), 45.0) / 45.0)
        lateral_term = max(0.0, 1.0 - min(abs(lateral_m), 12.0) / 12.0)
        path_term = max(0.0, 1.0 - min(path_distance_m, 4.0) / 4.0)
        speed_term = min(speed_mps, 12.0) / 12.0
        rear_penalty = 0.35 if forward_m < 0.0 else 0.0
        relevance_score = max(
            0.0,
            (0.42 * forward_term) + (0.38 * path_term) + (0.12 * lateral_term) + (0.08 * speed_term) - rear_penalty,
        )
        candidates.append(
            {
                "track_id": int(getattr(detection, "track_id", -1)),
                "track_state": track_state,
                "object_class": getattr(getattr(detection, "object_class", None), "value", getattr(detection, "object_class", "")),
                "confidence": float(getattr(detection, "confidence", 0.0)),
                "source_modality": source_modality,
                "footprint_xy": footprint_xy,
                "centroid_xy": [forward_m, lateral_m],
                "velocity_xy": velocity_xy,
                "speed_mps": speed_mps,
                "relevance_score": relevance_score,
                "is_path_relevant": is_path_relevant,
                "_prediction": prediction_by_track.get(int(getattr(detection, "track_id", -1))),
            }
        )

    confirmed = [candidate for candidate in candidates if candidate["track_state"] == "CONFIRMED"]
    ranked_confirmed = sorted(
        confirmed,
        key=lambda candidate: (-float(candidate["relevance_score"]), float(candidate["centroid_xy"][0] < 0.0), abs(float(candidate["centroid_xy"][1]))),
    )
    threat_ids = [int(candidate["track_id"]) for candidate in ranked_confirmed[:_LIDAR_THREAT_COUNT] if candidate["relevance_score"] > 0.22]
    threat_index = {track_id: rank for rank, track_id in enumerate(threat_ids, start=1)}

    objects: list[dict[str, Any]] = []
    for candidate in candidates:
        track_id = int(candidate["track_id"])
        if track_id in threat_index:
            ghost_xy = _prediction_ghost_xy(candidate["_prediction"], ego_pose)
        else:
            ghost_xy = None
        objects.append(
            {
                "track_id": track_id,
                "track_state": candidate["track_state"],
                "object_class": candidate["object_class"],
                "confidence": candidate["confidence"],
                "source_modality": candidate["source_modality"],
                "footprint_xy": candidate["footprint_xy"],
                "centroid_xy": candidate["centroid_xy"],
                "velocity_xy": candidate["velocity_xy"],
                "speed_mps": candidate["speed_mps"],
                "relevance_score": candidate["relevance_score"],
                "threat_rank": int(threat_index.get(track_id, 0)),
                "is_path_relevant": bool(candidate["is_path_relevant"]),
                "ghost_xy": ghost_xy,
            }
        )

    point_sample = _downsample_lidar(np.asarray(lidar_frame.points_xyz, dtype=np.float32))
    perception_status = snapshot.get(TopicName.PERCEPTION_STATUS.value)
    return {
        "points": point_sample,
        "objects": objects,
        "threat_ids": threat_ids,
        "path_polyline_xy": path_polyline_xy,
        "forward_cone": {
            "length_m": _LIDAR_FORWARD_CONE_LENGTH_M,
            "half_angle_deg": _LIDAR_FORWARD_CONE_HALF_ANGLE_DEG,
        },
        "status": _lidar_panel_status(
            perception_status,
            object_count=len(objects),
            confirmed_count=len(confirmed),
            point_count=len(point_sample),
        ),
    }


def _serialize_local_map_for_dashboard(local_map: Any) -> dict[str, Any] | None:
    if local_map is None:
        return None
    return serialize(
        {
        "closed_lanes": list(getattr(local_map, "closed_lanes", []) or []),
        "temporary_boundaries": list(getattr(local_map, "temporary_boundaries", []) or []),
        }
    )


def _sample_waypoints(waypoints: list[Any] | tuple[Any, ...] | None) -> list[Any]:
    if not waypoints:
        return []
    sampled = list(waypoints[::_WAYPOINT_STRIDE])
    if sampled and sampled[-1] is not waypoints[-1]:
        sampled.append(waypoints[-1])
    return sampled


def _serialize_ego_trajectory_for_dashboard(trajectory: Any) -> dict[str, Any] | None:
    if trajectory is None:
        return None
    return serialize(
        {
            "waypoints": _sample_waypoints(list(getattr(trajectory, "waypoints", []) or [])),
            "cost": getattr(trajectory, "cost", 0.0),
            "behavior_state": getattr(
                getattr(trajectory, "behavior_state", None),
                "value",
                getattr(trajectory, "behavior_state", ""),
            ),
        }
    )


def _serialize_candidates_for_dashboard(candidates: Any) -> list[dict[str, Any]]:
    if not candidates:
        return []
    serialized: list[dict[str, Any]] = []
    for candidate in list(candidates):
        cost_breakdown = getattr(candidate, "cost_breakdown", None)
        serialized.append(
            {
                "trajectory": _serialize_ego_trajectory_for_dashboard(getattr(candidate, "trajectory", None)),
                "lane_id": getattr(candidate, "lane_id", ""),
                "target_speed_mps": getattr(candidate, "target_speed_mps", 0.0),
                "score": getattr(candidate, "score", 0.0),
                "feasible": getattr(candidate, "feasible", True),
                "reject_reason": getattr(candidate, "reject_reason", None),
                "reference_lane_id": getattr(candidate, "reference_lane_id", ""),
                "target_lane_id": getattr(candidate, "target_lane_id", ""),
                "target_d_m": getattr(candidate, "target_d_m", 0.0),
                "terminal_time_s": getattr(candidate, "terminal_time_s", 0.0),
                "cost_breakdown": {
                    "collision": getattr(cost_breakdown, "collision", 0.0),
                    "cone_proximity": getattr(cost_breakdown, "cone_proximity", 0.0),
                    "lane_deviation": getattr(cost_breakdown, "lane_deviation", 0.0),
                    "jerk": getattr(cost_breakdown, "jerk", 0.0),
                    "speed_error": getattr(cost_breakdown, "speed_error", 0.0),
                    "traffic_violation": getattr(cost_breakdown, "traffic_violation", 0.0),
                    "route_progress": getattr(cost_breakdown, "route_progress", 0.0),
                    "total": getattr(cost_breakdown, "total", getattr(candidate, "score", 0.0)),
                },
            }
        )
    return serialize(serialized)


def _serialize_predictions_for_dashboard(predictions: Any) -> list[dict[str, Any]]:
    if not predictions:
        return []
    serialized: list[dict[str, Any]] = []
    for prediction in list(predictions):
        serialized.append(
            {
                "track_id": getattr(prediction, "track_id", -1),
                "object_class": getattr(getattr(prediction, "object_class", None), "value", getattr(prediction, "object_class", "")),
                "predicted_trajectory": _sample_waypoints(list(getattr(prediction, "predicted_trajectory", []) or [])),
                "confidence_by_step": list(getattr(prediction, "confidence_by_step", []) or [])[::_WAYPOINT_STRIDE],
                "covariance_by_step": list(getattr(prediction, "covariance_by_step", []) or [])[::_WAYPOINT_STRIDE],
            }
        )
    return serialize(serialized)


def _serialize_bev_grid(bev_grid: Any) -> dict[str, Any] | None:
    if not isinstance(bev_grid, dict) or not isinstance(bev_grid.get("grid"), np.ndarray):
        return None
    grid = np.asarray(bev_grid["grid"], dtype=np.uint8)
    return {
        "grid_b64": base64.b64encode(grid.tobytes()).decode("ascii"),
        "rows": int(bev_grid["rows"]),
        "cols": int(bev_grid["cols"]),
        "cell_size_m": float(bev_grid["cell_size_m"]),
        "x_min_m": float(bev_grid["x_min_m"]),
        "x_max_m": float(bev_grid["x_max_m"]),
        "y_min_m": float(bev_grid["y_min_m"]),
        "y_max_m": float(bev_grid["y_max_m"]),
    }


def _split_world_layer(world_layer: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(world_layer, dict):
        return None, None
    static_payload = {
        key: world_layer[key]
        for key in _WORLD_LAYER_STATIC_KEYS
        if key in world_layer
    }
    dynamic_payload = {
        key: world_layer[key]
        for key in _WORLD_LAYER_DYNAMIC_KEYS
        if key in world_layer
    }
    return static_payload or None, dynamic_payload or None


class WebSocketBridge:
    """Streams pipeline state over WebSocket to browser clients."""

    def __init__(self) -> None:
        self._event_bus: Any | None = None
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_bootstrap_text: str | None = None
        self._pending_static_text: str | None = None
        self._pending_dynamic_text: str | None = None
        self._send_active = False
        self._send_lock = threading.Lock()
        self._bootstrap_needed = False
        self._last_dynamic_monotonic = 0.0
        self._retained_hashes: dict[str, str] = {}
        self._retained_payloads: dict[str, Any] = {}
        self._heavy_hashes: dict[str, str] = {}
        self._heavy_last_sent_monotonic: dict[str, float] = {}
        self._stats_last_log_monotonic = 0.0

    def attach(self, event_bus: Any) -> None:
        self._event_bus = event_bus
        event_bus.subscribe(TopicName.TICK_COMPLETE.value, self._on_tick_complete)
        logger.info("WebSocketBridge attached to event bus")

    def flush(self) -> None:
        logger.info("WebSocketBridge flush â€” %d clients connected", len(self._clients))

    def register(self, ws: Any) -> None:
        self._clients.add(ws)
        with self._send_lock:
            self._bootstrap_needed = True
        logger.info("WebSocket client connected (%d total)", len(self._clients))
        self._queue_bootstrap_from_latest_state()

    def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self._clients))

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def _snapshot(self) -> dict[str, Any]:
        if self._event_bus is None:
            return {}
        if hasattr(self._event_bus, "latest_topics"):
            return self._event_bus.latest_topics(_WS_TOPICS)
        return self._event_bus.snapshot()

    def _latest_tick_payload(self) -> dict[str, Any]:
        if self._event_bus is None:
            return {}
        if hasattr(self._event_bus, "latest_topics"):
            latest = self._event_bus.latest_topics({TopicName.TICK_COMPLETE.value})
            payload = latest.get(TopicName.TICK_COMPLETE.value)
            return payload if isinstance(payload, dict) else {}
        snapshot = self._event_bus.snapshot()
        payload = snapshot.get(TopicName.TICK_COMPLETE.value)
        return payload if isinstance(payload, dict) else {}

    def _prepare_snapshot_topics(self, snapshot: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        retained: dict[str, Any] = {}
        dynamic: dict[str, Any] = {}
        heavy: dict[str, Any] = {}

        scenario_info = snapshot.get(TopicName.SCENARIO_INFO.value)
        if scenario_info is not None:
            retained[TopicName.SCENARIO_INFO.value] = _serialize_payload(scenario_info)

        local_map = snapshot.get(TopicName.MAP_LOCAL_MAP.value)
        serialized_map = _serialize_local_map_for_dashboard(local_map)
        if serialized_map is not None:
            retained[TopicName.MAP_LOCAL_MAP.value] = serialized_map

        world_layer = snapshot.get(TopicName.VISUALIZATION_WORLD_LAYER.value)
        static_world_layer, dynamic_world_layer = _split_world_layer(world_layer)
        if static_world_layer is not None:
            retained[TopicName.VISUALIZATION_WORLD_LAYER.value] = static_world_layer
        if dynamic_world_layer is not None:
            dynamic[TopicName.VISUALIZATION_WORLD_LAYER.value] = dynamic_world_layer

        prior_map = snapshot.get(TopicName.VISUALIZATION_PRIOR_MAP.value)
        if prior_map is not None:
            retained[TopicName.VISUALIZATION_PRIOR_MAP.value] = _serialize_payload(prior_map)

        road_corridor = snapshot.get(TopicName.VISUALIZATION_ROAD_CORRIDOR.value)
        if road_corridor is not None:
            dynamic[TopicName.VISUALIZATION_ROAD_CORRIDOR.value] = _serialize_payload(road_corridor)

        for topic in (
            TopicName.LOCALIZATION_EGO_POSE.value,
            TopicName.PERCEPTION_DETECTIONS.value,
            TopicName.PERCEPTION_LANES.value,
            TopicName.PERCEPTION_TRAFFIC_LIGHTS.value,
            TopicName.PERCEPTION_STATUS.value,
            TopicName.CONTROL_VEHICLE_COMMAND.value,
            TopicName.PIPELINE_LATENCY.value,
        ):
            payload = snapshot.get(topic)
            if payload is not None:
                dynamic[topic] = _serialize_payload(payload)

        trajectory = snapshot.get(TopicName.PLANNING_EGO_TRAJECTORY.value)
        serialized_trajectory = _serialize_ego_trajectory_for_dashboard(trajectory)
        if serialized_trajectory is not None:
            dynamic[TopicName.PLANNING_EGO_TRAJECTORY.value] = serialized_trajectory

        overlay_frame = snapshot.get(TopicName.VISUALIZATION_CAMERA_OVERLAY.value)
        if isinstance(overlay_frame, np.ndarray):
            jpeg_b64 = _encode_overlay_jpeg(overlay_frame)
            if jpeg_b64:
                heavy[TopicName.VISUALIZATION_CAMERA_OVERLAY.value] = jpeg_b64

        lidar_preview = _serialize_lidar_preview(snapshot)
        if lidar_preview is not None:
            heavy[TopicName.VISUALIZATION_LIDAR_PREVIEW.value] = lidar_preview

        bev_grid = _serialize_bev_grid(snapshot.get(TopicName.VISUALIZATION_BEV_DRIVABLE.value))
        if bev_grid is not None:
            heavy[TopicName.VISUALIZATION_BEV_DRIVABLE.value] = bev_grid

        predictions = snapshot.get(TopicName.PREDICTION_AGENTS.value)
        if predictions is not None:
            heavy[TopicName.PREDICTION_AGENTS.value] = _serialize_predictions_for_dashboard(predictions)

        candidates = snapshot.get(TopicName.PLANNING_CANDIDATES.value)
        if candidates is not None:
            heavy[TopicName.PLANNING_CANDIDATES.value] = _serialize_candidates_for_dashboard(candidates)

        depth_map = snapshot.get(TopicName.PERCEPTION_DEPTH.value)
        if depth_map is not None:
            depth_b64 = _encode_depth_jpeg(depth_map)
            if depth_b64:
                heavy[TopicName.PERCEPTION_DEPTH.value] = depth_b64

        return retained, dynamic, heavy

    def _update_retained_cache(self, retained_topics: dict[str, Any]) -> dict[str, Any]:
        changed: dict[str, Any] = {}
        for topic, payload in retained_topics.items():
            payload_hash = _hash_payload(payload)
            if self._retained_hashes.get(topic) != payload_hash:
                changed[topic] = payload
                self._retained_hashes[topic] = payload_hash
            self._retained_payloads[topic] = payload
        return changed

    def _collect_heavy_updates(self, heavy_topics: dict[str, Any], now_monotonic: float) -> dict[str, Any]:
        changed: dict[str, Any] = {}
        min_interval_s = 1.0 / _HEAVY_DYNAMIC_FPS
        for topic, payload in heavy_topics.items():
            payload_hash = _hash_payload(payload)
            last_sent = self._heavy_last_sent_monotonic.get(topic, 0.0)
            if topic in {
                TopicName.PREDICTION_AGENTS.value,
                TopicName.PLANNING_CANDIDATES.value,
            }:
                should_send = payload_hash != self._heavy_hashes.get(topic) or (now_monotonic - last_sent) >= min_interval_s
            else:
                should_send = payload_hash != self._heavy_hashes.get(topic) and (
                    last_sent <= 0.0 or (now_monotonic - last_sent) >= min_interval_s
                )
            if not should_send:
                continue
            changed[topic] = payload
            self._heavy_hashes[topic] = payload_hash
            self._heavy_last_sent_monotonic[topic] = now_monotonic
        return changed

    def _mark_bootstrap_sent(self, retained_topics: dict[str, Any], heavy_topics: dict[str, Any], now_monotonic: float) -> None:
        for topic, payload in retained_topics.items():
            self._retained_hashes[topic] = _hash_payload(payload)
            self._retained_payloads[topic] = payload
        for topic, payload in heavy_topics.items():
            self._heavy_hashes[topic] = _hash_payload(payload)
            self._heavy_last_sent_monotonic[topic] = now_monotonic

    def _queue_bootstrap_from_latest_state(self) -> None:
        if self._event_bus is None:
            return
        now_monotonic = time.monotonic()
        snapshot = self._snapshot()
        retained_topics, dynamic_topics, heavy_topics = self._prepare_snapshot_topics(snapshot)
        bootstrap_topics: dict[str, Any] = {}
        for source_topics in (retained_topics, dynamic_topics, heavy_topics):
            for topic, topic_payload in source_topics.items():
                _merge_topic(bootstrap_topics, topic, topic_payload)
        tick_payload = self._latest_tick_payload()
        tick_id = tick_payload.get("tick_id", -1)
        sim_time_s = tick_payload.get("sim_time_s", 0.0)
        envelope, text = self._build_envelope("bootstrap", tick_id, sim_time_s, bootstrap_topics)
        with self._send_lock:
            self._bootstrap_needed = not bool(bootstrap_topics)
        self._mark_bootstrap_sent(retained_topics, heavy_topics, now_monotonic)
        self._last_dynamic_monotonic = now_monotonic
        self._log_ws_stats(envelope, now_monotonic)
        self._queue_text(message_kind="bootstrap", text=text)

    def _build_envelope(self, message_kind: str, tick_id: int, sim_time_s: float, topics: dict[str, Any]) -> tuple[dict[str, Any], str]:
        serialized_topics = serialize(topics)
        topic_bytes = {topic: _json_size_bytes(payload) for topic, payload in serialized_topics.items()}
        envelope = {
            "message_kind": message_kind,
            "tick_id": int(tick_id),
            "sim_time_s": float(sim_time_s),
            "topics": serialized_topics,
            "ws_stats": {
                "message_kind": message_kind,
                "total_bytes": 0,
                "topic_count": len(serialized_topics),
                "topic_bytes": topic_bytes,
            },
        }
        text = json.dumps(envelope, default=str)
        envelope["ws_stats"]["total_bytes"] = len(text.encode("utf-8"))
        text = json.dumps(envelope, default=str)
        return envelope, text

    def _log_ws_stats(self, envelope: dict[str, Any], now_monotonic: float) -> None:
        if now_monotonic - self._stats_last_log_monotonic < _WS_STATS_LOG_INTERVAL_S:
            return
        self._stats_last_log_monotonic = now_monotonic
        stats = envelope.get("ws_stats", {})
        topic_bytes = dict(stats.get("topic_bytes", {}))
        top_topics = sorted(topic_bytes.items(), key=lambda item: item[1], reverse=True)[:4]
        logger.info(
            "WS %s total=%s bytes topics=%s",
            stats.get("message_kind", "unknown"),
            stats.get("total_bytes", 0),
            top_topics,
        )

    def _queue_text(self, *, message_kind: str, text: str) -> None:
        should_schedule = False
        with self._send_lock:
            if message_kind == "bootstrap":
                self._pending_bootstrap_text = text
            elif message_kind == "static_update":
                self._pending_static_text = text
            else:
                self._pending_dynamic_text = text
            if not self._send_active:
                self._send_active = True
                should_schedule = True
        if should_schedule and self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._drain_pending(), self._loop)

    def _next_pending_text(self) -> str | None:
        if self._pending_bootstrap_text is not None:
            text = self._pending_bootstrap_text
            self._pending_bootstrap_text = None
            return text
        if self._pending_dynamic_text is not None:
            text = self._pending_dynamic_text
            self._pending_dynamic_text = None
            return text
        if self._pending_static_text is not None:
            text = self._pending_static_text
            self._pending_static_text = None
            return text
        return None

    def _on_tick_complete(self, _topic: str, payload: Any) -> None:
        if not self._clients or self._event_bus is None:
            return

        now_monotonic = time.monotonic()
        tick_id = payload.get("tick_id", -1) if isinstance(payload, dict) else -1
        sim_time_s = payload.get("sim_time_s", 0.0) if isinstance(payload, dict) else 0.0
        snapshot = self._snapshot()
        retained_topics, dynamic_topics, heavy_topics = self._prepare_snapshot_topics(snapshot)

        with self._send_lock:
            bootstrap_needed = self._bootstrap_needed
            if bootstrap_needed:
                self._bootstrap_needed = False

        if bootstrap_needed:
            bootstrap_topics: dict[str, Any] = {}
            for source_topics in (retained_topics, dynamic_topics, heavy_topics):
                for topic, topic_payload in source_topics.items():
                    _merge_topic(bootstrap_topics, topic, topic_payload)
            envelope, text = self._build_envelope("bootstrap", tick_id, sim_time_s, bootstrap_topics)
            self._mark_bootstrap_sent(retained_topics, heavy_topics, now_monotonic)
            self._last_dynamic_monotonic = now_monotonic
            self._log_ws_stats(envelope, now_monotonic)
            self._queue_text(message_kind="bootstrap", text=text)
            return

        retained_updates = self._update_retained_cache(retained_topics)
        if retained_updates:
            envelope, text = self._build_envelope("static_update", tick_id, sim_time_s, retained_updates)
            self._log_ws_stats(envelope, now_monotonic)
            self._queue_text(message_kind="static_update", text=text)

        min_dynamic_interval_s = 1.0 / _FAST_DYNAMIC_FPS
        if (now_monotonic - self._last_dynamic_monotonic) < min_dynamic_interval_s:
            return

        dynamic_frame_topics: dict[str, Any] = {}
        for topic, topic_payload in dynamic_topics.items():
            _merge_topic(dynamic_frame_topics, topic, topic_payload)
        for topic, topic_payload in self._collect_heavy_updates(heavy_topics, now_monotonic).items():
            _merge_topic(dynamic_frame_topics, topic, topic_payload)

        if not dynamic_frame_topics:
            return
        envelope, text = self._build_envelope("dynamic_frame", tick_id, sim_time_s, dynamic_frame_topics)
        self._last_dynamic_monotonic = now_monotonic
        self._log_ws_stats(envelope, now_monotonic)
        self._queue_text(message_kind="dynamic_frame", text=text)

    async def _drain_pending(self) -> None:
        while True:
            with self._send_lock:
                text = self._next_pending_text()
                if text is None:
                    self._send_active = False
                    return
            await self._broadcast_once(text)

    async def _broadcast_once(self, text: str) -> None:
        stale: list[Any] = []
        for ws in self._clients:
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("WebSocket send timed out; dropping client")
                stale.append(ws)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._clients.discard(ws)
