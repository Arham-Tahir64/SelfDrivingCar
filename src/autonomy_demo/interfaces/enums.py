from __future__ import annotations

from enum import Enum


class ObjectClass(str, Enum):
    VEHICLE = "vehicle"
    CYCLIST = "cyclist"
    PEDESTRIAN = "pedestrian"
    CONE = "cone"
    TRAFFIC_LIGHT = "traffic_light"
    EMERGENCY_VEHICLE = "emergency_vehicle"


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    DELETED = "DELETED"


class LaneLineType(str, Enum):
    SOLID = "SOLID"
    DASHED = "DASHED"
    TEMPORARY = "TEMPORARY"


class TrafficLightState(str, Enum):
    RED = "RED"
    AMBER = "AMBER"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"


class BehaviorState(str, Enum):
    LANE_KEEP = "LANE_KEEP"
    PREPARE_MERGE = "PREPARE_MERGE"
    MERGING = "MERGING"
    INTERSECTION_APPROACH = "INTERSECTION_APPROACH"
    STOPPING_FOR_RED = "STOPPING_FOR_RED"
    PEDESTRIAN_YIELD = "PEDESTRIAN_YIELD"
    EMERGENCY_YIELD = "EMERGENCY_YIELD"
    CONSTRUCTION_NAVIGATE = "CONSTRUCTION_NAVIGATE"
    GOAL_REACHED = "GOAL_REACHED"


class SensorStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class TopicName(str, Enum):
    SENSOR_CAMERA_FRONT = "sensor/camera/front"
    SENSOR_LIDAR = "sensor/lidar"
    PERCEPTION_DETECTIONS = "perception/detections"
    PERCEPTION_LANES = "perception/lanes"
    PERCEPTION_DRIVABLE_SPACE = "perception/drivable_space"
    PERCEPTION_SEMANTIC_SEG = "perception/semantic_seg"
    PERCEPTION_TRAFFIC_LIGHTS = "perception/traffic_lights"
    PERCEPTION_CONES = "perception/cones"
    PERCEPTION_STATUS = "perception/status"
    LOCALIZATION_EGO_POSE = "localization/ego_pose"
    MAP_LOCAL_MAP = "map/local_map"
    PREDICTION_AGENTS = "prediction/agents"
    PLANNING_EGO_TRAJECTORY = "planning/ego_trajectory"
    CONTROL_VEHICLE_COMMAND = "control/vehicle_command"
    TICK_COMPLETE = "system/tick_complete"
    SCENARIO_INFO = "system/scenario_info"
    VISUALIZATION_CAMERA_OVERLAY = "visualization/camera_overlay"
    VISUALIZATION_LIDAR_PREVIEW = "visualization/lidar_preview"
    PLANNING_CANDIDATES = "planning/candidates"
    PIPELINE_LATENCY = "pipeline/latency"
    VISUALIZATION_BEV_DRIVABLE = "visualization/bev_drivable"
    VISUALIZATION_ROAD_CORRIDOR = "visualization/road_corridor"
    VISUALIZATION_WORLD_LAYER = "visualization/world_layer"
    VISUALIZATION_PRIOR_MAP = "visualization/prior_map"
