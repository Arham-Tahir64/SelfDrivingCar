export interface Waypoint {
  x: number;
  y: number;
  yaw: number;
  velocity: number;
  timestamp: number;
}

export interface ObjectDetection {
  track_id: number;
  object_class: string;
  world_bbox_3d: number[][];
  velocity: number[];
  confidence: number;
  track_state: string;
  source_modality: "camera" | "lidar" | "fused" | "bootstrap";
  source_sensor_ids: string[];
  position_estimate_kind:
    | "camera_projection"
    | "lidar_cluster"
    | "fusion"
    | "truth_fallback";
}

export interface LaneLine {
  lane_id: string;
  polyline_image?: number[][];
  polyline_world: number[][];
  line_type: string;
  confidence: number;
  source_modality: "camera" | "lidar" | "fused" | "bootstrap";
  source_sensor_ids: string[];
  position_estimate_kind:
    | "camera_projection"
    | "lidar_projection"
    | "fusion"
    | "truth_fallback";
}

export interface TrafficLightDetection {
  state: string;
  world_xyz: number[];
  confidence: number;
  source_modality: "camera" | "lidar" | "fused" | "bootstrap";
  source_sensor_ids: string[];
  position_estimate_kind:
    | "camera_projection"
    | "lidar_cluster"
    | "fusion"
    | "truth_fallback";
}

export interface AgentPrediction {
  track_id: number;
  object_class: string;
  predicted_trajectory: Waypoint[];
  confidence_by_step: number[];
  covariance_by_step?: number[][][];
}

export interface EgoPose {
  world_xyz: number[];
  yaw_rad: number;
  speed_mps: number;
  acceleration_mps2: number;
  current_lane_id: string;
  frenet_s: number;
  frenet_d: number;
  heading_error_rad: number;
}

export interface EgoTrajectory {
  waypoints: Waypoint[];
  cost: number;
  behavior_state: string;
}

export interface ControlCommand {
  throttle: number;
  steer: number;
  brake: number;
  emergency_override: boolean;
}

export interface StaticLane {
  lane_id: string;
  centerline_world: number[][];
  left_boundary_world: number[][];
  right_boundary_world: number[][];
  speed_limit_mps: number;
  is_junction?: boolean;
}

export interface LocalMap {
  static_lanes: StaticLane[];
  dynamic_agents: ObjectDetection[];
  temporary_boundaries: LaneLine[];
  closed_lanes: string[];
  traffic_signal_states: TrafficLightDetection[];
  perceived_lanes: LaneLine[];
}

export interface ScenarioInfo {
  scenario_id: string;
  name: string;
  map_name: string;
  max_duration_s: number;
}

export interface PerceptionStatus {
  active_mode: string;
  fallback_state: string;
  counts_by_modality: Record<string, number>;
  active_camera_sensors: string[];
  detection_count: number;
  traffic_light_count: number;
}

export interface PlannerCandidate {
  trajectory: EgoTrajectory;
  lane_id: string;
  target_speed_mps: number;
  score: number;
}

export interface PipelineLatency {
  perception: number;
  localization: number;
  mapping: number;
  prediction: number;
  planning: number;
  control: number;
  total: number;
  [key: string]: number;
}

export interface LidarPreview {
  points: number[][];
}

export interface BEVDrivableGrid {
  grid_b64: string; // base64-encoded uint8 flat array (rows * cols)
  rows: number;
  cols: number;
  cell_size_m: number;
  x_min_m: number;
  x_max_m: number;
  y_min_m: number;
  y_max_m: number;
}

export interface RoadCorridorStrip {
  lane_id: string;
  left_boundary_world: number[][];
  right_boundary_world: number[][];
  polygon_world: number[][];
  is_junction?: boolean;
}

export interface RoadCorridorPayload {
  strips: RoadCorridorStrip[];
}

export interface PipelineFrame {
  tick_id: number;
  sim_time_s: number;
  "localization/ego_pose"?: EgoPose;
  "perception/detections"?: ObjectDetection[];
  "perception/lanes"?: LaneLine[];
  "perception/traffic_lights"?: TrafficLightDetection[];
  "map/local_map"?: LocalMap;
  "prediction/agents"?: AgentPrediction[];
  "planning/ego_trajectory"?: EgoTrajectory;
  "control/vehicle_command"?: ControlCommand;
  "perception/status"?: PerceptionStatus;
  "system/scenario_info"?: ScenarioInfo;
  "planning/candidates"?: PlannerCandidate[];
  "visualization/camera_overlay"?: string;
  "visualization/lidar_preview"?: LidarPreview;
  "visualization/bev_drivable"?: BEVDrivableGrid;
  "visualization/road_corridor"?: RoadCorridorPayload;
  "pipeline/latency"?: PipelineLatency;
}
