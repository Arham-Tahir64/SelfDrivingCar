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
  feasible: boolean;
  reject_reason: string | null;
  reference_lane_id: string;
  target_lane_id: string;
  target_d_m: number;
  terminal_time_s: number;
  cost_breakdown: {
    collision: number;
    cone_proximity: number;
    lane_deviation: number;
    jerk: number;
    speed_error: number;
    traffic_violation: number;
    route_progress: number;
    total: number;
  };
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
  objects: LidarPanelObject[];
  threat_ids: number[];
  path_polyline_xy?: number[][];
  forward_cone?: {
    length_m: number;
    half_angle_deg: number;
  };
  status?: {
    mode: string;
    degraded: boolean;
    lidar_track_count: number;
    confirmed_track_count: number;
    point_count: number;
  };
}

export interface LidarPanelObject {
  track_id: number;
  track_state: string;
  object_class: string;
  confidence: number;
  source_modality: string;
  footprint_xy: number[][];
  centroid_xy: number[];
  velocity_xy: number[];
  speed_mps: number;
  relevance_score: number;
  threat_rank: number;
  is_path_relevant: boolean;
  ghost_xy?: number[];
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

export interface WorldRoad {
  lane_id: string;
  polygon_world: number[][];
  centerline_world: number[][];
  is_route: boolean;
  visibility_class?: "route" | "adjacent" | "background";
  is_junction?: boolean;
  is_junction_patch?: boolean;
  is_closed?: boolean;
}

export interface WorldLaneMarker {
  marker_id: string;
  polyline_world: number[][];
  is_route: boolean;
  visibility_class?: "route" | "adjacent" | "background";
}

export interface WorldSidewalk {
  sidewalk_id: string;
  polygon_world: number[][];
  edge_world: number[][];
  is_route_adjacent?: boolean;
  visibility_class?: "route" | "adjacent" | "background";
}

export interface WorldTrafficLight {
  actor_id: number;
  world_xyz: number[];
  yaw_deg: number;
  state: string;
  confidence: number;
  visibility_class?: "route" | "adjacent" | "background";
}

export interface WorldLayerPayload {
  signature: string;
  roads: WorldRoad[];
  lane_markers: WorldLaneMarker[];
  sidewalks: WorldSidewalk[];
  traffic_lights: WorldTrafficLight[];
}

export interface PriorMapBounds {
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
}

export interface PriorMapPayload {
  map_name: string;
  signature: string;
  bounds_world: PriorMapBounds;
  roads: WorldRoad[];
  lane_markers: WorldLaneMarker[];
  sidewalks: WorldSidewalk[];
  traffic_lights: WorldTrafficLight[];
  route_polyline_world: number[][];
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
  "visualization/world_layer"?: WorldLayerPayload;
  "visualization/prior_map"?: PriorMapPayload;
  "pipeline/latency"?: PipelineLatency;
}

export interface WebSocketStats {
  message_kind: "bootstrap" | "static_update" | "dynamic_frame";
  total_bytes: number;
  topic_count: number;
  topic_bytes: Record<string, number>;
}

export interface WebSocketEnvelope {
  message_kind: "bootstrap" | "static_update" | "dynamic_frame";
  tick_id: number;
  sim_time_s: number;
  topics: Partial<PipelineFrame>;
  ws_stats?: WebSocketStats;
}
