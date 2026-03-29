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
}

export interface LaneLine {
  lane_id: string;
  polyline_world: number[][];
  lane_type: string;
  confidence: number;
}

export interface TrafficLightDetection {
  state: string;
  world_xyz: number[];
  confidence: number;
}

export interface ConeDetection {
  world_xyz: number[];
  confidence: number;
}

export interface AgentPrediction {
  track_id: number;
  object_class: string;
  predicted_trajectory: number[][];
  confidence_by_step: number[];
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
  left_boundary: number[][];
  right_boundary: number[][];
  speed_limit_mps: number;
  is_closed: boolean;
}

export interface LocalMap {
  static_lanes: StaticLane[];
  dynamic_agents: ObjectDetection[];
  cone_instances: ConeDetection[];
  temporary_boundaries: number[][][];
  closed_lanes: string[];
  traffic_signal_states: Record<string, string>;
}

export interface ScenarioInfo {
  scenario_id: string;
  name: string;
  map_name: string;
  max_duration_s: number;
}

export interface PipelineFrame {
  tick_id: number;
  sim_time_s: number;
  "localization/ego_pose"?: EgoPose;
  "perception/detections"?: ObjectDetection[];
  "perception/lanes"?: LaneLine[];
  "perception/traffic_lights"?: TrafficLightDetection[];
  "perception/cones"?: ConeDetection[];
  "map/local_map"?: LocalMap;
  "prediction/agents"?: AgentPrediction[];
  "planning/ego_trajectory"?: EgoTrajectory;
  "control/vehicle_command"?: ControlCommand;
  "system/scenario_info"?: ScenarioInfo;
}
