from __future__ import annotations

import math

import numpy as np

from autonomy_demo.common.geometry import distance_xyz
from autonomy_demo.interfaces.enums import BehaviorState, ObjectClass, TrafficLightState
from autonomy_demo.interfaces.types import AgentPrediction, EgoPose, LocalMap
from autonomy_demo.mapping.lane_graph import parse_lane_id, project_point_to_centerline


class StubBehaviorPlanner:
    """Conservative fallback planner for stub mode and degraded cases."""

    def run(self, local_map: LocalMap, ego_pose: EgoPose) -> BehaviorState:
        if local_map.closed_lanes:
            return BehaviorState.PREPARE_MERGE
        return BehaviorState.LANE_KEEP


class RuleBasedBehaviorPlanner:
    """Rule-based hierarchical FSM with 9 behavior states."""

    def __init__(
        self,
        *,
        goal_tolerance_m: float = 6.0,
        red_light_approach_distance_m: float = 30.0,
        red_light_stop_distance_m: float = 12.0,
        lead_agent_merge_distance_m: float = 18.0,
        merge_cooldown_ticks: int = 30,
        pedestrian_yield_distance_m: float = 20.0,
        pedestrian_lateral_threshold_m: float = 3.5,
        emergency_vehicle_distance_m: float = 40.0,
    ) -> None:
        self.goal_tolerance_m = goal_tolerance_m
        self.red_light_approach_distance_m = red_light_approach_distance_m
        self.red_light_stop_distance_m = red_light_stop_distance_m
        self.lead_agent_merge_distance_m = lead_agent_merge_distance_m
        self.merge_cooldown_ticks = merge_cooldown_ticks
        self.pedestrian_yield_distance_m = pedestrian_yield_distance_m
        self.pedestrian_lateral_threshold_m = pedestrian_lateral_threshold_m
        self.emergency_vehicle_distance_m = emergency_vehicle_distance_m
        self.goal_xyz: np.ndarray | None = None
        self.current_state = BehaviorState.LANE_KEEP
        self._latest_predictions: list[AgentPrediction] = []
        self._merge_source_lane_id: str | None = None
        self._merge_cooldown_remaining = 0
        self._merge_complete_ticks = 0

    def prepare(self, simulation, scenario) -> None:
        self.goal_xyz = np.array(
            [float(scenario.ego_goal.x), float(scenario.ego_goal.y), float(scenario.ego_goal.z)],
            dtype=np.float32,
        )

    def set_context(self, local_map: LocalMap, predictions) -> None:
        self._latest_predictions = predictions

    def run(self, local_map: LocalMap, ego_pose: EgoPose) -> BehaviorState:
        # --- Priority 1: Goal reached ---
        if self.goal_xyz is not None and distance_xyz(ego_pose.world_xyz, self.goal_xyz) <= self.goal_tolerance_m:
            self.current_state = BehaviorState.GOAL_REACHED
            return self.current_state

        # --- Priority 2: Emergency vehicle detected ---
        if self._emergency_vehicle_nearby(local_map, ego_pose):
            self.current_state = BehaviorState.EMERGENCY_YIELD
            return self.current_state

        # --- Priority 3: Red light ---
        red_light = self._active_red_light(local_map)
        if red_light is not None:
            if red_light.stop_line_distance_m <= self.red_light_stop_distance_m:
                self.current_state = BehaviorState.STOPPING_FOR_RED
            else:
                self.current_state = BehaviorState.INTERSECTION_APPROACH
            return self.current_state

        # --- Priority 4: Pedestrian in ego lane ---
        if self._pedestrian_in_path(local_map, ego_pose):
            self.current_state = BehaviorState.PEDESTRIAN_YIELD
            return self.current_state

        # --- Priority 5: Construction zone ---
        if local_map.temporary_boundaries and ego_pose.current_lane_id in local_map.closed_lanes:
            if not self._adjacent_lane_available(local_map, ego_pose.current_lane_id):
                self.current_state = BehaviorState.CONSTRUCTION_NAVIGATE
                return self.current_state

        # --- Priority 6: Active merge completion ---
        if (
            self.current_state in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING}
            and self._merge_source_lane_id is not None
        ):
            in_new_lane = (
                ego_pose.current_lane_id != self._merge_source_lane_id
                and ego_pose.current_lane_id not in local_map.closed_lanes
                and not local_map.temporary_boundaries
                and abs(ego_pose.frenet_d) < 0.5
            )
            if in_new_lane:
                self._merge_complete_ticks += 1
            else:
                self._merge_complete_ticks = 0
            if self._merge_complete_ticks >= 5:
                self.current_state = BehaviorState.LANE_KEEP
                self._merge_source_lane_id = None
                self._merge_complete_ticks = 0
                self._merge_cooldown_remaining = self.merge_cooldown_ticks
                return self.current_state

        # --- Priority 7: Merge request (closure or blocked lead) ---
        closure_merge_requested = (
            ego_pose.current_lane_id in local_map.closed_lanes
            or bool(local_map.temporary_boundaries)
        )
        lead_merge_requested = (
            self._merge_cooldown_remaining <= 0
            and self._blocked_lead_vehicle(local_map, ego_pose)
        )
        should_merge = closure_merge_requested or lead_merge_requested
        if should_merge and self._adjacent_lane_available(local_map, ego_pose.current_lane_id):
            if self._merge_source_lane_id is None:
                self._merge_source_lane_id = ego_pose.current_lane_id
            if self.current_state in {BehaviorState.PREPARE_MERGE, BehaviorState.MERGING} or abs(ego_pose.frenet_d) > 0.8:
                self.current_state = BehaviorState.MERGING
            else:
                self.current_state = BehaviorState.PREPARE_MERGE
            return self.current_state

        if self._merge_cooldown_remaining > 0:
            self._merge_cooldown_remaining -= 1
        self.current_state = BehaviorState.LANE_KEEP
        return self.current_state

    # -- Detectors ----------------------------------------------------------------

    def _active_red_light(self, local_map: LocalMap):
        red_like_states = {TrafficLightState.RED, TrafficLightState.AMBER}
        candidates = [
            signal
            for signal in local_map.traffic_signal_states
            if signal.state in red_like_states and signal.stop_line_distance_m <= self.red_light_approach_distance_m
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda signal: signal.stop_line_distance_m)

    def _pedestrian_in_path(self, local_map: LocalMap, ego_pose: EgoPose) -> bool:
        """Check if any pedestrian's predicted trajectory intersects the ego lane ahead."""
        current_lane = next(
            (lane for lane in local_map.static_lanes if lane.lane_id == ego_pose.current_lane_id),
            None,
        )
        if current_lane is None:
            return False
        ego_projection = project_point_to_centerline(current_lane.centerline_world, ego_pose.world_xyz)

        # Check detected pedestrians directly
        for agent in local_map.dynamic_agents:
            if agent.object_class != ObjectClass.PEDESTRIAN:
                continue
            agent_center = np.mean(np.asarray(agent.world_bbox_3d, dtype=np.float32), axis=0)
            projection = project_point_to_centerline(current_lane.centerline_world, agent_center)
            ahead = projection.s - ego_projection.s
            in_lane = abs(projection.d) <= self.pedestrian_lateral_threshold_m
            if in_lane and 0.0 < ahead <= self.pedestrian_yield_distance_m:
                return True

        # Check predicted trajectories of pedestrians
        for prediction in self._latest_predictions:
            if prediction.object_class != ObjectClass.PEDESTRIAN:
                continue
            for waypoint in prediction.predicted_trajectory[:20]:  # check first 2s
                wp_xyz = np.array([waypoint.x, waypoint.y, 0.0], dtype=np.float32)
                projection = project_point_to_centerline(current_lane.centerline_world, wp_xyz)
                ahead = projection.s - ego_projection.s
                in_lane = abs(projection.d) <= self.pedestrian_lateral_threshold_m
                if in_lane and 0.0 < ahead <= self.pedestrian_yield_distance_m:
                    return True

        return False

    def _emergency_vehicle_nearby(self, local_map: LocalMap, ego_pose: EgoPose) -> bool:
        """Detect an emergency vehicle approaching from behind or nearby."""
        ego_xy = np.asarray(ego_pose.world_xyz, dtype=np.float32)[:2]
        heading_vec = np.array(
            [math.cos(ego_pose.yaw_rad), math.sin(ego_pose.yaw_rad)], dtype=np.float32
        )
        for agent in local_map.dynamic_agents:
            if agent.object_class != ObjectClass.EMERGENCY_VEHICLE:
                continue
            agent_center = np.mean(np.asarray(agent.world_bbox_3d, dtype=np.float32), axis=0)[:2]
            delta = agent_center - ego_xy
            dist = float(np.linalg.norm(delta))
            if dist > self.emergency_vehicle_distance_m:
                continue
            # Check if approaching (behind or converging)
            longitudinal = float(np.dot(delta, heading_vec))
            agent_speed = float(np.linalg.norm(np.asarray(agent.velocity, dtype=np.float32)[:2]))
            # Yield if emergency vehicle is behind and moving, or very close from any direction
            if longitudinal < 5.0 and agent_speed > 1.0:
                return True
            if dist < 15.0:
                return True
        return False

    def _adjacent_lane_available(self, local_map: LocalMap, current_lane_id: str) -> bool:
        current_parts = parse_lane_id(current_lane_id)
        if current_parts is None:
            return False
        road_id, section_id, lane_index = current_parts
        for segment in local_map.static_lanes:
            parts = parse_lane_id(segment.lane_id)
            if parts is None:
                continue
            if (
                parts[0] == road_id
                and parts[1] == section_id
                and abs(parts[2] - lane_index) == 1
                and segment.lane_id not in local_map.closed_lanes
            ):
                return True
        return False

    def _blocked_lead_vehicle(self, local_map: LocalMap, ego_pose: EgoPose) -> bool:
        current_lane = next(
            (lane for lane in local_map.static_lanes if lane.lane_id == ego_pose.current_lane_id),
            None,
        )
        if current_lane is None:
            return False
        ego_projection = project_point_to_centerline(current_lane.centerline_world, ego_pose.world_xyz)
        for agent in local_map.dynamic_agents:
            agent_center = np.mean(np.asarray(agent.world_bbox_3d, dtype=np.float32), axis=0)
            projection = project_point_to_centerline(current_lane.centerline_world, agent_center)
            same_lane = abs(projection.d) <= 2.5
            ahead_distance = projection.s - ego_projection.s
            slow_agent = float(np.linalg.norm(np.asarray(agent.velocity, dtype=np.float32)[:2])) < max(ego_pose.speed_mps * 0.8, 2.0)
            if same_lane and 0.0 < ahead_distance <= self.lead_agent_merge_distance_m and slow_agent:
                return True
        return False
