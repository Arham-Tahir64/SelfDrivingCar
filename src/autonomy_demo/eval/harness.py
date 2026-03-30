from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from autonomy_demo.common.geometry import distance_xyz
from autonomy_demo.common.logging import get_logger
from autonomy_demo.eval.metrics import LatencyAccumulator
from autonomy_demo.interfaces.enums import TopicName
from autonomy_demo.interfaces.types import ControlCommand, DrivableSpaceMask, EgoPose, EgoTrajectory, EvaluationSummary, LaneLine, ObjectDetection, RoutePlan
from autonomy_demo.planning.route_following import route_progress_distance


class SimpleEvaluationHarness:
    """Minimal online evaluator aligned with the scenario eval block."""

    def __init__(self, scenario) -> None:
        self.scenario = scenario
        self.tick_count = 0
        self.latest_snapshot: dict[str, Any] = {}
        self.final_summary: EvaluationSummary | None = None

    def update(self, tick_id: int, snapshot: dict[str, Any]) -> None:
        self.tick_count = tick_id + 1
        self.latest_snapshot = snapshot

    def finalize(self) -> EvaluationSummary:
        if self.final_summary is not None:
            return self.final_summary
        completion_rate = 1.0 if self.tick_count else 0.0
        success = completion_rate >= self.scenario.eval.min_completion_rate
        self.final_summary = EvaluationSummary(
            scenario_id=self.scenario.scenario_id,
            success=success,
            completion_rate=completion_rate,
            collision_count=0,
            red_light_violations=0,
            pedestrian_clearance_min_m=max(self.scenario.eval.min_pedestrian_clearance_m, 2.5),
            latency_ms={
                "perception": 5.0,
                "localization": 1.0,
                "mapping": 2.0,
                "prediction": 2.0,
                "planning": 3.0,
                "control": 1.0,
            },
            distance_traveled_m=0.0,
            goal_reached=completion_rate >= 1.0,
            sim_duration_s=float(self.tick_count / 20.0),
            mean_speed_mps=0.0,
            max_speed_mps=0.0,
            notes=["Stub evaluation only; replace with scenario-specific metrics."],
        )
        return self.final_summary

    def write_summary(self, output_dir: Path) -> Path:
        summary = self.finalize()
        path = output_dir / "evaluation_summary.json"
        path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return path


class LiveEvaluationHarness:
    """Real online metrics for the first live CARLA driving slice."""

    def __init__(self, scenario, *, tick_hz: int, backend) -> None:
        self.scenario = scenario
        self.tick_hz = tick_hz
        self.backend = backend
        self.tick_count = 0
        self.latest_snapshot: dict[str, Any] = {}
        self.route_plan: RoutePlan | None = None
        self.previous_position: np.ndarray | None = None
        self.last_position: np.ndarray | None = None
        self.distance_traveled_m = 0.0
        self.speed_sum_mps = 0.0
        self.speed_samples = 0
        self.max_speed_mps = 0.0
        self.goal_reached = False
        self.final_summary: EvaluationSummary | None = None
        self.logger = get_logger(__name__, evaluator="live")
        self.goal_xyz = np.array(
            [scenario.ego_goal.x, scenario.ego_goal.y, scenario.ego_goal.z],
            dtype=np.float32,
        )
        self.goal_tolerance_m = 6.0
        self.detection_count_sum = 0
        self.ticks_with_lanes = 0
        self.ticks_with_drivable = 0
        self.perception_ticks = 0
        self.previous_track_ids: set[int] = set()
        self.track_continuity_sum = 0.0
        self.valid_lane_ticks = 0
        self.abs_lateral_offset_sum = 0.0
        self.route_progress_alignment_offset: float | None = None
        self.route_progress_error_sum = 0.0
        self.route_progress_error_samples = 0
        self.tracking_error_sum = 0.0
        self.tracking_error_samples = 0
        self.steer_sum = 0.0
        self.max_abs_steer = 0.0
        self.command_samples = 0
        self.command_toggle_count = 0
        self.emergency_override_count = 0
        self.previous_command_mode: str | None = None
        self.red_light_stop_checks = 0
        self.red_light_stop_successes = 0
        self._latency: LatencyAccumulator | None = None

    def set_latency(self, latency: LatencyAccumulator) -> None:
        self._latency = latency

    def set_route_plan(self, route_plan: RoutePlan | None) -> None:
        self.route_plan = route_plan
        if route_plan is not None:
            self.goal_xyz = np.asarray(route_plan.goal_xyz, dtype=np.float32)
            self.goal_tolerance_m = float(route_plan.goal_tolerance_m)

    def update(self, tick_id: int, snapshot: dict[str, Any]) -> None:
        self.tick_count = tick_id + 1
        self.latest_snapshot = snapshot
        ego_pose = snapshot.get(TopicName.LOCALIZATION_EGO_POSE.value)
        if not isinstance(ego_pose, EgoPose):
            return
        detections = snapshot.get(TopicName.PERCEPTION_DETECTIONS.value, [])
        lanes = snapshot.get(TopicName.PERCEPTION_LANES.value, [])
        drivable = snapshot.get(TopicName.PERCEPTION_DRIVABLE_SPACE.value)
        trajectory = snapshot.get(TopicName.PLANNING_EGO_TRAJECTORY.value)
        command = snapshot.get(TopicName.CONTROL_VEHICLE_COMMAND.value)
        traffic_lights = snapshot.get(TopicName.PERCEPTION_TRAFFIC_LIGHTS.value, [])
        detection_track_ids = {
            detection.track_id
            for detection in detections
            if isinstance(detection, ObjectDetection)
        }
        self.perception_ticks += 1
        self.detection_count_sum += len(detection_track_ids)
        if detection_track_ids and self.previous_track_ids:
            overlap = len(detection_track_ids & self.previous_track_ids)
            self.track_continuity_sum += overlap / max(len(self.previous_track_ids), 1)
        self.previous_track_ids = detection_track_ids
        if isinstance(lanes, list) and any(isinstance(lane, LaneLine) for lane in lanes):
            self.ticks_with_lanes += 1
        if isinstance(drivable, DrivableSpaceMask) and bool(np.any(drivable.mask)):
            self.ticks_with_drivable += 1
        position = np.asarray(ego_pose.world_xyz, dtype=np.float32)
        if self.previous_position is not None:
            self.distance_traveled_m += distance_xyz(self.previous_position, position)
        self.previous_position = position
        self.last_position = position
        if ego_pose.current_lane_id:
            self.valid_lane_ticks += 1
        self.abs_lateral_offset_sum += abs(float(ego_pose.frenet_d))
        if self.route_plan is not None and self.route_plan.total_distance_m > 0.0:
            route_progress = route_progress_distance(self.route_plan, position)
            if self.route_progress_alignment_offset is None:
                self.route_progress_alignment_offset = route_progress - float(ego_pose.frenet_s)
            aligned_progress = float(ego_pose.frenet_s) + float(self.route_progress_alignment_offset)
            self.route_progress_error_sum += abs(route_progress - aligned_progress)
            self.route_progress_error_samples += 1
        if isinstance(trajectory, EgoTrajectory) and trajectory.waypoints:
            target = trajectory.waypoints[min(1, len(trajectory.waypoints) - 1)]
            self.tracking_error_sum += distance_xyz(
                position,
                np.array([target.x, target.y, float(position[2])], dtype=np.float32),
            )
            self.tracking_error_samples += 1
        if isinstance(command, ControlCommand):
            self.steer_sum += abs(float(command.steer))
            self.max_abs_steer = max(self.max_abs_steer, abs(float(command.steer)))
            self.command_samples += 1
            if command.emergency_override:
                self.emergency_override_count += 1
            command_mode = "brake" if command.brake > 0.05 else "throttle" if command.throttle > 0.05 else "coast"
            if self.previous_command_mode is not None and command_mode != self.previous_command_mode:
                self.command_toggle_count += 1
            self.previous_command_mode = command_mode
        red_lights = [
            light
            for light in traffic_lights
            if getattr(light, "state", None) is not None and getattr(light.state, "value", "") in {"RED", "AMBER"}
        ]
        if red_lights:
            nearest_red = min(red_lights, key=lambda light: light.stop_line_distance_m)
            if nearest_red.stop_line_distance_m <= 10.0:
                self.red_light_stop_checks += 1
                if ego_pose.speed_mps <= 1.5:
                    self.red_light_stop_successes += 1
        self.speed_sum_mps += float(ego_pose.speed_mps)
        self.speed_samples += 1
        self.max_speed_mps = max(self.max_speed_mps, float(ego_pose.speed_mps))
        self.goal_reached = self.goal_reached or (
            distance_xyz(position, self.goal_xyz) <= self.goal_tolerance_m
        )
        if tick_id % max(self.tick_hz, 1) == 0:
            self.logger.info(
                "Live metrics tick=%s speed=%.2f m/s distance=%.1f m goal_reached=%s",
                tick_id,
                ego_pose.speed_mps,
                self.distance_traveled_m,
                self.goal_reached,
            )

    def finalize(self) -> EvaluationSummary:
        if self.final_summary is not None:
            return self.final_summary
        sim_duration_s = float(self.tick_count / max(self.tick_hz, 1))
        completion_rate = 0.0
        if self.goal_reached:
            completion_rate = 1.0
        elif self.route_plan is not None and self.route_plan.total_distance_m > 0.0 and self.last_position is not None:
            completion_rate = min(
                1.0,
                route_progress_distance(self.route_plan, self.last_position)
                / self.route_plan.total_distance_m,
            )
        elif self.last_position is not None:
            start_xyz = np.array(
                [self.scenario.ego_spawn.x, self.scenario.ego_spawn.y, self.scenario.ego_spawn.z],
                dtype=np.float32,
            )
            straight_line_total = max(distance_xyz(start_xyz, self.goal_xyz), 1.0)
            completion_rate = min(1.0, distance_xyz(start_xyz, self.last_position) / straight_line_total)

        collision_count = len(getattr(getattr(self.backend, "state", None), "collision_events", []))
        mean_speed_mps = self.speed_sum_mps / self.speed_samples if self.speed_samples else 0.0
        avg_detection_count = self.detection_count_sum / self.perception_ticks if self.perception_ticks else 0.0
        track_continuity_ratio = (
            self.track_continuity_sum / max(self.perception_ticks - 1, 1)
            if self.perception_ticks > 1
            else 0.0
        )
        lane_output_ratio = self.ticks_with_lanes / self.perception_ticks if self.perception_ticks else 0.0
        drivable_output_ratio = (
            self.ticks_with_drivable / self.perception_ticks if self.perception_ticks else 0.0
        )
        valid_lane_ratio = self.valid_lane_ticks / self.tick_count if self.tick_count else 0.0
        mean_abs_lateral_offset_m = (
            self.abs_lateral_offset_sum / self.tick_count if self.tick_count else 0.0
        )
        mean_route_progress_error_m = (
            self.route_progress_error_sum / self.route_progress_error_samples
            if self.route_progress_error_samples
            else 0.0
        )
        mean_tracking_error_m = (
            self.tracking_error_sum / self.tracking_error_samples if self.tracking_error_samples else 0.0
        )
        mean_abs_steer = self.steer_sum / self.command_samples if self.command_samples else 0.0
        throttle_brake_oscillation_ratio = (
            self.command_toggle_count / max(self.command_samples - 1, 1) if self.command_samples > 1 else 0.0
        )
        red_light_stop_compliance = (
            self.red_light_stop_successes / self.red_light_stop_checks if self.red_light_stop_checks else 1.0
        )
        success = (
            completion_rate >= self.scenario.eval.min_completion_rate
            and collision_count <= self.scenario.eval.max_collisions
        )
        notes = [
            "Live CARLA metrics enabled for distance, duration, speed, goal reach, and collisions.",
            "Red-light and pedestrian-clearance metrics remain placeholder until scenario-specific event logic lands.",
            f"Perception avg detections/tick: {avg_detection_count:.2f}",
            f"Perception track continuity ratio: {track_continuity_ratio:.2f}",
            f"Perception lane output ratio: {lane_output_ratio:.2f}",
            f"Perception drivable output ratio: {drivable_output_ratio:.2f}",
            f"Localization valid lane ratio: {valid_lane_ratio:.2f}",
            f"Localization mean |d|: {mean_abs_lateral_offset_m:.2f} m",
            f"Localization/route progress mean error: {mean_route_progress_error_m:.2f} m",
            f"Planning mean tracking error: {mean_tracking_error_m:.2f} m",
            f"Control mean |steer|: {mean_abs_steer:.2f}",
            f"Control max |steer|: {self.max_abs_steer:.2f}",
            f"Control throttle/brake oscillation ratio: {throttle_brake_oscillation_ratio:.2f}",
            f"Emergency override count: {self.emergency_override_count}",
            f"Red-light stop compliance: {red_light_stop_compliance:.2f}",
        ]
        self.final_summary = EvaluationSummary(
            scenario_id=self.scenario.scenario_id,
            success=success,
            completion_rate=float(completion_rate),
            collision_count=collision_count,
            red_light_violations=0,
            pedestrian_clearance_min_m=max(self.scenario.eval.min_pedestrian_clearance_m, 0.0),
            latency_ms=self._latency.mean() if self._latency else {},
            distance_traveled_m=float(self.distance_traveled_m),
            goal_reached=bool(self.goal_reached),
            sim_duration_s=sim_duration_s,
            mean_speed_mps=float(mean_speed_mps),
            max_speed_mps=float(self.max_speed_mps),
            notes=notes,
        )
        return self.final_summary

    def write_summary(self, output_dir: Path) -> Path:
        summary = self.finalize()
        path = output_dir / "evaluation_summary.json"
        path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        return path
