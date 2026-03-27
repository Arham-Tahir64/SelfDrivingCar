from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from autonomy_demo.common.geometry import distance_xyz
from autonomy_demo.common.logging import get_logger
from autonomy_demo.interfaces.enums import TopicName
from autonomy_demo.interfaces.types import EgoPose, EvaluationSummary, RoutePlan
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
        position = np.asarray(ego_pose.world_xyz, dtype=np.float32)
        if self.previous_position is not None:
            self.distance_traveled_m += distance_xyz(self.previous_position, position)
        self.previous_position = position
        self.last_position = position
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
        success = (
            completion_rate >= self.scenario.eval.min_completion_rate
            and collision_count <= self.scenario.eval.max_collisions
        )
        notes = [
            "Live CARLA metrics enabled for distance, duration, speed, goal reach, and collisions.",
            "Red-light and pedestrian-clearance metrics remain placeholder until scenario-specific event logic lands.",
        ]
        self.final_summary = EvaluationSummary(
            scenario_id=self.scenario.scenario_id,
            success=success,
            completion_rate=float(completion_rate),
            collision_count=collision_count,
            red_light_violations=0,
            pedestrian_clearance_min_m=max(self.scenario.eval.min_pedestrian_clearance_m, 0.0),
            latency_ms={},
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
