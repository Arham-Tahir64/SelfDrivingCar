from __future__ import annotations

import math

import numpy as np

from autonomy_demo.common.geometry import clamp, normalize_angle, signed_lateral_error
from autonomy_demo.interfaces.enums import BehaviorState, TrackState
from autonomy_demo.interfaces.types import ControlCommand, EgoPose, EgoTrajectory, LocalMap
from autonomy_demo.mapping.lane_graph import project_point_to_centerline


class StubController:
    """Fallback controller for stub mode."""

    def run(self, trajectory: EgoTrajectory, ego_pose: EgoPose) -> ControlCommand:
        target_speed = trajectory.waypoints[0].velocity if trajectory.waypoints else 0.0
        speed_error = max(target_speed - ego_pose.speed_mps, 0.0)
        throttle = min(speed_error / 10.0, 1.0)
        return ControlCommand(throttle=throttle, steer=0.0, brake=0.0, emergency_override=False)


class RouteFollowerController:
    """Stanley-style lateral control with PID longitudinal control and emergency override."""

    def __init__(
        self,
        *,
        stanley_gain: float = 2.2,
        heading_gain: float = 1.1,
        kp_speed: float = 0.32,
        ki_speed: float = 0.04,
        kd_speed: float = 0.03,
        emergency_ttc_s: float = 1.6,
        emergency_gap_m: float = 6.0,
    ) -> None:
        self.stanley_gain = stanley_gain
        self.heading_gain = heading_gain
        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed
        self.emergency_ttc_s = emergency_ttc_s
        self.emergency_gap_m = emergency_gap_m
        self.lead_lane_tolerance_m = 2.25
        self._integral_speed_error = 0.0
        self._previous_speed_error = 0.0
        self._latest_local_map: LocalMap | None = None
        self._latest_predictions = []

    def set_context(self, local_map: LocalMap, predictions) -> None:
        self._latest_local_map = local_map
        self._latest_predictions = predictions

    def run(self, trajectory: EgoTrajectory, ego_pose: EgoPose) -> ControlCommand:
        if not trajectory.waypoints:
            return ControlCommand(throttle=0.0, steer=0.0, brake=1.0, emergency_override=True)

        target_index = min(2, len(trajectory.waypoints) - 1)
        target_waypoint = trajectory.waypoints[target_index]
        target_speed = max(waypoint.velocity for waypoint in trajectory.waypoints)
        heading_error = normalize_angle(target_waypoint.yaw - ego_pose.yaw_rad)
        lateral_error = signed_lateral_error(
            origin_x=float(ego_pose.world_xyz[0]),
            origin_y=float(ego_pose.world_xyz[1]),
            origin_yaw_rad=float(ego_pose.yaw_rad),
            target_x=float(target_waypoint.x),
            target_y=float(target_waypoint.y),
        )
        stanley_term = math.atan2(
            self.stanley_gain * lateral_error,
            max(abs(ego_pose.speed_mps), 1.0),
        )
        steer = clamp(
            (self.heading_gain * heading_error + stanley_term) / math.pi,
            -1.0,
            1.0,
        )

        if trajectory.behavior_state == BehaviorState.GOAL_REACHED:
            command = ControlCommand(
                throttle=0.0,
                steer=steer,
                brake=0.65,
                emergency_override=False,
            )
            command = self._apply_emergency_override(command, ego_pose)
            return self._apply_launch_assist(command, ego_pose, trajectory.behavior_state, target_speed)

        speed_error = float(target_speed - ego_pose.speed_mps)
        self._integral_speed_error = clamp(
            self._integral_speed_error + (speed_error * 0.1),
            -10.0,
            10.0,
        )
        derivative = (speed_error - self._previous_speed_error) / 0.1
        self._previous_speed_error = speed_error
        pid_output = (
            (self.kp_speed * speed_error)
            + (self.ki_speed * self._integral_speed_error)
            + (self.kd_speed * derivative)
        )
        throttle = clamp(pid_output, 0.0, 0.8)
        brake = clamp(-pid_output, 0.0, 1.0)
        if trajectory.behavior_state == BehaviorState.STOPPING_FOR_RED:
            brake = max(brake, 0.45)
            throttle = min(throttle, 0.15)
        elif trajectory.behavior_state == BehaviorState.PEDESTRIAN_YIELD:
            brake = max(brake, 0.5)
            throttle = min(throttle, 0.1)
        elif trajectory.behavior_state == BehaviorState.EMERGENCY_YIELD:
            brake = max(brake, 0.35)
            throttle = min(throttle, 0.25)
        elif trajectory.behavior_state == BehaviorState.CONSTRUCTION_NAVIGATE:
            throttle = min(throttle, 0.35)
        elif trajectory.behavior_state == BehaviorState.INTERSECTION_APPROACH:
            throttle = min(throttle, 0.45)
        command = ControlCommand(
            throttle=throttle,
            steer=steer,
            brake=brake,
            emergency_override=False,
        )
        command = self._apply_emergency_override(command, ego_pose)
        return self._apply_launch_assist(command, ego_pose, trajectory.behavior_state, target_speed)

    def _apply_emergency_override(self, command: ControlCommand, ego_pose: EgoPose) -> ControlCommand:
        risk = self._lead_vehicle_risk(ego_pose)
        if risk is None:
            return command
        gap_m, ttc_s = risk
        # Hard emergency brake: very close or imminent collision
        if gap_m <= self.emergency_gap_m * 0.5 or ttc_s <= self.emergency_ttc_s * 0.5:
            return ControlCommand(
                throttle=0.0,
                steer=command.steer,
                brake=max(command.brake, 0.9),
                hand_brake=command.hand_brake,
                reverse=command.reverse,
                emergency_override=True,
            )
        # Graduated braking: within caution zone, scale brake by proximity
        if gap_m <= self.emergency_gap_m or ttc_s <= self.emergency_ttc_s:
            gap_ratio = clamp(gap_m / self.emergency_gap_m, 0.0, 1.0)
            ttc_ratio = clamp(ttc_s / self.emergency_ttc_s, 0.0, 1.0)
            urgency = 1.0 - min(gap_ratio, ttc_ratio)
            brake_amount = 0.3 + 0.5 * urgency
            return ControlCommand(
                throttle=0.0,
                steer=command.steer,
                brake=max(command.brake, brake_amount),
                hand_brake=command.hand_brake,
                reverse=command.reverse,
                emergency_override=urgency > 0.7,
            )
        # Comfort zone: lead vehicle ahead but not critical — allow gentle following
        if gap_m <= self.emergency_gap_m * 2.0:
            throttle_scale = clamp((gap_m - self.emergency_gap_m) / self.emergency_gap_m, 0.0, 1.0)
            comfort_brake = command.brake
            if ego_pose.speed_mps >= 2.0:
                comfort_brake = max(command.brake, 0.1 * (1.0 - throttle_scale))
            return ControlCommand(
                throttle=command.throttle * throttle_scale,
                steer=command.steer,
                brake=comfort_brake,
                hand_brake=command.hand_brake,
                reverse=command.reverse,
                emergency_override=False,
            )
        return command

    def _apply_launch_assist(
        self,
        command: ControlCommand,
        ego_pose: EgoPose,
        behavior_state: BehaviorState,
        target_speed: float,
    ) -> ControlCommand:
        if command.emergency_override:
            return command
        if behavior_state in {
            BehaviorState.GOAL_REACHED,
            BehaviorState.STOPPING_FOR_RED,
            BehaviorState.PEDESTRIAN_YIELD,
        }:
            return command
        if target_speed <= 0.5 or ego_pose.speed_mps > 1.0:
            return command
        if command.throttle <= 0.0 or command.brake >= 0.15:
            return command
        return ControlCommand(
            throttle=max(float(command.throttle), 0.25),
            steer=command.steer,
            brake=0.0,
            hand_brake=command.hand_brake,
            reverse=command.reverse,
            emergency_override=False,
        )

    def _lead_vehicle_risk(self, ego_pose: EgoPose) -> tuple[float, float] | None:
        current_lane = None
        ego_projection = None
        detections_by_track: dict[int, object] = {}
        current_detections: list[object] = []
        if self._latest_local_map is not None:
            current_lane = next(
                (
                    lane
                    for lane in self._latest_local_map.static_lanes
                    if lane.lane_id == ego_pose.current_lane_id
                ),
                None,
            )
            if current_lane is not None:
                ego_projection = project_point_to_centerline(current_lane.centerline_world, ego_pose.world_xyz)
            current_detections = list(self._latest_local_map.dynamic_agents)
            detections_by_track = {
                int(agent.track_id): agent for agent in self._latest_local_map.dynamic_agents
            }
        image_detection_risk = self._front_camera_image_risk(
            ego_pose,
            current_detections,
            current_lane=current_lane,
            ego_projection=ego_projection,
        )
        if image_detection_risk is not None:
            return image_detection_risk
        direct_detection_risk = self._direct_vehicle_risk(
            ego_pose,
            current_lane=current_lane,
            ego_projection=ego_projection,
            detections=current_detections,
        )
        if direct_detection_risk is not None:
            return direct_detection_risk
        if not self._latest_predictions:
            return None
        ego_xy = np.asarray(ego_pose.world_xyz, dtype=np.float32)[:2]
        heading_vec = np.array([math.cos(ego_pose.yaw_rad), math.sin(ego_pose.yaw_rad)], dtype=np.float32)
        best: tuple[float, float] | None = None
        for prediction in self._latest_predictions:
            if not prediction.predicted_trajectory:
                continue
            predicted = prediction.predicted_trajectory[0]
            detection = detections_by_track.get(int(prediction.track_id))
            if detection is not None and current_lane is not None and ego_projection is not None:
                if getattr(detection, "track_state", TrackState.TENTATIVE) != TrackState.CONFIRMED:
                    continue
                confidence_threshold = (
                    0.25
                    if str(getattr(detection, "source_modality", "")) in {"camera", "fused"}
                    else 0.55
                )
                if float(getattr(detection, "confidence", 0.0)) < confidence_threshold:
                    continue
                world_bbox = np.asarray(detection.world_bbox_3d, dtype=np.float32)
                size_xyz = np.max(world_bbox, axis=0) - np.min(world_bbox, axis=0)
                if (
                    max(float(size_xyz[0]), float(size_xyz[1])) > 10.0
                    or min(float(size_xyz[0]), float(size_xyz[1])) > 4.5
                    or float(size_xyz[2]) > 4.5
                ):
                    continue
                target_xy = np.mean(world_bbox, axis=0)[:2]
                projection = project_point_to_centerline(current_lane.centerline_world, np.mean(world_bbox, axis=0))
                longitudinal_gap = float(projection.s - ego_projection.s)
                lateral_gap = float(abs(projection.d))
                if longitudinal_gap <= 0.0 or lateral_gap > self.lead_lane_tolerance_m:
                    continue
                lead_speed = float(
                    np.linalg.norm(np.asarray(getattr(detection, "velocity", [0.0, 0.0]), dtype=np.float32)[:2])
                )
                # For camera-projected positions, reduce gap by position uncertainty
                # to brake earlier when depth estimate is unreliable.
                pos_uncertainty = getattr(detection, "position_uncertainty_m", 0.0)
                if pos_uncertainty > 0.0:
                    longitudinal_gap = max(1.0, longitudinal_gap - pos_uncertainty)
            else:
                target_xy = np.array([predicted.x, predicted.y], dtype=np.float32)
                delta = target_xy - ego_xy
                longitudinal_gap = float(np.dot(delta, heading_vec))
                lateral_gap = float(abs((-heading_vec[1] * delta[0]) + (heading_vec[0] * delta[1])))
                if longitudinal_gap <= 0.0 or lateral_gap > 3.0:
                    continue
                lead_speed = float(predicted.velocity)
            closing_speed = max(ego_pose.speed_mps - lead_speed, 0.1)
            ttc_s = longitudinal_gap / closing_speed
            if best is None or longitudinal_gap < best[0]:
                best = (float(longitudinal_gap), float(ttc_s))
        return best

    def _front_camera_image_risk(
        self,
        ego_pose: EgoPose,
        detections: list[object],
        *,
        current_lane,
        ego_projection,
    ) -> tuple[float, float] | None:
        best: tuple[float, float] | None = None
        for detection in detections:
            detection_class = getattr(detection, "object_class", None)
            if detection_class is None:
                continue
            object_class = getattr(detection_class, "value", str(detection_class))
            if object_class != "vehicle":
                continue
            if "front_camera" not in list(getattr(detection, "source_sensor_ids", [])):
                continue
            image_bbox = getattr(detection, "image_bbox_xyxy", None)
            if image_bbox is None:
                continue
            bbox = np.asarray(image_bbox, dtype=np.float32)
            if bbox.shape != (4,):
                continue
            bbox_width = float(max(bbox[2] - bbox[0], 1.0))
            bbox_height = float(max(bbox[3] - bbox[1], 1.0))
            bbox_bottom = float(bbox[3])
            bbox_center_x = float((bbox[0] + bbox[2]) * 0.5)
            confidence = float(getattr(detection, "confidence", 0.0))
            if confidence < 0.15:
                continue
            world_bbox = np.asarray(getattr(detection, "world_bbox_3d", np.zeros((8, 3), dtype=np.float32)), dtype=np.float32)
            if world_bbox.shape == (8, 3) and current_lane is not None and ego_projection is not None:
                center_xyz = np.mean(world_bbox, axis=0)
                projection = project_point_to_centerline(current_lane.centerline_world, center_xyz)
                longitudinal_gap = float(projection.s - ego_projection.s)
                lateral_gap = float(abs(projection.d))
                is_camera_projection = (
                    str(getattr(detection, "position_estimate_kind", "")) == "camera_projection"
                )
                if longitudinal_gap <= 0.0:
                    continue
                if lateral_gap > self.lead_lane_tolerance_m and not is_camera_projection:
                    continue
            if bbox_height < 16.0 or bbox_width < 14.0:
                continue
            # Approximate front-camera frame center for current configs (works for 960/1280 width class).
            if abs(bbox_center_x - 480.0) > 280.0 and abs(bbox_center_x - 640.0) > 340.0:
                continue
            if bbox_bottom < 110.0:
                continue
            estimated_gap_m = float(np.clip((650.0 / bbox_height) - 1.5, 2.0, 24.0))
            if bbox_height >= 18.0 and bbox_width >= 18.0 and bbox_bottom >= 120.0:
                estimated_gap_m = min(estimated_gap_m, 9.0)
            # Use velocity from detection if available, otherwise assume lead is
            # stationary only when bbox is large (close).  For distant bboxes,
            # assume the lead vehicle moves at a similar speed to avoid
            # over-braking on vehicles that are simply cruising ahead.
            lead_speed = 0.0
            velocity_vec = getattr(detection, "velocity", None)
            if velocity_vec is not None:
                lead_speed = float(np.linalg.norm(np.asarray(velocity_vec, dtype=np.float32)[:2]))
            elif bbox_height < 60.0:
                # Distant vehicle — assume it moves at ~70% of ego speed
                lead_speed = ego_pose.speed_mps * 0.7
            closing_speed = max(float(ego_pose.speed_mps) - lead_speed, 0.1)
            ttc_s = estimated_gap_m / closing_speed
            if best is None or estimated_gap_m < best[0]:
                best = (estimated_gap_m, ttc_s)
        return best

    def _direct_vehicle_risk(
        self,
        ego_pose: EgoPose,
        *,
        current_lane,
        ego_projection,
        detections: list[object],
    ) -> tuple[float, float] | None:
        if not detections:
            return None
        ego_xy = np.asarray(ego_pose.world_xyz, dtype=np.float32)[:2]
        heading_vec = np.array([math.cos(ego_pose.yaw_rad), math.sin(ego_pose.yaw_rad)], dtype=np.float32)
        best: tuple[float, float] | None = None
        for detection in detections:
            detection_class = getattr(detection, "object_class", None)
            if detection_class is None:
                continue
            object_class = getattr(detection_class, "value", str(detection_class))
            if object_class != "vehicle":
                continue
            confidence = float(getattr(detection, "confidence", 0.0))
            modality = str(getattr(detection, "source_modality", ""))
            threshold = 0.2 if modality in {"camera", "fused"} else 0.5
            if confidence < threshold:
                continue
            if getattr(detection, "track_state", TrackState.TENTATIVE) == TrackState.DELETED:
                continue
            world_bbox = np.asarray(getattr(detection, "world_bbox_3d", np.zeros((8, 3), dtype=np.float32)), dtype=np.float32)
            if world_bbox.shape != (8, 3):
                continue
            size_xyz = np.max(world_bbox, axis=0) - np.min(world_bbox, axis=0)
            if (
                max(float(size_xyz[0]), float(size_xyz[1])) > 10.0
                or min(float(size_xyz[0]), float(size_xyz[1])) > 4.5
                or float(size_xyz[2]) > 4.5
            ):
                continue
            center_xyz = np.mean(world_bbox, axis=0)
            if current_lane is not None and ego_projection is not None:
                projection = project_point_to_centerline(current_lane.centerline_world, center_xyz)
                longitudinal_gap = float(projection.s - ego_projection.s)
                lateral_gap = float(abs(projection.d))
                if longitudinal_gap <= 0.0 or lateral_gap > self.lead_lane_tolerance_m:
                    continue
            else:
                delta = center_xyz[:2] - ego_xy
                longitudinal_gap = float(np.dot(delta, heading_vec))
                lateral_gap = float(abs((-heading_vec[1] * delta[0]) + (heading_vec[0] * delta[1])))
                if longitudinal_gap <= 0.0 or lateral_gap > 3.0:
                    continue
            lead_speed = float(
                np.linalg.norm(np.asarray(getattr(detection, "velocity", [0.0, 0.0]), dtype=np.float32)[:2])
            )
            closing_speed = max(ego_pose.speed_mps - lead_speed, 0.1)
            ttc_s = longitudinal_gap / closing_speed
            if best is None or longitudinal_gap < best[0]:
                best = (float(longitudinal_gap), float(ttc_s))
        return best


StanleyPidController = RouteFollowerController
