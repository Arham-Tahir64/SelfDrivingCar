from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.types import DrivableSpaceMask, EgoPose, LocalMap, StaticLaneSegment, RoutePlan, RouteWaypoint
from autonomy_demo.perception.bev_projection import BEVDrivableProjector, GRID_SIZE


def _ego_pose() -> EgoPose:
    return EgoPose(
        world_xyz=np.zeros(3, dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=0.0,
        acceleration_mps2=0.0,
        current_lane_id="lane_001",
        frenet_s=0.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )


def _drivable(mask: np.ndarray, confidence: float = 0.9) -> DrivableSpaceMask:
    probabilities = np.zeros(mask.shape + (2,), dtype=np.float32)
    probabilities[..., 1] = np.where(mask, confidence, 0.0).astype(np.float32)
    probabilities[..., 0] = 1.0 - probabilities[..., 1]
    return DrivableSpaceMask(
        mask=mask.astype(np.bool_),
        class_probabilities=probabilities,
        source_sensor_id="front_camera",
    )


def _pitched_calibration(height: int, width: int) -> dict[str, object]:
    return {
        "fov_deg": 90.0,
        "image_width": width,
        "image_height": height,
        "mount_xyz": [2.3, 0.0, 1.2],
        "mount_rpy_deg": [0.0, 12.0, 0.0],
    }


def _static_lane(
    lane_id: str,
    *,
    x0: float,
    x1: float,
    center_y: float = 0.0,
    predecessors: list[str] | None = None,
    successors: list[str] | None = None,
    is_junction: bool = False,
) -> StaticLaneSegment:
    centerline = np.array([[x0, center_y, 0.0], [x1, center_y, 0.0]], dtype=np.float32)
    left = np.array([[x0, center_y + 1.75, 0.0], [x1, center_y + 1.75, 0.0]], dtype=np.float32)
    right = np.array([[x0, center_y - 1.75, 0.0], [x1, center_y - 1.75, 0.0]], dtype=np.float32)
    return StaticLaneSegment(
        lane_id=lane_id,
        centerline_world=centerline,
        speed_limit_mps=12.0,
        left_boundary_world=left,
        right_boundary_world=right,
        predecessor_lane_ids=list(predecessors or []),
        successor_lane_ids=list(successors or []),
        is_junction=is_junction,
    )


def _local_map(static_lanes: list[StaticLaneSegment]) -> LocalMap:
    return LocalMap(
        static_lanes=static_lanes,
        dynamic_agents=[],
        cone_instances=[],
        temporary_boundaries=[],
        closed_lanes=[],
        traffic_signal_states=[],
        perceived_lanes=[],
        drivable_space=None,
    )


def _route_plan() -> RoutePlan:
    return RoutePlan(
        waypoints=[
            RouteWaypoint(x=-10.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=0.0, target_speed_mps=12.0),
            RouteWaypoint(x=0.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=10.0, target_speed_mps=12.0),
            RouteWaypoint(x=10.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=20.0, target_speed_mps=12.0),
            RouteWaypoint(x=20.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=30.0, target_speed_mps=12.0),
            RouteWaypoint(x=30.0, y=0.0, z=0.0, yaw=0.0, cumulative_distance_m=40.0, target_speed_mps=12.0),
        ],
        goal_xyz=np.array([30.0, 0.0, 0.0], dtype=np.float32),
        total_distance_m=40.0,
        goal_tolerance_m=5.0,
    )


def test_bev_projection_projects_centered_road_band_into_centered_ribbon() -> None:
    projector = BEVDrivableProjector()
    mask = np.zeros((120, 200), dtype=np.bool_)
    mask[60:, 70:130] = True

    grid = projector.project(
        _drivable(mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*mask.shape),
    )

    occupied = np.argwhere(grid > 0)
    assert occupied.size > 0
    occupied_rows = occupied[:, 0]
    occupied_cols = occupied[:, 1]
    assert np.ptp(occupied_rows) >= 6
    assert 4 <= np.ptp(occupied_cols) <= 20
    assert abs(float(occupied_cols.mean()) - (GRID_SIZE / 2.0)) < 6.0


def test_bev_projection_preserves_left_right_lateral_structure() -> None:
    projector = BEVDrivableProjector()
    left_mask = np.zeros((120, 200), dtype=np.bool_)
    right_mask = np.zeros((120, 200), dtype=np.bool_)
    left_mask[65:, 20:70] = True
    right_mask[65:, 130:180] = True

    left_grid = projector.project(
        _drivable(left_mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*left_mask.shape),
    )
    center_mask = np.zeros((120, 200), dtype=np.bool_)
    center_mask[65:, 75:125] = True
    center_grid = projector.project(
        _drivable(center_mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*center_mask.shape),
    )
    right_grid = projector.project(
        _drivable(right_mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*right_mask.shape),
    )

    left_cols = np.argwhere(left_grid > 0)[:, 1]
    center_cols = np.argwhere(center_grid > 0)[:, 1]
    right_cols = np.argwhere(right_grid > 0)[:, 1]
    assert left_cols.size > 0
    assert center_cols.size > 0
    assert right_cols.size > 0
    assert float(left_cols.mean()) < float(center_cols.mean())
    assert float(center_cols.mean()) < float(right_cols.mean())
    assert float(right_cols.mean()) - float(left_cols.mean()) >= 6.0


def test_bev_projection_discards_pixels_above_horizon_or_behind_camera() -> None:
    projector = BEVDrivableProjector()
    mask = np.zeros((120, 200), dtype=np.bool_)
    mask[:18, 75:125] = True

    grid = projector.project(
        _drivable(mask),
        _ego_pose(),
        camera_calibration={
            "fov_deg": 90.0,
            "image_width": 200,
            "image_height": 120,
            "mount_xyz": [2.3, 0.0, 1.2],
            "mount_rpy_deg": [0.0, 0.0, 0.0],
        },
    )

    assert not grid.any()


def test_bev_projection_turns_sparse_stripes_into_contiguous_ground_patch() -> None:
    projector = BEVDrivableProjector()
    mask = np.zeros((120, 200), dtype=np.bool_)
    for col in range(70, 130, 4):
        mask[62:, col : col + 2] = True

    grid = projector.project(
        _drivable(mask),
        _ego_pose(),
        camera_calibration=_pitched_calibration(*mask.shape),
    )

    occupied = np.argwhere(grid > 0)
    assert occupied.size > 0
    row_min, col_min = occupied.min(axis=0)
    row_max, col_max = occupied.max(axis=0)
    window = grid[row_min : row_max + 1, col_min : col_max + 1] > 0
    density = float(window.mean())
    assert density >= 0.55


def test_world_history_persists_behind_ego_after_motion() -> None:
    projector = BEVDrivableProjector()
    lane = _static_lane("lane_001", x0=-10.0, x1=40.0)
    ego_start = _ego_pose()
    corridor = projector.build_route_corridor(_local_map([lane]), ego_start, route_plan=_route_plan())

    projector.update_world_history(
        np.array([[5.0, 0.0]], dtype=np.float32),
        np.array([220.0], dtype=np.float32),
        sim_time_s=0.0,
        corridor_polygons_xy=list(corridor["polygons_xy"]),
    )

    ego_moved = EgoPose(
        world_xyz=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        yaw_rad=0.0,
        speed_mps=0.0,
        acceleration_mps2=0.0,
        current_lane_id="lane_001",
        frenet_s=0.0,
        frenet_d=0.0,
        heading_error_rad=0.0,
    )
    crop = projector.render_local_crop(ego_moved, sim_time_s=1.0)
    grid = crop["grid"]

    expected_row = int(np.floor((crop["x_max_m"] - (-5.0)) / crop["cell_size_m"]))
    occupied = np.argwhere(grid > 0)
    assert occupied.size > 0
    assert np.any(np.abs(occupied[:, 0] - expected_row) <= 1)


def test_route_corridor_prefers_route_successor_over_side_branch() -> None:
    projector = BEVDrivableProjector()
    predecessor = _static_lane("lane_prev", x0=-20.0, x1=0.0, successors=["lane_curr"])
    current = _static_lane(
        "lane_curr",
        x0=0.0,
        x1=20.0,
        predecessors=["lane_prev"],
        successors=["lane_next", "lane_branch"],
    )
    successor = _static_lane("lane_next", x0=20.0, x1=40.0, predecessors=["lane_curr"])
    branch = _static_lane("lane_branch", x0=20.0, x1=35.0, center_y=6.0, predecessors=["lane_curr"])

    corridor = projector.build_route_corridor(
        _local_map([predecessor, current, successor, branch]),
        EgoPose(
            world_xyz=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            yaw_rad=0.0,
            speed_mps=0.0,
            acceleration_mps2=0.0,
            current_lane_id="lane_curr",
            frenet_s=0.0,
            frenet_d=0.0,
            heading_error_rad=0.0,
        ),
        route_plan=_route_plan(),
    )

    lane_ids = [strip["lane_id"] for strip in corridor["strips"]]
    assert "lane_prev" in lane_ids
    assert "lane_curr" in lane_ids
    assert "lane_next" in lane_ids
    assert "lane_branch" not in lane_ids


def test_world_history_clips_out_of_corridor_points() -> None:
    projector = BEVDrivableProjector()
    lane = _static_lane("lane_001", x0=-10.0, x1=40.0)
    corridor = projector.build_route_corridor(_local_map([lane]), _ego_pose(), route_plan=_route_plan())

    projector.update_world_history(
        np.array([[5.0, 0.0], [5.0, 8.0]], dtype=np.float32),
        np.array([200.0, 220.0], dtype=np.float32),
        sim_time_s=0.0,
        corridor_polygons_xy=list(corridor["polygons_xy"]),
    )
    crop = projector.render_local_crop(_ego_pose(), sim_time_s=0.2)
    occupied = np.argwhere(crop["grid"] > 0)

    assert occupied.size > 0
    col_values = occupied[:, 1]
    assert np.all(col_values < crop["cols"] - 8)
