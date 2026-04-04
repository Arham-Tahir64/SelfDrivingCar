from __future__ import annotations

import numpy as np

from autonomy_demo.interfaces.enums import TrafficLightState
from autonomy_demo.interfaces.types import (
    DrivableSpaceMask,
    EgoPose,
    LocalMap,
    RoutePlan,
    RouteWaypoint,
    StaticLaneSegment,
    TrafficLightDetection,
)
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


def test_world_layer_builds_roads_markers_and_outer_sidewalks() -> None:
    projector = BEVDrivableProjector()
    lane_a = _static_lane("lane_a", x0=-10.0, x1=30.0, center_y=0.0, successors=["lane_c"])
    lane_b = _static_lane("lane_b", x0=-10.0, x1=30.0, center_y=3.5)
    local_map = _local_map([lane_a, lane_b])

    payload = projector.build_world_layer(
        local_map,
        EgoPose(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            yaw_rad=0.0,
            speed_mps=0.0,
            acceleration_mps2=0.0,
            current_lane_id="lane_a",
            frenet_s=0.0,
            frenet_d=0.0,
            heading_error_rad=0.0,
        ),
        route_lane_ids={"lane_a"},
    )

    assert len(payload["roads"]) == 2
    assert len(payload["lane_markers"]) == 1
    assert len(payload["sidewalks"]) == 2
    route_roads = [road for road in payload["roads"] if road["is_route"]]
    assert [road["lane_id"] for road in route_roads] == ["lane_a"]
    route_markers = [marker for marker in payload["lane_markers"] if marker["is_route"]]
    assert len(route_markers) == 1
    adjacent_roads = [road for road in payload["roads"] if road["visibility_class"] == "adjacent"]
    assert [road["lane_id"] for road in adjacent_roads] == ["lane_b"]


def test_world_layer_marks_closed_lanes_without_dropping_geometry() -> None:
    projector = BEVDrivableProjector()
    lane = _static_lane("lane_001", x0=-10.0, x1=30.0)
    local_map = _local_map([lane])
    local_map.closed_lanes = ["lane_001"]

    payload = projector.build_world_layer(local_map, _ego_pose(), route_lane_ids={"lane_001"})

    assert len(payload["roads"]) == 1
    assert payload["roads"][0]["is_closed"] is True
    assert len(payload["sidewalks"]) == 2


def test_world_layer_drops_far_parallel_background_lanes() -> None:
    projector = BEVDrivableProjector()
    route_lane = _static_lane("lane_route", x0=-10.0, x1=30.0, center_y=0.0)
    adjacent_lane = _static_lane("lane_adjacent", x0=-10.0, x1=30.0, center_y=3.5)
    far_lane = _static_lane("lane_far", x0=-10.0, x1=30.0, center_y=14.0)

    payload = projector.build_world_layer(
        _local_map([route_lane, adjacent_lane, far_lane]),
        EgoPose(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            yaw_rad=0.0,
            speed_mps=0.0,
            acceleration_mps2=0.0,
            current_lane_id="lane_route",
            frenet_s=0.0,
            frenet_d=0.0,
            heading_error_rad=0.0,
        ),
        route_lane_ids={"lane_route"},
    )

    road_ids = {road["lane_id"] for road in payload["roads"]}
    assert "lane_route" in road_ids
    assert "lane_adjacent" in road_ids
    assert "lane_far" not in road_ids
    marker_lane_classes = {marker["visibility_class"] for marker in payload["lane_markers"]}
    assert "background" not in marker_lane_classes


def test_world_layer_merges_junction_lanes_into_single_patch() -> None:
    projector = BEVDrivableProjector()
    route_lane = _static_lane("lane_route", x0=-10.0, x1=5.0, center_y=0.0)
    junction_a = _static_lane("junction_a", x0=0.0, x1=12.0, center_y=0.0, is_junction=True)
    junction_b = _static_lane("junction_b", x0=0.0, x1=12.0, center_y=2.5, is_junction=True)
    crossing = _static_lane("junction_cross", x0=-2.0, x1=10.0, center_y=6.0, is_junction=True)

    payload = projector.build_world_layer(
        _local_map([route_lane, junction_a, junction_b, crossing]),
        EgoPose(
            world_xyz=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            yaw_rad=0.0,
            speed_mps=0.0,
            acceleration_mps2=0.0,
            current_lane_id="lane_route",
            frenet_s=0.0,
            frenet_d=0.0,
            heading_error_rad=0.0,
        ),
        route_lane_ids={"lane_route", "junction_a"},
    )

    junction_patches = [road for road in payload["roads"] if road["is_junction_patch"]]
    assert 1 <= len(junction_patches) <= 2
    assert len(junction_patches) < 3
    assert all(patch["visibility_class"] in {"route", "adjacent"} for patch in junction_patches)
    assert all(marker["visibility_class"] != "background" for marker in payload["lane_markers"])


def test_world_layer_merges_live_traffic_light_state_onto_stable_anchor() -> None:
    projector = BEVDrivableProjector()
    route_lane = _static_lane("lane_route", x0=-10.0, x1=30.0, center_y=0.0)
    stable_anchor = {
        "actor_id": 101,
        "world_xyz": [8.0, 1.5, 3.2],
        "yaw_deg": 90.0,
        "state": "RED",
    }
    live_detection = TrafficLightDetection(
        world_xyz=np.array([8.8, 1.2, 3.0], dtype=np.float32),
        state=TrafficLightState.GREEN,
        stop_line_distance_m=10.0,
        confidence=0.92,
    )

    payload = projector.build_world_layer(
        _local_map([route_lane]),
        EgoPose(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            yaw_rad=0.0,
            speed_mps=0.0,
            acceleration_mps2=0.0,
            current_lane_id="lane_route",
            frenet_s=0.0,
            frenet_d=0.0,
            heading_error_rad=0.0,
        ),
        route_lane_ids={"lane_route"},
        stable_traffic_lights=[stable_anchor],
        live_traffic_lights=[live_detection],
    )

    assert len(payload["traffic_lights"]) == 1
    signal = payload["traffic_lights"][0]
    assert signal["actor_id"] == 101
    assert signal["state"] == "GREEN"
    assert signal["visibility_class"] == "route"
    assert abs(signal["confidence"] - 0.92) < 1e-6


def test_prior_map_builds_full_static_geometry_and_route_polyline() -> None:
    projector = BEVDrivableProjector()
    lane_a = _static_lane("lane_a", x0=-10.0, x1=30.0, center_y=0.0)
    lane_b = _static_lane("lane_b", x0=-10.0, x1=30.0, center_y=3.5)
    lane_graph = type("LaneGraph", (), {"segments": {"lane_a": lane_a, "lane_b": lane_b}})()
    stable_anchor = {
        "actor_id": 101,
        "world_xyz": [8.0, 1.5, 3.2],
        "yaw_deg": 90.0,
        "state": "RED",
    }

    payload = projector.build_prior_map(
        lane_graph=lane_graph,
        map_name="Town01",
        route_plan=_route_plan(),
        stable_traffic_lights=[stable_anchor],
    )

    assert payload["map_name"] == "Town01"
    assert len(payload["roads"]) == 2
    assert len(payload["lane_markers"]) == 1
    assert len(payload["sidewalks"]) == 2
    assert len(payload["route_polyline_world"]) == len(_route_plan().waypoints)
    assert len(payload["traffic_lights"]) == 1


def test_prior_map_bounds_cover_route_and_static_geometry() -> None:
    projector = BEVDrivableProjector()
    lane = _static_lane("lane_001", x0=-25.0, x1=45.0, center_y=0.0)
    lane_graph = type("LaneGraph", (), {"segments": {"lane_001": lane}})()

    payload = projector.build_prior_map(
        lane_graph=lane_graph,
        map_name="Town02",
        route_plan=_route_plan(),
        stable_traffic_lights=[],
    )

    bounds = payload["bounds_world"]
    assert bounds["min_x"] <= -25.0
    assert bounds["max_x"] >= 45.0
    assert bounds["min_y"] < 0.0
    assert bounds["max_y"] > 0.0
