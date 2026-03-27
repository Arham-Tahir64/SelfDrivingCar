from __future__ import annotations

from autonomy_demo.interfaces.enums import BehaviorState
from autonomy_demo.interfaces.types import EgoPose, LocalMap


class StubBehaviorPlanner:
    """TODO(PRD 3.2.7): replace with hierarchical FSM and scenario-aware transitions."""

    def run(self, local_map: LocalMap, ego_pose: EgoPose) -> BehaviorState:
        if local_map.closed_lanes:
            return BehaviorState.PREPARE_MERGE
        return BehaviorState.LANE_KEEP

