from __future__ import annotations


class PygameDashboardStub:
    """TODO(PRD 6): replace with bird's-eye, camera, LiDAR, and HUD rendering."""

    def render(self, snapshot) -> dict[str, int]:
        return {"topic_count": len(snapshot)}

