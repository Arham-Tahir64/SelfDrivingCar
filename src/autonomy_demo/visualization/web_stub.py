from __future__ import annotations


class WebDashboardStub:
    """TODO(PRD 6 / 8): replace with FastAPI websocket stream surface."""

    def publish(self, snapshot) -> dict[str, bool]:
        return {"published": bool(snapshot)}

