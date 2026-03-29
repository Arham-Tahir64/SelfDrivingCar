from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from autonomy_demo.common.logging import get_logger
from autonomy_demo.visualization.websocket_bridge import WebSocketBridge

logger = get_logger(__name__)

_DASHBOARD_DIST = Path(__file__).resolve().parents[3] / "dashboard" / "dist"


def create_app(bridge: WebSocketBridge):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Autonomy Demo Dashboard")

    @app.get("/api/status")
    async def status():
        return JSONResponse({"connected_clients": bridge.client_count})

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        bridge.register(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            bridge.unregister(ws)

    if _DASHBOARD_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(_DASHBOARD_DIST), html=True), name="static")

    return app


def start_server_thread(
    bridge: WebSocketBridge,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> threading.Thread:
    """Run the FastAPI server in a daemon thread with its own event loop."""

    def _run() -> None:
        import uvicorn

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bridge.set_event_loop(loop)

        app = create_app(bridge)
        config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True, name="ws-server")
    thread.start()
    logger.info("WebSocket server started on ws://%s:%d/ws", host, port)
    return thread
