from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from autonomy_demo.common.logging import get_logger
from autonomy_demo.visualization.websocket_bridge import WebSocketBridge

logger = get_logger(__name__)

_DASHBOARD_DIST = Path(__file__).resolve().parents[3] / "dashboard" / "dist"


def create_app(bridge: WebSocketBridge):
    app = FastAPI(title="Autonomy Demo Dashboard")

    @app.get("/api/status")
    async def status():
        return JSONResponse({"connected_clients": bridge.client_count})

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        bridge.register(ws)
        logger.info("Dashboard websocket accepted (%d clients)", bridge.client_count)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            bridge.unregister(ws)

    if _DASHBOARD_DIST.is_dir():
        assets_dir = _DASHBOARD_DIST / "assets"
        if assets_dir.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/")
        async def dashboard_index():
            return FileResponse(_DASHBOARD_DIST / "index.html")

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
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    logger.info(
        "WebSocket server bound to %s:%d; open http://%s:%d/ and connect to ws://%s:%d/ws",
        host,
        port,
        browser_host,
        port,
        browser_host,
        port,
    )
    return thread
