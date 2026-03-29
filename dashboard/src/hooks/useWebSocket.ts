import { useEffect, useRef } from "react";
import { useFrameStore } from "../store/frameStore";
import type { PipelineFrame } from "../utils/types";

const WS_URL = `ws://${window.location.hostname}:8765/ws`;
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;

export function useWebSocket() {
  const pushFrame = useFrameStore((s) => s.pushFrame);
  const setConnected = useFrameStore((s) => s.setConnected);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout>;

    function connect() {
      if (cancelled) return;
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        retriesRef.current = 0;
        setConnected(true);
      };

      ws.onmessage = (ev) => {
        try {
          const frame: PipelineFrame = JSON.parse(ev.data);
          pushFrame(frame);
        } catch {
          // skip malformed messages
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          const delay = Math.min(
            RECONNECT_BASE_MS * 2 ** retriesRef.current,
            RECONNECT_MAX_MS,
          );
          retriesRef.current++;
          timeout = setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        ws.close();
      };
    }

    connect();

    return () => {
      cancelled = true;
      clearTimeout(timeout);
      wsRef.current?.close();
    };
  }, [pushFrame, setConnected]);
}
