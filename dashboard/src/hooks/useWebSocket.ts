import { useEffect, useRef } from "react";
import { useFrameStore } from "../store/frameStore";
import type { WebSocketEnvelope } from "../utils/types";

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;

function resolveWebSocketUrl(): string {
  const pageHost = window.location.hostname;
  const host =
    !pageHost || pageHost === "0.0.0.0" || pageHost === "[::]"
      ? "127.0.0.1"
      : pageHost;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${host}:8765/ws`;
}

export function useWebSocket() {
  const applyEnvelope = useFrameStore((s) => s.applyEnvelope);
  const setConnected = useFrameStore((s) => s.setConnected);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const lastAppliedRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout>;

    function connect() {
      if (cancelled) return;
      const ws = new WebSocket(resolveWebSocketUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        retriesRef.current = 0;
        setConnected(true);
      };

      ws.onmessage = (ev) => {
        try {
          const envelope: WebSocketEnvelope = JSON.parse(ev.data);
          const speed = useFrameStore.getState().playbackSpeed;

          // For non-1x speeds, throttle dynamic frame application
          if (envelope.message_kind === "dynamic_frame" && speed !== 1.0) {
            const now = performance.now();
            const minInterval = 50 / Math.max(speed, 0.1); // base 50ms interval scaled
            if (now - lastAppliedRef.current < minInterval) return;
            lastAppliedRef.current = now;
          }

          applyEnvelope(envelope);
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
  }, [applyEnvelope, setConnected]);
}
