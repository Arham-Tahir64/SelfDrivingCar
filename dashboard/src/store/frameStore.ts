import { create } from "zustand";
import type { PipelineFrame, WebSocketEnvelope } from "../utils/types";

interface FrameStore {
  currentFrame: PipelineFrame | null;
  previousFrame: PipelineFrame | null;
  currentFrameReceivedAtMs: number;
  frameIntervalMs: number;
  connected: boolean;
  paused: boolean;
  playbackSpeed: number;
  applyEnvelope: (envelope: WebSocketEnvelope) => void;
  setConnected: (connected: boolean) => void;
  togglePause: () => void;
  setPlaybackSpeed: (speed: number) => void;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function isObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function mergeTopicValue<T>(previousValue: T | undefined, nextValue: T): T {
  if (isObjectRecord(previousValue) && isObjectRecord(nextValue)) {
    return { ...previousValue, ...nextValue } as T;
  }
  return nextValue;
}

function mergeFrame(
  previousFrame: PipelineFrame | null,
  topics: Partial<PipelineFrame>,
  tickId: number,
  simTimeS: number,
): PipelineFrame {
  const base: PipelineFrame = previousFrame
    ? { ...previousFrame }
    : {
        tick_id: tickId,
        sim_time_s: simTimeS,
      };
  base.tick_id = tickId;
  base.sim_time_s = simTimeS;

  const mutableBase = base as unknown as Record<string, unknown>;
  for (const [topic, value] of Object.entries(topics) as [keyof PipelineFrame, PipelineFrame[keyof PipelineFrame]][]) {
    if (value === undefined) continue;
    mutableBase[String(topic)] = mergeTopicValue(mutableBase[String(topic)], value);
  }
  return base;
}

export const useFrameStore = create<FrameStore>((set) => ({
  currentFrame: null,
  previousFrame: null,
  currentFrameReceivedAtMs: 0,
  frameIntervalMs: 50,
  connected: false,
  paused: false,
  playbackSpeed: 1.0,
  togglePause: () => set((state) => ({ paused: !state.paused })),
  setPlaybackSpeed: (speed: number) => set({ playbackSpeed: speed }),
  applyEnvelope: (envelope) =>
    set((state) => {
      // When paused, only accept bootstrap/static updates (scene setup), skip dynamic frames
      if (state.paused && envelope.message_kind === "dynamic_frame") {
        return {};
      }

      if (envelope.message_kind === "static_update") {
        if (state.currentFrame == null) {
          const bootFrame = mergeFrame(null, envelope.topics, envelope.tick_id, envelope.sim_time_s);
          return {
            previousFrame: bootFrame,
            currentFrame: bootFrame,
            currentFrameReceivedAtMs: state.currentFrameReceivedAtMs,
            frameIntervalMs: state.frameIntervalMs,
          };
        }
        return {
          previousFrame: state.previousFrame,
          currentFrame: mergeFrame(
            state.currentFrame,
            envelope.topics,
            state.currentFrame.tick_id,
            state.currentFrame.sim_time_s,
          ),
          currentFrameReceivedAtMs: state.currentFrameReceivedAtMs,
          frameIntervalMs: state.frameIntervalMs,
        };
      }

      if (envelope.message_kind === "bootstrap" && state.currentFrame != null) {
        return {
          previousFrame: state.previousFrame,
          currentFrame: mergeFrame(
            state.currentFrame,
            envelope.topics,
            state.currentFrame.tick_id,
            state.currentFrame.sim_time_s,
          ),
          currentFrameReceivedAtMs: state.currentFrameReceivedAtMs,
          frameIntervalMs: state.frameIntervalMs,
        };
      }

      const mergedFrame = mergeFrame(
        state.currentFrame,
        envelope.topics,
        envelope.tick_id,
        envelope.sim_time_s,
      );
      return {
        previousFrame: state.currentFrame ?? mergedFrame,
        currentFrame: mergedFrame,
        currentFrameReceivedAtMs: nowMs(),
        frameIntervalMs:
          state.currentFrame != null
            ? Math.max(
                16,
                Math.round((envelope.sim_time_s - state.currentFrame.sim_time_s) * 1000),
              )
            : state.frameIntervalMs,
      };
    }),
  setConnected: (connected) => set({ connected }),
}));
