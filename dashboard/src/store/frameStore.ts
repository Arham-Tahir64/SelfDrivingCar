import { create } from "zustand";
import type { PipelineFrame } from "../utils/types";

interface FrameStore {
  currentFrame: PipelineFrame | null;
  previousFrame: PipelineFrame | null;
  currentFrameReceivedAtMs: number;
  frameIntervalMs: number;
  connected: boolean;
  pushFrame: (frame: PipelineFrame) => void;
  setConnected: (connected: boolean) => void;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

export const useFrameStore = create<FrameStore>((set) => ({
  currentFrame: null,
  previousFrame: null,
  currentFrameReceivedAtMs: 0,
  frameIntervalMs: 50,
  connected: false,
  pushFrame: (frame) =>
    set((state) => ({
      previousFrame: state.currentFrame,
      currentFrame: frame,
      currentFrameReceivedAtMs: nowMs(),
      frameIntervalMs:
        state.currentFrame != null
          ? Math.max(
              16,
              Math.round((frame.sim_time_s - state.currentFrame.sim_time_s) * 1000),
            )
          : state.frameIntervalMs,
    })),
  setConnected: (connected) => set({ connected }),
}));
