import { create } from "zustand";
import type { PipelineFrame } from "../utils/types";

interface FrameStore {
  currentFrame: PipelineFrame | null;
  previousFrame: PipelineFrame | null;
  connected: boolean;
  pushFrame: (frame: PipelineFrame) => void;
  setConnected: (connected: boolean) => void;
}

export const useFrameStore = create<FrameStore>((set) => ({
  currentFrame: null,
  previousFrame: null,
  connected: false,
  pushFrame: (frame) =>
    set((state) => ({
      previousFrame: state.currentFrame,
      currentFrame: frame,
    })),
  setConnected: (connected) => set({ connected }),
}));
