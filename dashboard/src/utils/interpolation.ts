import type { EgoPose } from "./types";

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function lerp(a: number, b: number, alpha: number): number {
  return a + (b - a) * alpha;
}

function lerpAngle(a: number, b: number, alpha: number): number {
  const delta = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + delta * alpha;
}

export function frameInterpolationAlpha(
  currentFrameReceivedAtMs: number,
  frameIntervalMs: number,
  nowMs: number,
): number {
  if (frameIntervalMs <= 0) return 1;
  return clamp01((nowMs - currentFrameReceivedAtMs) / frameIntervalMs);
}

export function interpolateEgoPose(
  previous: EgoPose | undefined,
  current: EgoPose | undefined,
  alpha: number,
): EgoPose | undefined {
  if (!current) return previous;
  if (!previous) return current;

  const safeAlpha = clamp01(alpha);
  const previousWorld = previous.world_xyz ?? [0, 0, 0];
  const currentWorld = current.world_xyz ?? [0, 0, 0];

  return {
    ...current,
    world_xyz: [
      lerp(previousWorld[0] ?? 0, currentWorld[0] ?? 0, safeAlpha),
      lerp(previousWorld[1] ?? 0, currentWorld[1] ?? 0, safeAlpha),
      lerp(previousWorld[2] ?? 0, currentWorld[2] ?? 0, safeAlpha),
    ],
    yaw_rad: lerpAngle(previous.yaw_rad, current.yaw_rad, safeAlpha),
    speed_mps: lerp(previous.speed_mps, current.speed_mps, safeAlpha),
    acceleration_mps2: lerp(previous.acceleration_mps2, current.acceleration_mps2, safeAlpha),
    frenet_s: lerp(previous.frenet_s, current.frenet_s, safeAlpha),
    frenet_d: lerp(previous.frenet_d, current.frenet_d, safeAlpha),
    heading_error_rad: lerpAngle(previous.heading_error_rad, current.heading_error_rad, safeAlpha),
  };
}
