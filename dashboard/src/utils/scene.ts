import * as THREE from "three";

export function worldToScene(worldXYZ: number[]): THREE.Vector3 {
  return new THREE.Vector3(worldXYZ[0] ?? 0, 0, -(worldXYZ[1] ?? 0));
}

export function yawToScene(yawRad: number): number {
  return -yawRad;
}

export function dampAngle(
  current: number,
  target: number,
  alpha: number,
): number {
  const delta = Math.atan2(
    Math.sin(target - current),
    Math.cos(target - current),
  );
  return current + delta * alpha;
}
