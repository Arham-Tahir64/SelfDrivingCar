import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { classColor, classOpacity, modalityColor } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { dampAngle, worldToScene, yawToScene } from "../utils/scene";
import { buildAgentProxy } from "./agentProxies";
import { isVehicleLikeClass, pickVehicleVariant, type VehicleVariant } from "./vehicleMeshes";

const DEFAULT_SIZE = new THREE.Vector3(4.5, 1.6, 2.0);
const POSITION_SMOOTHING = 0.24;
const ROTATION_SMOOTHING = 0.18;
const SCALE_SMOOTHING = 0.22;
const MIN_HEADING_SPEED_MPS = 0.75;

function bboxCentroid(bbox: number[][]): THREE.Vector3 {
  if (!bbox || bbox.length === 0) return new THREE.Vector3();
  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (const corner of bbox) {
    sx += corner[0] ?? 0;
    sy += corner[1] ?? 0;
    sz += corner[2] ?? 0;
  }
  const n = bbox.length;
  return new THREE.Vector3(sx / n, sy / n, sz / n);
}

function bboxSize(bbox: number[][]): THREE.Vector3 {
  if (!bbox || bbox.length < 2) return DEFAULT_SIZE.clone();
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const c of bbox) {
    minX = Math.min(minX, c[0] ?? 0);
    maxX = Math.max(maxX, c[0] ?? 0);
    minY = Math.min(minY, c[1] ?? 0);
    maxY = Math.max(maxY, c[1] ?? 0);
    minZ = Math.min(minZ, c[2] ?? 0);
    maxZ = Math.max(maxZ, c[2] ?? 0);
  }
  const dx = Math.max(maxX - minX, 0.5);
  const dy = Math.max(maxY - minY, 0.5);
  const dz = Math.max(maxZ - minZ, 0.5);
  return new THREE.Vector3(dx, dy, dz);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function isHeadingComparable(a: number | undefined, b: number | undefined): a is number {
  return typeof a === "number" && typeof b === "number";
}

function chooseAlignedYaw(candidate: number, reference?: number): number {
  if (reference === undefined) return candidate;
  const opposite = candidate + Math.PI;
  const delta = Math.abs(Math.atan2(Math.sin(candidate - reference), Math.cos(candidate - reference)));
  const oppositeDelta = Math.abs(Math.atan2(Math.sin(opposite - reference), Math.cos(opposite - reference)));
  return oppositeDelta < delta ? opposite : candidate;
}

function velocityHeading(velocity: number[] | undefined): number | undefined {
  if (!velocity || velocity.length < 2) return undefined;
  const vx = velocity[0] ?? 0;
  const vy = velocity[1] ?? 0;
  if (Math.hypot(vx, vy) < MIN_HEADING_SPEED_MPS) return undefined;
  return yawToScene(Math.atan2(vy, vx));
}

function bboxHeading(bbox: number[][], reference?: number): number | undefined {
  if (!bbox || bbox.length < 4) return undefined;
  let bestVector: [number, number] | null = null;
  let bestLength = 0;
  const footprint = bbox.slice(0, 4);
  for (let index = 0; index < footprint.length; index += 1) {
    const start = footprint[index];
    const end = footprint[(index + 1) % footprint.length];
    const dx = (end[0] ?? 0) - (start[0] ?? 0);
    const dy = (end[1] ?? 0) - (start[1] ?? 0);
    const length = Math.hypot(dx, dy);
    if (length > bestLength) {
      bestLength = length;
      bestVector = [dx, dy];
    }
  }
  if (!bestVector || bestLength < 0.75) return undefined;
  const candidate = yawToScene(Math.atan2(bestVector[1], bestVector[0]));
  return chooseAlignedYaw(candidate, reference);
}

function inferVehicleYaw(
  velocity: number[] | undefined,
  bbox: number[][],
  previousYaw?: number,
): number {
  const motionYaw = velocityHeading(velocity);
  if (motionYaw !== undefined) {
    return chooseAlignedYaw(motionYaw, previousYaw);
  }
  const footprintYaw = bboxHeading(bbox, previousYaw);
  if (footprintYaw !== undefined) {
    return footprintYaw;
  }
  return previousYaw ?? 0;
}

function displayScaleForDetection(size: THREE.Vector3, objectClass: string): THREE.Vector3 {
  if (isVehicleLikeClass(objectClass)) {
    return new THREE.Vector3(
      clamp(size.x, 3.4, 7.6),
      clamp(size.y, 1.2, 2.8),
      clamp(size.z, 1.5, 2.8),
    );
  }
  return new THREE.Vector3(
    Math.max(size.x, 0.6),
    Math.max(size.y, 0.6),
    Math.max(size.z, 0.6),
  );
}

interface PooledProxy {
  group: THREE.Group;
  objectClass: string;
  modality: string;
  variant?: VehicleVariant;
  targetPosition: THREE.Vector3;
  targetScale: THREE.Vector3;
  targetYaw: number;
  initialized: boolean;
}

export default function DetectedAgents() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);
  const poolRef = useRef<Map<number, PooledProxy>>(new Map());

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;

    if (frame && lastTickRef.current !== frame.tick_id) {
      lastTickRef.current = frame.tick_id;
      const detections = frame["perception/detections"];
      const activeIds = new Set<number>();

      if (detections) {
        for (const det of detections) {
          if (det.track_state === "DELETED" || det.track_state === "LOST") continue;

          const trackId = det.track_id;
          activeIds.add(trackId);
          const existing = poolRef.current.get(trackId);
          const centroid = bboxCentroid(det.world_bbox_3d);
          const size = bboxSize(det.world_bbox_3d);
          const sceneCenter = worldToScene([centroid.x, centroid.y, centroid.z]);
          const displaySize = displayScaleForDetection(size, det.object_class);
          const variant = isVehicleLikeClass(det.object_class)
            ? pickVehicleVariant(trackId, det.object_class)
            : undefined;
          const targetYaw = isVehicleLikeClass(det.object_class)
            ? inferVehicleYaw(det.velocity, det.world_bbox_3d, existing?.targetYaw)
            : existing?.targetYaw ?? 0;

          const classChanged = existing && existing.objectClass !== det.object_class;
          const modalityChanged = existing && existing.modality !== det.source_modality;
          const variantChanged = existing && existing.variant !== variant;

          if (existing && !classChanged && !modalityChanged && !variantChanged) {
            existing.targetPosition.set(sceneCenter.x, 0.02, sceneCenter.z);
            existing.targetScale.copy(displaySize);
            existing.targetYaw = targetYaw;
          } else {
            if (existing) {
              group.remove(existing.group);
              disposeObject3D(existing.group);
              poolRef.current.delete(trackId);
            }

            const color = classColor(det.object_class);
            const opacity = classOpacity(det.object_class);
            const outlineColor = modalityColor(det.source_modality);
            const proxy = buildAgentProxy(det.object_class, color, outlineColor, opacity, {
              vehicleVariant: variant,
            });
            proxy.position.set(sceneCenter.x, 0.02, sceneCenter.z);
            proxy.scale.copy(displaySize);
            proxy.rotation.y = targetYaw;
            group.add(proxy);
            poolRef.current.set(trackId, {
              group: proxy,
              objectClass: det.object_class,
              modality: det.source_modality,
              variant,
              targetPosition: new THREE.Vector3(sceneCenter.x, 0.02, sceneCenter.z),
              targetScale: displaySize.clone(),
              targetYaw,
              initialized: true,
            });
          }
        }
      }

      for (const [trackId, pooled] of poolRef.current) {
        if (!activeIds.has(trackId)) {
          group.remove(pooled.group);
          disposeObject3D(pooled.group);
          poolRef.current.delete(trackId);
        }
      }
    }

    for (const pooled of poolRef.current.values()) {
      if (!pooled.initialized) {
        pooled.group.position.copy(pooled.targetPosition);
        pooled.group.scale.copy(pooled.targetScale);
        pooled.group.rotation.y = pooled.targetYaw;
        pooled.initialized = true;
        continue;
      }

      pooled.group.position.lerp(pooled.targetPosition, POSITION_SMOOTHING);
      pooled.group.scale.lerp(pooled.targetScale, SCALE_SMOOTHING);
      pooled.group.rotation.y = dampAngle(
        pooled.group.rotation.y,
        pooled.targetYaw,
        ROTATION_SMOOTHING,
      );
    }
  });

  return <group ref={groupRef} />;
}
