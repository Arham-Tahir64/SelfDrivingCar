import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { frameInterpolationAlpha, interpolateEgoPose } from "../utils/interpolation";
import { worldToScene, yawToScene, dampAngle } from "../utils/scene";

const GRID_SIZE = 100;
const CELL_SIZE_M = 0.5;
const PLANE_SIZE = GRID_SIZE * CELL_SIZE_M; // 50 m

// Green-tinted colour for drivable area
const DRIVABLE_R = 0;
const DRIVABLE_G = 220;
const DRIVABLE_B = 90;
const MAX_ALPHA = 140; // semi-transparent

const POSITION_SMOOTHING = 0.22;
const ROTATION_SMOOTHING = 0.18;

function decodeBase64(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export default function DrivableSurface() {
  const meshRef = useRef<THREE.Mesh>(null);
  const textureRef = useRef<THREE.DataTexture | null>(null);
  const lastTickRef = useRef<number | null>(null);
  const targetPosition = useRef(new THREE.Vector3());
  const initialized = useRef(false);

  useFrame(() => {
    const state = useFrameStore.getState();
    const frame = state.currentFrame;
    const mesh = meshRef.current;
    if (!mesh || !frame) return;

    // Interpolate ego pose for smooth movement
    const alpha = frameInterpolationAlpha(
      state.currentFrameReceivedAtMs,
      state.frameIntervalMs,
      performance.now(),
    );
    const ego = interpolateEgoPose(
      state.previousFrame?.["localization/ego_pose"],
      state.currentFrame?.["localization/ego_pose"],
      alpha,
    );

    if (ego) {
      // Position: ego centre, offset forward by half the grid (grid goes 50m ahead of ego)
      const egoScene = worldToScene(ego.world_xyz);
      const yaw = yawToScene(ego.yaw_rad);

      // The grid centre is PLANE_SIZE/2 ahead of ego in the ego-forward direction
      const forwardOffset = PLANE_SIZE / 2;
      const offsetX = Math.cos(-yaw) * forwardOffset;
      const offsetZ = Math.sin(-yaw) * forwardOffset;

      targetPosition.current.set(
        egoScene.x + offsetX,
        0.02, // just above ground
        egoScene.z + offsetZ,
      );

      if (!initialized.current) {
        mesh.position.copy(targetPosition.current);
        mesh.rotation.y = yaw;
        initialized.current = true;
      } else {
        mesh.position.lerp(targetPosition.current, POSITION_SMOOTHING);
        mesh.rotation.y = dampAngle(mesh.rotation.y, yaw, ROTATION_SMOOTHING);
      }
    }

    // Update texture only on new ticks
    const bevData = frame["visualization/bev_drivable"];
    if (!bevData || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    const grid = decodeBase64(bevData.grid_b64);
    const rows = bevData.rows;
    const cols = bevData.cols;

    // Create RGBA texture data
    const rgba = new Uint8Array(rows * cols * 4);
    for (let r = 0; r < rows; r++) {
      // Flip vertically: texture row 0 is bottom, but grid row 0 is far ahead
      const srcRow = rows - 1 - r;
      for (let c = 0; c < cols; c++) {
        const confidence = grid[srcRow * cols + c];
        const idx = (r * cols + c) * 4;
        rgba[idx] = DRIVABLE_R;
        rgba[idx + 1] = DRIVABLE_G;
        rgba[idx + 2] = DRIVABLE_B;
        rgba[idx + 3] = confidence > 20 ? Math.min(confidence, MAX_ALPHA) : 0;
      }
    }

    if (!textureRef.current) {
      textureRef.current = new THREE.DataTexture(
        rgba,
        cols,
        rows,
        THREE.RGBAFormat,
      );
      textureRef.current.magFilter = THREE.NearestFilter;
      textureRef.current.minFilter = THREE.NearestFilter;
      textureRef.current.needsUpdate = true;

      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.map = textureRef.current;
      mat.needsUpdate = true;
    } else {
      textureRef.current.image.data.set(rgba);
      textureRef.current.needsUpdate = true;
    }
  });

  return (
    <mesh ref={meshRef} rotation-x={-Math.PI / 2}>
      <planeGeometry args={[PLANE_SIZE, PLANE_SIZE]} />
      <meshBasicMaterial transparent depthWrite={false} side={THREE.DoubleSide} />
    </mesh>
  );
}
