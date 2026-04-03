import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { frameInterpolationAlpha, interpolateEgoPose } from "../utils/interpolation";
import { worldToScene, yawToScene, dampAngle } from "../utils/scene";

// Green-tinted colour for drivable area
const DRIVABLE_R = 0;
const DRIVABLE_G = 220;
const DRIVABLE_B = 90;
const MAX_ALPHA = 70; // keep the learned signal subtle over the structured road layer

const POSITION_SMOOTHING = 0.22;
const ROTATION_SMOOTHING = 0.18;
const GROUND_EPSILON = 0.02;

function decodeBase64(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function buildDrivableTexture(
  grid: Uint8Array,
  rows: number,
  cols: number,
): Uint8Array {
  // Texture contract:
  // - texture width = forward distance (near -> far)
  // - texture height = lateral position (left -> right)
  // - grid rows are far -> near, so reverse them into texture x
  // - grid cols are left -> right, so mirror them to compensate for the
  //   plane's rotated local-y -> world-z convention in the dashboard scene
  const rgba = new Uint8Array(rows * cols * 4);
  for (let row = 0; row < rows; row++) {
    const texX = rows - 1 - row;
    for (let col = 0; col < cols; col++) {
      const texY = cols - 1 - col;
      const confidence = grid[(row * cols) + col];
      const idx = ((texY * rows) + texX) * 4;
      rgba[idx] = DRIVABLE_R;
      rgba[idx + 1] = DRIVABLE_G;
      rgba[idx + 2] = DRIVABLE_B;
      rgba[idx + 3] = confidence > 20 ? Math.min(confidence, MAX_ALPHA) : 0;
    }
  }
  return rgba;
}

export default function DrivableSurface() {
  const groupRef = useRef<THREE.Group>(null);
  const meshRef = useRef<THREE.Mesh>(null);
  const textureRef = useRef<THREE.DataTexture | null>(null);
  const lastTickRef = useRef<number | null>(null);
  const targetPosition = useRef(new THREE.Vector3());
  const initialized = useRef(false);

  useFrame(() => {
    const state = useFrameStore.getState();
    const frame = state.currentFrame;
    const mesh = meshRef.current;
    const group = groupRef.current;
    if (!mesh || !group || !frame) return;

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
      const egoScene = worldToScene(ego.world_xyz);
      const yaw = yawToScene(ego.yaw_rad);

      targetPosition.current.set(
        egoScene.x,
        0.0,
        egoScene.z,
      );

      if (!initialized.current) {
        group.position.copy(targetPosition.current);
        group.rotation.y = yaw;
        initialized.current = true;
      } else {
        group.position.lerp(targetPosition.current, POSITION_SMOOTHING);
        group.rotation.y = dampAngle(group.rotation.y, yaw, ROTATION_SMOOTHING);
      }
    }

    // Update texture only on new ticks
    const bevData = frame["visualization/bev_drivable"];
    if (!bevData || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    const grid = decodeBase64(bevData.grid_b64);
    const rows = bevData.rows;
    const cols = bevData.cols;
    const xMin = bevData.x_min_m;
    const xMax = bevData.x_max_m;
    const yMin = bevData.y_min_m;
    const yMax = bevData.y_max_m;
    const forwardRange = xMax - xMin;
    const lateralWidth = yMax - yMin;

    mesh.position.set((xMin + xMax) * 0.5, GROUND_EPSILON, (yMin + yMax) * 0.5);
    mesh.scale.set(forwardRange, lateralWidth, 1.0);

    const rgba = buildDrivableTexture(grid, rows, cols);
    const textureBytes = rgba as unknown as ArrayBufferView<ArrayBuffer>;

    if (
      !textureRef.current
      || textureRef.current.image.width !== rows
      || textureRef.current.image.height !== cols
    ) {
      textureRef.current = new THREE.DataTexture(
        textureBytes,
        rows,
        cols,
        THREE.RGBAFormat,
      );
      textureRef.current.magFilter = THREE.NearestFilter;
      textureRef.current.minFilter = THREE.NearestFilter;
      textureRef.current.flipY = false;
      textureRef.current.needsUpdate = true;

      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.map = textureRef.current;
      mat.needsUpdate = true;
    } else {
      (textureRef.current.image.data as Uint8Array).set(rgba);
      textureRef.current.needsUpdate = true;
    }
  });

  return (
    <group ref={groupRef}>
      <mesh ref={meshRef} rotation-x={-Math.PI / 2}>
        <planeGeometry args={[1, 1]} />
        <meshBasicMaterial transparent depthWrite={false} side={THREE.FrontSide} />
      </mesh>
    </group>
  );
}
