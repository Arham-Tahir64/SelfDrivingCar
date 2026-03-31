import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { frameInterpolationAlpha, interpolateEgoPose } from "../utils/interpolation";
import { dampAngle, worldToScene, yawToScene } from "../utils/scene";

const BODY_COLOR = new THREE.Color("#f5f7fa");
const CABIN_COLOR = new THREE.Color("#111827");
const GLASS_COLOR = new THREE.Color("#0f172a");
const LIGHT_COLOR = new THREE.Color("#00e5ff");
const TAIL_LIGHT_COLOR = new THREE.Color("#ff4d5a");
const WHEEL_COLOR = new THREE.Color("#0b0d12");

const BODY_SHAPE = (() => {
  const shape = new THREE.Shape();
  shape.moveTo(-2.45, -0.78);
  shape.quadraticCurveTo(-2.55, -0.78, -2.55, -0.62);
  shape.lineTo(-2.55, 0.62);
  shape.quadraticCurveTo(-2.55, 0.78, -2.45, 0.78);
  shape.lineTo(-2.05, 0.96);
  shape.lineTo(-1.35, 1.02);
  shape.lineTo(-0.55, 1.08);
  shape.lineTo(0.85, 1.08);
  shape.lineTo(1.6, 1.0);
  shape.lineTo(2.1, 0.86);
  shape.quadraticCurveTo(2.45, 0.72, 2.55, 0.45);
  shape.lineTo(2.55, -0.45);
  shape.quadraticCurveTo(2.45, -0.72, 2.1, -0.86);
  shape.lineTo(1.6, -1.0);
  shape.lineTo(0.85, -1.08);
  shape.lineTo(-0.55, -1.08);
  shape.lineTo(-1.35, -1.02);
  shape.lineTo(-2.05, -0.96);
  shape.quadraticCurveTo(-2.45, -0.9, -2.45, -0.78);
  return shape;
})();

const BODY_GEOMETRY = new THREE.ShapeGeometry(BODY_SHAPE);
const CABIN_GEOMETRY = new THREE.BoxGeometry(2.35, 0.42, 1.42);
const ROOF_GEOMETRY = new THREE.BoxGeometry(1.55, 0.18, 1.18);
const BUMPER_GEOMETRY = new THREE.BoxGeometry(0.42, 0.18, 1.78);
const LIGHT_BAR_GEOMETRY = new THREE.BoxGeometry(0.32, 0.08, 0.18);
const WHEEL_GEOMETRY = new THREE.CylinderGeometry(0.23, 0.23, 0.18, 14);

const POSITION_SMOOTHING = 0.22;
const ROTATION_SMOOTHING = 0.18;

function BodyShell() {
  return (
    <mesh geometry={BODY_GEOMETRY} rotation-x={-Math.PI / 2} position-y={0.06}>
      <meshStandardMaterial
        color={BODY_COLOR}
        roughness={0.45}
        metalness={0.08}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function Wheel({
  position,
}: {
  position: [number, number, number];
}) {
  return (
    <mesh geometry={WHEEL_GEOMETRY} position={position}>
      <meshStandardMaterial
        color={WHEEL_COLOR}
        roughness={0.95}
        metalness={0.02}
      />
    </mesh>
  );
}

export default function EgoVehicle() {
  const groupRef = useRef<THREE.Group>(null);
  const targetPosition = useRef(new THREE.Vector3());
  const initialized = useRef(false);

  useFrame(() => {
    const state = useFrameStore.getState();
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
    if (!ego || !groupRef.current) return;

    targetPosition.current.copy(worldToScene(ego.world_xyz));

    if (!initialized.current) {
      groupRef.current.position.copy(targetPosition.current);
      groupRef.current.rotation.y = yawToScene(ego.yaw_rad);
      initialized.current = true;
    } else {
      groupRef.current.position.lerp(targetPosition.current, POSITION_SMOOTHING);
      groupRef.current.rotation.y = dampAngle(
        groupRef.current.rotation.y,
        yawToScene(ego.yaw_rad),
        ROTATION_SMOOTHING,
      );
    }
  });

  return (
    <group ref={groupRef}>
      <group position={[0, 0, 0]}>
        <BodyShell />

        <mesh position={[0.05, 0.45, 0]} geometry={CABIN_GEOMETRY}>
          <meshStandardMaterial
            color={CABIN_COLOR}
            roughness={0.25}
            metalness={0.15}
          />
        </mesh>

        <mesh position={[0.08, 0.62, 0]} geometry={ROOF_GEOMETRY}>
          <meshStandardMaterial
            color={GLASS_COLOR}
            roughness={0.15}
            metalness={0.2}
            transparent
            opacity={0.92}
          />
        </mesh>

        <mesh position={[2.22, 0.17, 0]} geometry={BUMPER_GEOMETRY}>
          <meshStandardMaterial
            color={BODY_COLOR}
            roughness={0.38}
            metalness={0.08}
          />
        </mesh>

        <mesh position={[-2.22, 0.17, 0]} geometry={BUMPER_GEOMETRY}>
          <meshStandardMaterial
            color={BODY_COLOR}
            roughness={0.38}
            metalness={0.08}
          />
        </mesh>

        <mesh position={[2.34, 0.22, 0.58]} geometry={LIGHT_BAR_GEOMETRY}>
          <meshStandardMaterial color={LIGHT_COLOR} emissive={LIGHT_COLOR} emissiveIntensity={0.65} />
        </mesh>
        <mesh position={[2.34, 0.22, -0.58]} geometry={LIGHT_BAR_GEOMETRY}>
          <meshStandardMaterial color={LIGHT_COLOR} emissive={LIGHT_COLOR} emissiveIntensity={0.65} />
        </mesh>
        <mesh position={[-2.34, 0.22, 0.58]} geometry={LIGHT_BAR_GEOMETRY}>
          <meshStandardMaterial color={TAIL_LIGHT_COLOR} emissive={TAIL_LIGHT_COLOR} emissiveIntensity={0.8} />
        </mesh>
        <mesh position={[-2.34, 0.22, -0.58]} geometry={LIGHT_BAR_GEOMETRY}>
          <meshStandardMaterial color={TAIL_LIGHT_COLOR} emissive={TAIL_LIGHT_COLOR} emissiveIntensity={0.8} />
        </mesh>

        <Wheel position={[1.72, 0.18, 0.95]} />
        <Wheel position={[1.72, 0.18, -0.95]} />
        <Wheel position={[-1.72, 0.18, 0.95]} />
        <Wheel position={[-1.72, 0.18, -0.95]} />
      </group>
    </group>
  );
}
