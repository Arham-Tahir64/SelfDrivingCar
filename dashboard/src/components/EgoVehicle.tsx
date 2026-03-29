import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { dampAngle, worldToScene, yawToScene } from "../utils/scene";

const EGO_LENGTH = 4.5;
const EGO_WIDTH = 2.0;
const EGO_HEIGHT = 1.5;
const CYAN = new THREE.Color("#00E5FF");
const POSITION_SMOOTHING = 0.22;
const ROTATION_SMOOTHING = 0.18;

export default function EgoVehicle() {
  const meshRef = useRef<THREE.Mesh>(null);
  const edgesRef = useRef<THREE.LineSegments>(null);
  const targetPosition = useRef(new THREE.Vector3());
  const initialized = useRef(false);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const ego = frame?.["localization/ego_pose"];
    if (!ego || !meshRef.current || !edgesRef.current) return;

    targetPosition.current.copy(worldToScene(ego.world_xyz));
    targetPosition.current.y = EGO_HEIGHT / 2;

    if (!initialized.current) {
      meshRef.current.position.copy(targetPosition.current);
      meshRef.current.rotation.y = yawToScene(ego.yaw_rad);
      initialized.current = true;
    } else {
      meshRef.current.position.lerp(targetPosition.current, POSITION_SMOOTHING);
      meshRef.current.rotation.y = dampAngle(
        meshRef.current.rotation.y,
        yawToScene(ego.yaw_rad),
        ROTATION_SMOOTHING,
      );
    }

    edgesRef.current.position.copy(meshRef.current.position);
    edgesRef.current.rotation.copy(meshRef.current.rotation);
  });

  const geometry = new THREE.BoxGeometry(EGO_LENGTH, EGO_HEIGHT, EGO_WIDTH);
  const edgesGeometry = new THREE.EdgesGeometry(geometry);

  return (
    <>
      <mesh ref={meshRef} geometry={geometry}>
        <meshStandardMaterial color="white" transparent opacity={0.85} />
      </mesh>
      <lineSegments ref={edgesRef} geometry={edgesGeometry}>
        <lineBasicMaterial color={CYAN} linewidth={2} />
      </lineSegments>
    </>
  );
}
