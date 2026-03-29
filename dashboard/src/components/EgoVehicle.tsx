import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";

const EGO_LENGTH = 4.5;
const EGO_WIDTH = 2.0;
const EGO_HEIGHT = 1.5;
const CYAN = new THREE.Color("#00E5FF");

export default function EgoVehicle() {
  const meshRef = useRef<THREE.Mesh>(null);
  const edgesRef = useRef<THREE.LineSegments>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const ego = frame?.["localization/ego_pose"];
    if (!ego || !meshRef.current || !edgesRef.current) return;

    const [x, , z] = ego.world_xyz;
    // CARLA uses left-handed Y-up; Three.js uses right-handed Y-up.
    // Map CARLA (x, z) → Three.js (x, z), y is up.
    meshRef.current.position.set(x, EGO_HEIGHT / 2, -z);
    meshRef.current.rotation.y = -ego.yaw_rad;

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
