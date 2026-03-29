import { useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import EgoVehicle from "./EgoVehicle";
import LaneLines from "./LaneLines";
import DetectedAgents from "./DetectedAgents";
import PlannedTrajectory from "./PlannedTrajectory";
import PredictedPaths from "./PredictedPaths";
import Cones from "./Cones";

// Camera offset: above and behind the ego vehicle (rear-follow BEV)
const CAM_OFFSET = new THREE.Vector3(0, 80, 45);
const CAM_LOOK_AHEAD = new THREE.Vector3(0, 0, -20);
const LERP_SPEED = 0.06;

function CameraController() {
  const { camera } = useThree();
  const targetPos = useRef(new THREE.Vector3(0, 80, 45));
  const targetLook = useRef(new THREE.Vector3(0, 0, 0));

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const ego = frame?.["localization/ego_pose"];

    if (ego) {
      const [x, , z] = ego.world_xyz;
      const egoPos = new THREE.Vector3(x, 0, -z);

      // Rotate offset by ego yaw
      const yaw = -ego.yaw_rad;
      const rotatedOffset = CAM_OFFSET.clone().applyAxisAngle(
        new THREE.Vector3(0, 1, 0),
        yaw,
      );
      const rotatedLookAhead = CAM_LOOK_AHEAD.clone().applyAxisAngle(
        new THREE.Vector3(0, 1, 0),
        yaw,
      );

      targetPos.current.copy(egoPos).add(rotatedOffset);
      targetLook.current.copy(egoPos).add(rotatedLookAhead);
    }

    camera.position.lerp(targetPos.current, LERP_SPEED);
    const currentLook = new THREE.Vector3();
    camera.getWorldDirection(currentLook);
    const smoothLook = new THREE.Vector3()
      .copy(camera.position)
      .add(currentLook);
    smoothLook.lerp(targetLook.current, LERP_SPEED);
    camera.lookAt(targetLook.current);
  });

  return null;
}

function Ground() {
  return (
    <>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.01} receiveShadow>
        <planeGeometry args={[2000, 2000]} />
        <meshStandardMaterial color="#1a1a24" />
      </mesh>
      <gridHelper
        args={[2000, 400, "#2a2a3a", "#1f1f2e"]}
        position-y={0.0}
      />
    </>
  );
}

export default function BEVScene() {
  return (
    <>
      <color attach="background" args={["#0a0a0f"]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[50, 100, 50]} intensity={0.8} />
      <CameraController />
      <Ground />
      <EgoVehicle />
      <LaneLines />
      <DetectedAgents />
      <PlannedTrajectory />
      <PredictedPaths />
      <Cones />
    </>
  );
}
