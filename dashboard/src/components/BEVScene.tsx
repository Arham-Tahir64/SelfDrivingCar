import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { frameInterpolationAlpha, interpolateEgoPose } from "../utils/interpolation";
import { worldToScene, yawToScene } from "../utils/scene";
import EgoVehicle from "./EgoVehicle";
import LaneLines from "./LaneLines";
import DetectedAgents from "./DetectedAgents";
import PlannedTrajectory from "./PlannedTrajectory";
import PredictedPaths from "./PredictedPaths";
import TrajectoryCandidates from "./TrajectoryCandidates";
import TrafficLights from "./TrafficLights";
import DrivableSurface from "./DrivableSurface";
import WorldLayerSurface from "./WorldLayerSurface";

const CAMERA_OFFSET_LOCAL = new THREE.Vector3(-18, 20, 0);
const LOOK_AHEAD_LOCAL = new THREE.Vector3(22, 0, 0);
const CAMERA_POSITION_SMOOTHING = 0.12;
const CAMERA_LOOK_SMOOTHING = 0.16;

function CameraController() {
  const { camera } = useThree();
  const targetPosition = useRef(new THREE.Vector3(0, 20, 0));
  const targetLook = useRef(new THREE.Vector3(20, 0, 0));
  const smoothLook = useRef(new THREE.Vector3(20, 0, 0));
  const initialized = useRef(false);

  useEffect(() => {
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = 44;
      camera.near = 0.1;
      camera.far = 3000;
      camera.updateProjectionMatrix();
    }
  }, [camera]);

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

    if (ego) {
      const egoPosition = worldToScene(ego.world_xyz);
      const yaw = yawToScene(ego.yaw_rad);
      const rotatedOffset = CAMERA_OFFSET_LOCAL.clone().applyAxisAngle(
        new THREE.Vector3(0, 1, 0),
        yaw,
      );
      const rotatedLookAhead = LOOK_AHEAD_LOCAL.clone().applyAxisAngle(
        new THREE.Vector3(0, 1, 0),
        yaw,
      );

      targetPosition.current.copy(egoPosition).add(rotatedOffset);
      targetLook.current.copy(egoPosition).add(rotatedLookAhead);
    }

    if (!initialized.current) {
      camera.position.copy(targetPosition.current);
      smoothLook.current.copy(targetLook.current);
      initialized.current = true;
    } else {
      camera.position.lerp(targetPosition.current, CAMERA_POSITION_SMOOTHING);
      smoothLook.current.lerp(targetLook.current, CAMERA_LOOK_SMOOTHING);
    }

    camera.lookAt(smoothLook.current);
  });

  return null;
}

function Ground() {
  return (
    <>
      <mesh rotation-x={-Math.PI / 2} position-y={-0.01} receiveShadow>
        <planeGeometry args={[1200, 1200]} />
        <meshStandardMaterial color="#1a1a24" />
      </mesh>
      <gridHelper
        args={[1200, 240, "#2a2a3a", "#1f1f2e"]}
        position-y={0}
      />
    </>
  );
}

export default function BEVScene() {
  return (
    <>
      <color attach="background" args={["#0a0a0f"]} />
      <ambientLight intensity={0.56} />
      <hemisphereLight args={["#93c5fd", "#05070d", 0.62]} />
      <directionalLight position={[50, 100, 50]} intensity={1.25} />
      <directionalLight position={[-30, 26, -24]} intensity={0.35} color="#6dd3ff" />
      <CameraController />
      <Ground />
      <WorldLayerSurface />
      <DrivableSurface />
      <EgoVehicle />
      <LaneLines />
      <DetectedAgents />
      <TrafficLights />
      <TrajectoryCandidates />
      <PlannedTrajectory />
      <PredictedPaths />
    </>
  );
}
