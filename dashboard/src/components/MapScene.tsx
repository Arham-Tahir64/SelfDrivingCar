import { useEffect, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { MapControls, OrthographicCamera } from "@react-three/drei";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import PriorMapSurface from "./PriorMapSurface";
import EgoVehicle from "./EgoVehicle";
import LaneLines from "./LaneLines";
import DetectedAgents from "./DetectedAgents";
import PlannedTrajectory from "./PlannedTrajectory";
import PredictedPaths from "./PredictedPaths";
import TrajectoryCandidates from "./TrajectoryCandidates";
import TrafficLights from "./TrafficLights";
import DrivableSurface from "./DrivableSurface";
import RoadCorridorSurface from "./RoadCorridorSurface";

const FIT_PADDING = 1.1;
const DEFAULT_VIEW_SIZE_M = 220;

function MapCameraController() {
  const cameraRef = useRef<THREE.OrthographicCamera>(null);
  const controlsRef = useRef<any>(null);
  const lastFitKeyRef = useRef<string>("");
  const { size } = useThree();

  useEffect(() => {
    const camera = cameraRef.current;
    if (!camera) return;
    camera.up.set(0, 0, -1);
    camera.lookAt(0, 0, 0);
  }, []);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const priorMap = frame?.["visualization/prior_map"];
    const camera = cameraRef.current;
    if (!camera || !priorMap) return;
    const fitKey = `${priorMap.signature}:${size.width}x${size.height}`;
    if (fitKey === lastFitKeyRef.current) return;
    lastFitKeyRef.current = fitKey;

    const bounds = priorMap.bounds_world;
    const width = Math.max(bounds.max_x - bounds.min_x, DEFAULT_VIEW_SIZE_M);
    const height = Math.max(bounds.max_y - bounds.min_y, DEFAULT_VIEW_SIZE_M);
    const centerX = (bounds.min_x + bounds.max_x) * 0.5;
    const centerY = (bounds.min_y + bounds.max_y) * 0.5;
    const zoom = Math.min(
      size.width / (width * FIT_PADDING),
      size.height / (height * FIT_PADDING),
    );

    camera.position.set(centerX, 220, centerY);
    camera.zoom = Math.max(0.05, zoom);
    camera.lookAt(centerX, 0, centerY);
    camera.updateProjectionMatrix();

    if (controlsRef.current) {
      controlsRef.current.target.set(centerX, 0, centerY);
      controlsRef.current.update();
    }
  });

  return (
    <>
      <OrthographicCamera ref={cameraRef} makeDefault near={0.1} far={5000} position={[0, 220, 0]} zoom={1} />
      <MapControls
        ref={controlsRef}
        enableRotate={false}
        screenSpacePanning
        zoomSpeed={0.9}
        panSpeed={1.0}
        minZoom={0.04}
        maxZoom={8}
      />
    </>
  );
}

export default function MapScene() {
  return (
    <>
      <color attach="background" args={["#07090c"]} />
      <ambientLight intensity={0.72} />
      <hemisphereLight args={["#9ed6ff", "#05070d", 0.48]} />
      <directionalLight position={[40, 120, 35]} intensity={0.58} color="#eef6ff" />
      <MapCameraController />
      <PriorMapSurface />
      <RoadCorridorSurface />
      <DrivableSurface />
      <LaneLines />
      <TrafficLights />
      <TrajectoryCandidates />
      <PlannedTrajectory />
      <PredictedPaths />
      <DetectedAgents />
      <EgoVehicle />
    </>
  );
}
