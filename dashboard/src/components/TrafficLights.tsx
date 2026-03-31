import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { COLORS, modalityColor } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

function signalColor(state: string): THREE.Color {
  switch (state) {
    case "RED":
      return new THREE.Color(COLORS.trafficRed);
    case "YELLOW":
      return new THREE.Color(COLORS.trafficAmber);
    case "GREEN":
      return new THREE.Color(COLORS.trafficGreen);
    default:
      return new THREE.Color("#9aa4b2");
  }
}

export default function TrafficLights() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    disposeObject3D(group);

    const trafficLights = frame["perception/traffic_lights"] ?? [];
    for (const signal of trafficLights) {
      const world = signal.world_xyz;
      if (!world || world.length < 2) continue;

      const stateColor = signalColor(signal.state);
      const sourceColor = modalityColor(signal.source_modality);

      const marker = new THREE.Group();
      const scenePosition = worldToScene(world);
      marker.position.set(
        scenePosition.x,
        Math.max((world[2] ?? 0) * 0.15, 0.45),
        scenePosition.z,
      );

      const glow = new THREE.Mesh(
        new THREE.SphereGeometry(0.45, 16, 16),
        new THREE.MeshStandardMaterial({
          color: stateColor,
          emissive: stateColor,
          emissiveIntensity: 0.9,
          transparent: true,
          opacity: Math.max(0.45, Math.min(signal.confidence ?? 1, 1)),
        }),
      );

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.75, 0.08, 10, 32),
        new THREE.MeshStandardMaterial({
          color: sourceColor,
          emissive: sourceColor,
          emissiveIntensity: 0.35,
          transparent: true,
          opacity: 0.95,
        }),
      );
      ring.rotation.x = Math.PI * 0.5;

      marker.add(glow);
      marker.add(ring);
      group.add(marker);
    }
  });

  return <group ref={groupRef} />;
}
