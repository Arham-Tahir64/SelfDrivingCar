import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { COLORS } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene, yawToScene } from "../utils/scene";
import { isNavigationFirstMode } from "../utils/bevMode";

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

    const priorAnchors = frame["visualization/prior_map"]?.traffic_lights ?? [];
    const liveAnchors = frame["visualization/world_layer"]?.traffic_lights ?? [];
    const liveByActorId = new Map<number, typeof liveAnchors[number]>();
    for (const signal of liveAnchors) {
      liveByActorId.set(signal.actor_id, signal);
    }
    const anchoredTrafficLights = priorAnchors.map((signal) => ({
      ...signal,
      ...(liveByActorId.get(signal.actor_id) ?? {}),
    }));
    if (isNavigationFirstMode() && anchoredTrafficLights.length > 0) {
      for (const signal of anchoredTrafficLights) {
        const world = signal.world_xyz;
        if (!world || world.length < 2) continue;

        const marker = new THREE.Group();
        const scenePosition = worldToScene(world);
        const visibility = signal.visibility_class ?? "adjacent";
        const poleHeight = visibility === "route" ? 2.4 : 2.1;
        marker.position.set(scenePosition.x, 0.03, scenePosition.z);
        marker.rotation.y = yawToScene((signal.yaw_deg * Math.PI) / 180.0);

        const pole = new THREE.Mesh(
          new THREE.CylinderGeometry(0.04, 0.05, poleHeight, 10),
          new THREE.MeshBasicMaterial({
            color: "#5e6268",
            transparent: true,
            opacity: visibility === "route" ? 0.95 : 0.72,
          }),
        );
        pole.position.y = poleHeight * 0.5;

        const housing = new THREE.Mesh(
          new THREE.BoxGeometry(0.28, 0.78, 0.22),
          new THREE.MeshBasicMaterial({
            color: "#22262c",
            transparent: true,
            opacity: visibility === "route" ? 0.96 : 0.82,
          }),
        );
        housing.position.set(0.0, poleHeight + 0.38, 0.0);

        const lamp = new THREE.Mesh(
          new THREE.SphereGeometry(0.085, 12, 12),
          new THREE.MeshStandardMaterial({
            color: signalColor(signal.state),
            emissive: signalColor(signal.state),
            emissiveIntensity: visibility === "route" ? 1.1 : 0.7,
            transparent: true,
            opacity: Math.max(0.5, Math.min(signal.confidence ?? 1, 1)),
          }),
        );
        lamp.position.set(0.0, poleHeight + 0.52, 0.11);

        marker.add(pole);
        marker.add(housing);
        marker.add(lamp);
        group.add(marker);
      }
      return;
    }

    const trafficLights = frame["perception/traffic_lights"] ?? [];
    for (const signal of trafficLights) {
      const world = signal.world_xyz;
      if (!world || world.length < 2) continue;

      const marker = new THREE.Group();
      const scenePosition = worldToScene(world);
      marker.position.set(
        scenePosition.x,
        Math.max((world[2] ?? 0) * 0.15, 0.45),
        scenePosition.z,
      );

      const glow = new THREE.Mesh(
        new THREE.SphereGeometry(0.4, 14, 14),
        new THREE.MeshStandardMaterial({
          color: signalColor(signal.state),
          emissive: signalColor(signal.state),
          emissiveIntensity: 0.85,
          transparent: true,
          opacity: Math.max(0.45, Math.min(signal.confidence ?? 1, 1)),
        }),
      );

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.68, 0.07, 10, 24),
        new THREE.MeshStandardMaterial({
          color: COLORS.trafficAmber,
          emissive: COLORS.trafficAmber,
          emissiveIntensity: 0.2,
          transparent: true,
          opacity: 0.85,
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
