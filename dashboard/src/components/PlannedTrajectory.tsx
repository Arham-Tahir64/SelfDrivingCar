import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { COLORS } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

const TRAJ_COLOR = new THREE.Color(COLORS.trajectory);

// Shared materials — created once, reused across frames
const lineMaterial = new THREE.LineBasicMaterial({
  color: TRAJ_COLOR,
  transparent: true,
  opacity: 0.95,
});
const dotGeometry = new THREE.SphereGeometry(0.3);
const dotMaterial = new THREE.MeshBasicMaterial({ color: TRAJ_COLOR });

export default function PlannedTrajectory() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    // Dispose old geometry but NOT shared materials
    for (const child of [...group.children]) {
      if (child instanceof THREE.Line || child instanceof THREE.Mesh) {
        if (child.geometry) child.geometry.dispose();
      }
    }
    group.clear();

    const traj = frame?.["planning/ego_trajectory"];
    if (!traj?.waypoints || traj.waypoints.length < 2) return;

    const points = traj.waypoints.map(
      (wp) => {
        const scene = worldToScene([wp.x, wp.y, 0]);
        return new THREE.Vector3(scene.x, 0.15, scene.z);
      },
    );
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    group.add(new THREE.Line(geom, lineMaterial));

    for (let i = 0; i < points.length; i += 3) {
      const dot = new THREE.Mesh(dotGeometry, dotMaterial);
      dot.position.copy(points[i]);
      group.add(dot);
    }
  });

  return <group ref={groupRef} />;
}
