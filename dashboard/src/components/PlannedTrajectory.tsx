import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { COLORS } from "../utils/colors";

const TRAJ_COLOR = new THREE.Color(COLORS.trajectory);

export default function PlannedTrajectory() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;

    // Clear previous
    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if (child instanceof THREE.Line) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }

    const traj = frame?.["planning/ego_trajectory"];
    if (!traj?.waypoints || traj.waypoints.length < 2) return;

    const points = traj.waypoints.map(
      (wp) => new THREE.Vector3(wp.x, 0.15, -wp.y),
    );
    const geom = new THREE.BufferGeometry().setFromPoints(points);
    const mat = new THREE.LineBasicMaterial({
      color: TRAJ_COLOR,
      transparent: true,
      opacity: 0.95,
    });
    group.add(new THREE.Line(geom, mat));

    // Add dots at waypoint positions for visibility
    const dotGeom = new THREE.SphereGeometry(0.3);
    const dotMat = new THREE.MeshBasicMaterial({ color: TRAJ_COLOR });
    for (let i = 0; i < points.length; i += 3) {
      const dot = new THREE.Mesh(dotGeom, dotMat);
      dot.position.copy(points[i]);
      group.add(dot);
    }
  });

  return <group ref={groupRef} />;
}
