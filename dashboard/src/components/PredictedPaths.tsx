import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { classColor, COLORS } from "../utils/colors";

export default function PredictedPaths() {
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;

    while (group.children.length > 0) {
      const child = group.children[0];
      group.remove(child);
      if (child instanceof THREE.Line) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }

    const predictions = frame?.["prediction/agents"];
    if (!predictions) return;

    for (const pred of predictions) {
      if (!pred.predicted_trajectory || pred.predicted_trajectory.length < 2) continue;

      const points = pred.predicted_trajectory.map(
        (p: number[]) => new THREE.Vector3(p[0], 0.1, -(p[1] ?? 0)),
      );
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const color = classColor(pred.object_class);
      const mat = new THREE.LineDashedMaterial({
        color,
        transparent: true,
        opacity: COLORS.predictionOpacity,
        dashSize: 1,
        gapSize: 0.8,
      });
      const line = new THREE.Line(geom, mat);
      line.computeLineDistances();
      group.add(line);
    }
  });

  return <group ref={groupRef} />;
}
