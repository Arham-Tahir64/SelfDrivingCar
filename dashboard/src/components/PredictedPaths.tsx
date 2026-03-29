import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { classColor, COLORS } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

export default function PredictedPaths() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    disposeObject3D(group);

    const predictions = frame?.["prediction/agents"];
    if (!predictions) return;

    for (const pred of predictions) {
      if (!pred.predicted_trajectory || pred.predicted_trajectory.length < 2) continue;

      const points = pred.predicted_trajectory.map(
        (p) => {
          const scene = worldToScene([p.x, p.y, 0]);
          return new THREE.Vector3(scene.x, 0.1, scene.z);
        },
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
