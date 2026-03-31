import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { classColor, COLORS } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

// Only render every Nth step to keep geometry count manageable
const ELLIPSE_STEP_STRIDE = 5; // render ellipse at steps 0, 5, 10, ...
const ELLIPSE_SEGMENTS = 24;
const SCALE_FACTOR = 1.0; // visual scale for ellipses (1.0 = 1 sigma)

function createEllipseGeometry(
  sigmaX: number,
  sigmaY: number,
  segments: number,
): THREE.BufferGeometry {
  const points: THREE.Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const theta = (i / segments) * Math.PI * 2;
    points.push(
      new THREE.Vector3(
        Math.cos(theta) * sigmaX * SCALE_FACTOR,
        0,
        Math.sin(theta) * sigmaY * SCALE_FACTOR,
      ),
    );
  }
  return new THREE.BufferGeometry().setFromPoints(points);
}

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
      if (!pred.predicted_trajectory || pred.predicted_trajectory.length < 2)
        continue;

      const color = classColor(pred.object_class);
      const trajectory = pred.predicted_trajectory;
      const covariances = pred.covariance_by_step;
      const numSteps = trajectory.length;

      // --- Trajectory line (sampled every few steps for long trajectories) ---
      const lineStride = numSteps > 20 ? 3 : 1;
      const linePoints: THREE.Vector3[] = [];
      for (let i = 0; i < numSteps; i += lineStride) {
        const p = trajectory[i];
        const scene = worldToScene([p.x, p.y, 0]);
        linePoints.push(new THREE.Vector3(scene.x, 0.12, scene.z));
      }
      // Always include last point
      if (numSteps > 1) {
        const last = trajectory[numSteps - 1];
        const scene = worldToScene([last.x, last.y, 0]);
        linePoints.push(new THREE.Vector3(scene.x, 0.12, scene.z));
      }

      const lineGeom = new THREE.BufferGeometry().setFromPoints(linePoints);
      const lineMat = new THREE.LineDashedMaterial({
        color,
        transparent: true,
        opacity: COLORS.predictionOpacity,
        dashSize: 1,
        gapSize: 0.8,
      });
      const line = new THREE.Line(lineGeom, lineMat);
      line.computeLineDistances();
      group.add(line);

      // --- Uncertainty ellipses ---
      if (!covariances) continue;

      for (let i = 0; i < numSteps; i += ELLIPSE_STEP_STRIDE) {
        if (i >= covariances.length) break;
        const cov = covariances[i];
        if (!cov || cov.length < 2) continue;

        // cov is [[sigma_lon^2, ...], [..., sigma_lat^2]]
        const sigmaLon = Math.sqrt(Math.abs(cov[0]?.[0] ?? 0));
        const sigmaLat = Math.sqrt(Math.abs(cov[1]?.[1] ?? 0));

        if (sigmaLon < 0.05 && sigmaLat < 0.05) continue;

        const p = trajectory[i];
        const scene = worldToScene([p.x, p.y, 0]);

        // Ellipse ring
        const ellipseGeom = createEllipseGeometry(
          sigmaLon,
          sigmaLat,
          ELLIPSE_SEGMENTS,
        );
        const timeFraction = i / Math.max(numSteps - 1, 1);
        const ellipseMat = new THREE.LineBasicMaterial({
          color,
          transparent: true,
          opacity: Math.max(0.08, 0.35 * (1.0 - timeFraction * 0.7)),
        });
        const ellipseLine = new THREE.Line(ellipseGeom, ellipseMat);
        ellipseLine.position.set(scene.x, 0.08, scene.z);
        // Rotate ellipse to align with agent heading
        ellipseLine.rotation.y = -p.yaw;
        group.add(ellipseLine);

        // Semi-transparent filled disc for the first few ellipses
        if (i <= ELLIPSE_STEP_STRIDE * 4) {
          const discGeom = new THREE.CircleGeometry(
            Math.max(sigmaLon, sigmaLat) * SCALE_FACTOR,
            ELLIPSE_SEGMENTS,
          );
          const discMat = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: Math.max(0.02, 0.08 * (1.0 - timeFraction)),
            side: THREE.DoubleSide,
            depthWrite: false,
          });
          const disc = new THREE.Mesh(discGeom, discMat);
          disc.position.set(scene.x, 0.06, scene.z);
          disc.rotation.x = -Math.PI / 2;
          group.add(disc);
        }
      }
    }
  });

  return <group ref={groupRef} />;
}
