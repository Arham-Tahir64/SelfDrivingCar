import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";
import type { PlannerCandidate } from "../utils/types";

const CHOSEN_COLOR = new THREE.Color("#00E5FF");
const REJECTED_BASE = new THREE.Color("#555566");
const REJECTED_BAD = new THREE.Color("#FF4444");

function candidateColor(candidate: PlannerCandidate, minScore: number, maxScore: number): THREE.Color {
  const range = Math.max(maxScore - minScore, 0.01);
  const t = Math.min(1, (candidate.score - minScore) / range);
  // Chosen (best) = cyan, rejected = grey blending to red as score increases
  return new THREE.Color().copy(REJECTED_BASE).lerp(REJECTED_BAD, t);
}

export default function TrajectoryCandidates() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    disposeObject3D(group);

    const candidates = frame?.["planning/candidates"];
    if (!candidates || candidates.length === 0) return;

    const bestTrajectory = frame?.["planning/ego_trajectory"];

    const scores = candidates.map((c) => c.score);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);

    // Sort so the best (chosen) candidate renders last (on top)
    const sorted = [...candidates].sort((a, b) => b.score - a.score);

    for (const candidate of sorted) {
      const wp = candidate.trajectory.waypoints;
      if (!wp || wp.length < 2) continue;

      const isBest =
        bestTrajectory &&
        wp.length > 0 &&
        bestTrajectory.waypoints.length > 0 &&
        Math.abs(candidate.score - minScore) < 0.01;

      // Skip the best candidate — PlannedTrajectory.tsx already renders it
      if (isBest) continue;

      const points = wp.map((p) => {
        const scene = worldToScene([p.x, p.y, 0]);
        return new THREE.Vector3(scene.x, 0.15, scene.z);
      });

      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const color = candidateColor(candidate, minScore, maxScore);
      const mat = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: 0.25,
        depthWrite: false,
      });
      const line = new THREE.Line(geom, mat);
      group.add(line);
    }
  });

  return <group ref={groupRef} />;
}
