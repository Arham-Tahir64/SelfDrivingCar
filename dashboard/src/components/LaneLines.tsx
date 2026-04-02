import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { COLORS } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";
import { isNavigationFirstMode } from "../utils/bevMode";

function scenePoint(point: number[], y: number): THREE.Vector3 {
  const scene = worldToScene(point);
  return new THREE.Vector3(scene.x, y, scene.z);
}

export default function LaneLines() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    disposeObject3D(group);

    const localMap = frame["map/local_map"];
    if (!isNavigationFirstMode()) {
      const lanes = frame["perception/lanes"] ?? localMap?.perceived_lanes ?? [];
      for (const lane of lanes) {
        if (!lane.polyline_world || lane.polyline_world.length < 2) continue;
        const points = lane.polyline_world.map((p: number[]) => scenePoint(p, 0.045));
        const geom = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({
          color: COLORS.laneLine,
          transparent: true,
          opacity: Math.max(COLORS.laneLineOpacity, lane.confidence ?? 0.45),
        });
        group.add(new THREE.Line(geom, mat));
      }
    }

    if (localMap?.temporary_boundaries) {
      for (const boundary of localMap.temporary_boundaries) {
        if (!boundary?.polyline_world || boundary.polyline_world.length < 2) continue;
        const pts = boundary.polyline_world.map((p: number[]) => scenePoint(p, 0.055));
        const geom = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineDashedMaterial({
          color: COLORS.temporaryBoundary,
          transparent: true,
          opacity: COLORS.temporaryBoundaryOpacity,
          dashSize: 1.5,
          gapSize: 1,
        });
        const line = new THREE.Line(geom, mat);
        line.computeLineDistances();
        group.add(line);
      }
    }
  });

  return <group ref={groupRef} />;
}
