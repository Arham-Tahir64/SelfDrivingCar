import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { COLORS } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

const MAX_LANES = 30;

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

    const lanes = frame["perception/lanes"];
    if (lanes) {
      for (let i = 0; i < Math.min(lanes.length, MAX_LANES); i++) {
        const lane = lanes[i];
        if (!lane.polyline_world || lane.polyline_world.length < 2) continue;
        const points = lane.polyline_world.map((p: number[]) => scenePoint(p, 0.05));
        const geom = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({
          color: COLORS.laneLine,
          transparent: true,
          opacity: COLORS.laneLineOpacity,
        });
        group.add(new THREE.Line(geom, mat));
      }
    }

    const localMap = frame["map/local_map"];
    if (localMap?.static_lanes) {
      for (let i = 0; i < Math.min(localMap.static_lanes.length, MAX_LANES); i++) {
        const sl = localMap.static_lanes[i];
        const isClosed = localMap.closed_lanes?.includes(sl.lane_id);

        if (sl.centerline_world && sl.centerline_world.length >= 2) {
          const pts = sl.centerline_world.map((p: number[]) => scenePoint(p, 0.04));
          const geom = new THREE.BufferGeometry().setFromPoints(pts);
          const mat = new THREE.LineDashedMaterial({
            color: isClosed ? COLORS.closedLane : COLORS.laneLine,
            transparent: true,
            opacity: isClosed ? 0.7 : 0.3,
            dashSize: 2,
            gapSize: 2,
          });
          const line = new THREE.Line(geom, mat);
          line.computeLineDistances();
          group.add(line);
        }

        if (sl.left_boundary_world && sl.left_boundary_world.length >= 2) {
          const pts = sl.left_boundary_world.map((p: number[]) => scenePoint(p, 0.04));
          const geom = new THREE.BufferGeometry().setFromPoints(pts);
          const mat = new THREE.LineBasicMaterial({
            color: isClosed ? COLORS.closedLane : COLORS.laneLine,
            transparent: true,
            opacity: isClosed ? 0.6 : COLORS.laneLineOpacity,
          });
          group.add(new THREE.Line(geom, mat));
        }

        if (sl.right_boundary_world && sl.right_boundary_world.length >= 2) {
          const pts = sl.right_boundary_world.map((p: number[]) => scenePoint(p, 0.04));
          const geom = new THREE.BufferGeometry().setFromPoints(pts);
          const mat = new THREE.LineBasicMaterial({
            color: isClosed ? COLORS.closedLane : COLORS.laneLine,
            transparent: true,
            opacity: isClosed ? 0.6 : COLORS.laneLineOpacity,
          });
          group.add(new THREE.Line(geom, mat));
        }
      }
    }

    if (localMap?.temporary_boundaries) {
      for (const boundary of localMap.temporary_boundaries) {
        if (!boundary?.polyline_world || boundary.polyline_world.length < 2) continue;
        const pts = boundary.polyline_world.map((p: number[]) => scenePoint(p, 0.06));
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
