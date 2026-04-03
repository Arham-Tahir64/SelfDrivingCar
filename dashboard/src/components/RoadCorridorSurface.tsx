import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

const ROAD_FILL = new THREE.Color("#27443c");
const ROAD_BORDER = new THREE.Color("#4f7d70");

export default function RoadCorridorSurface() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);
  const lastSignatureRef = useRef<string>("");

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    const corridor = frame["visualization/road_corridor"];
    const strips = corridor?.strips ?? [];
    const signature = strips.map((strip) => strip.lane_id).join("|");
    if (signature === lastSignatureRef.current) {
      return;
    }
    lastSignatureRef.current = signature;

    disposeObject3D(group);
    for (const strip of strips) {
      const polygon = strip.polygon_world ?? [];
      if (polygon.length < 3) continue;

      const points = polygon.map((point) => {
        const scene = worldToScene(point);
        return new THREE.Vector2(scene.x, scene.z);
      });
      const shape = new THREE.Shape(points);
      const geometry = new THREE.ShapeGeometry(shape);
      const material = new THREE.MeshBasicMaterial({
        color: ROAD_FILL,
        transparent: true,
        opacity: strip.is_junction ? 0.36 : 0.28,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = 0.01;
      group.add(mesh);

      const borderPoints = [...points, points[0]].map(
        (point) => new THREE.Vector3(point.x, 0.035, point.y),
      );
      const borderGeometry = new THREE.BufferGeometry().setFromPoints(borderPoints);
      const borderMaterial = new THREE.LineBasicMaterial({
        color: ROAD_BORDER,
        transparent: true,
        opacity: 0.45,
      });
      group.add(new THREE.Line(borderGeometry, borderMaterial));
    }
  });

  return <group ref={groupRef} />;
}
