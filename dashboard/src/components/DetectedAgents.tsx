import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { classColor, classOpacity, modalityColor } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

const AGENT_HEIGHT = 1.6;
const DEFAULT_SIZE = new THREE.Vector3(4.5, AGENT_HEIGHT, 2.0);

function bboxCentroid(bbox: number[][]): THREE.Vector3 {
  if (!bbox || bbox.length === 0) return new THREE.Vector3();
  let sx = 0;
  let sy = 0;
  let sz = 0;
  for (const corner of bbox) {
    sx += corner[0] ?? 0;
    sy += corner[1] ?? 0;
    sz += corner[2] ?? 0;
  }
  const n = bbox.length;
  return new THREE.Vector3(sx / n, sy / n, sz / n);
}

function bboxSize(bbox: number[][]): THREE.Vector3 {
  if (!bbox || bbox.length < 2) return DEFAULT_SIZE.clone();
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const c of bbox) {
    minX = Math.min(minX, c[0] ?? 0);
    maxX = Math.max(maxX, c[0] ?? 0);
    minY = Math.min(minY, c[1] ?? 0);
    maxY = Math.max(maxY, c[1] ?? 0);
    minZ = Math.min(minZ, c[2] ?? 0);
    maxZ = Math.max(maxZ, c[2] ?? 0);
  }
  const dx = Math.max(maxX - minX, 0.5);
  const dy = Math.max(maxY - minY, 0.5);
  const dz = Math.max(maxZ - minZ, 0.5);
  return new THREE.Vector3(dx, dy, dz);
}

export default function DetectedAgents() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    disposeObject3D(group);

    const detections = frame["perception/detections"];
    if (!detections) return;

    for (const det of detections) {
      if (det.track_state === "DELETED" || det.track_state === "LOST") continue;

      const centroid = bboxCentroid(det.world_bbox_3d);
      const size = bboxSize(det.world_bbox_3d);
      const color = classColor(det.object_class);
      const opacity = classOpacity(det.object_class);
      const outlineColor = modalityColor(det.source_modality);
      const sceneCenter = worldToScene([centroid.x, centroid.y, centroid.z]);

      const geom = new THREE.BoxGeometry(size.x, size.y, size.z);
      const mat = new THREE.MeshStandardMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
        emissive: outlineColor,
        emissiveIntensity: det.source_modality === "fused" ? 0.18 : 0.08,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(sceneCenter.x, size.y / 2, sceneCenter.z);

      const edgesGeom = new THREE.EdgesGeometry(geom);
      const edgesMat = new THREE.LineBasicMaterial({
        color: outlineColor,
        transparent: true,
        opacity: 0.95,
      });
      const edges = new THREE.LineSegments(edgesGeom, edgesMat);
      mesh.add(edges);

      group.add(mesh);
    }
  });

  return <group ref={groupRef} />;
}
