import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { COLORS } from "../utils/colors";
import { disposeObject3D } from "../utils/dispose";

const CONE_COLOR = new THREE.Color(COLORS.cone);
const CONE_RADIUS = 0.3;
const CONE_HEIGHT = 0.7;

export default function Cones() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    disposeObject3D(group);

    const cones = frame?.["perception/cones"];
    const mapCones = frame?.["map/local_map"]?.cone_instances;
    const allCones = [...(cones ?? []), ...(mapCones ?? [])];

    // Deduplicate by proximity
    const seen: THREE.Vector3[] = [];
    for (const cone of allCones) {
      if (!cone.world_xyz) continue;
      const pos = new THREE.Vector3(cone.world_xyz[0], 0, -(cone.world_xyz[1] ?? 0));
      const duplicate = seen.some((s) => s.distanceTo(pos) < 1.0);
      if (duplicate) continue;
      seen.push(pos);

      const geom = new THREE.ConeGeometry(CONE_RADIUS, CONE_HEIGHT, 8);
      const mat = new THREE.MeshStandardMaterial({
        color: CONE_COLOR,
        transparent: true,
        opacity: 0.85,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.copy(pos);
      mesh.position.y = CONE_HEIGHT / 2;
      group.add(mesh);
    }
  });

  return <group ref={groupRef} />;
}
