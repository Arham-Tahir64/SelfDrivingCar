import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

const ROAD_FILL = new THREE.Color("#1d2a31");
const JUNCTION_FILL = new THREE.Color("#24313a");
const ROAD_ROUTE = new THREE.Color("#294a44");
const SIDEWALK_FILL = new THREE.Color("#353b46");
const CURB_COLOR = new THREE.Color("#687180");
const LANE_MARKER = new THREE.Color("#8f9486");
const ROUTE_LINE = new THREE.Color("#d8f07a");

function polygonShape(pointsWorld: number[][]): THREE.Shape | null {
  if (!pointsWorld || pointsWorld.length < 3) return null;
  const shapePoints = pointsWorld.map((point) => {
    const scene = worldToScene(point);
    return new THREE.Vector2(scene.x, scene.z);
  });
  return new THREE.Shape(shapePoints);
}

export default function PriorMapSurface() {
  const groupRef = useRef<THREE.Group>(null);
  const lastSignatureRef = useRef<string>("");

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;

    const priorMap = frame?.["visualization/prior_map"];
    const signature = priorMap?.signature ?? "";
    if (!priorMap || signature === lastSignatureRef.current) return;
    lastSignatureRef.current = signature;

    disposeObject3D(group);

    for (const road of priorMap.roads ?? []) {
      const shape = polygonShape(road.polygon_world);
      if (!shape) continue;
      const geometry = new THREE.ShapeGeometry(shape);
      const material = new THREE.MeshBasicMaterial({
        color: road.is_route ? ROAD_ROUTE : road.is_junction ? JUNCTION_FILL : ROAD_FILL,
        transparent: true,
        opacity: road.is_route ? 0.84 : road.is_junction ? 0.62 : 0.56,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = 0.0;
      group.add(mesh);
    }

    for (const sidewalk of priorMap.sidewalks ?? []) {
      const shape = polygonShape(sidewalk.polygon_world);
      if (!shape) continue;
      const geometry = new THREE.ShapeGeometry(shape);
      const material = new THREE.MeshBasicMaterial({
        color: SIDEWALK_FILL,
        transparent: true,
        opacity: 0.46,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = 0.01;
      group.add(mesh);

      if (sidewalk.edge_world && sidewalk.edge_world.length >= 2) {
        const edgePoints = sidewalk.edge_world.map((point) => {
          const scene = worldToScene(point);
          return new THREE.Vector3(scene.x, 0.03, scene.z);
        });
        const edgeGeometry = new THREE.BufferGeometry().setFromPoints(edgePoints);
        const edgeMaterial = new THREE.LineBasicMaterial({
          color: CURB_COLOR,
          transparent: true,
          opacity: 0.45,
        });
        group.add(new THREE.Line(edgeGeometry, edgeMaterial));
      }
    }

    for (const marker of priorMap.lane_markers ?? []) {
      if (!marker.polyline_world || marker.polyline_world.length < 2) continue;
      const markerPoints = marker.polyline_world.map((point) => {
        const scene = worldToScene(point);
        return new THREE.Vector3(scene.x, 0.035, scene.z);
      });
      const markerGeometry = new THREE.BufferGeometry().setFromPoints(markerPoints);
      const markerMaterial = new THREE.LineBasicMaterial({
        color: LANE_MARKER,
        transparent: true,
        opacity: 0.42,
      });
      group.add(new THREE.Line(markerGeometry, markerMaterial));
    }

    if (priorMap.route_polyline_world && priorMap.route_polyline_world.length >= 2) {
      const routePoints = priorMap.route_polyline_world.map((point) => {
        const scene = worldToScene(point);
        return new THREE.Vector3(scene.x, 0.05, scene.z);
      });
      const routeGeometry = new THREE.BufferGeometry().setFromPoints(routePoints);
      const routeMaterial = new THREE.LineBasicMaterial({
        color: ROUTE_LINE,
        transparent: true,
        opacity: 0.9,
      });
      group.add(new THREE.Line(routeGeometry, routeMaterial));
    }
  });

  return <group ref={groupRef} />;
}
