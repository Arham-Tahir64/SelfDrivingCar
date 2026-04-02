import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { disposeObject3D } from "../utils/dispose";
import { worldToScene } from "../utils/scene";

const ROUTE_ROAD = new THREE.Color("#5da396");
const ADJACENT_ROAD = new THREE.Color("#314640");
const BACKGROUND_ROAD = new THREE.Color("#18211e");
const JUNCTION_ROUTE = new THREE.Color("#678c81");
const JUNCTION_ADJACENT = new THREE.Color("#2b3733");
const JUNCTION_BACKGROUND = new THREE.Color("#141917");
const CLOSED_ROAD = new THREE.Color("#4b2b30");
const SIDEWALK_COLOR = new THREE.Color("#4a4f59");
const SIDEWALK_ROUTE_COLOR = new THREE.Color("#5a616d");
const CURB_COLOR = new THREE.Color("#8c94a1");
const LANE_MARKER = new THREE.Color("#9ea38a");
const ROUTE_MARKER = new THREE.Color("#f0f6b4");
const ROAD_SURFACE_Y = 0.012;
const MARKER_Y = 0.03;
const CURB_Y = 0.08;
const SIDEWALK_Y = 0.018;
const SIDEWALK_HEIGHT = 0.08;

function polygonShape(pointsWorld: number[][]): THREE.Shape | null {
  if (!pointsWorld || pointsWorld.length < 3) return null;
  const shapePoints = pointsWorld.map((point) => {
    const scene = worldToScene(point);
    return new THREE.Vector2(scene.x, scene.z);
  });
  return new THREE.Shape(shapePoints);
}

function roadMaterial(road: {
  visibility_class?: string;
  is_route?: boolean;
  is_junction?: boolean;
  is_junction_patch?: boolean;
  is_closed?: boolean;
}): THREE.MeshBasicMaterial {
  let color = ADJACENT_ROAD;
  let opacity = 0.18;
  const visibility = road.visibility_class ?? (road.is_route ? "route" : "adjacent");
  if (visibility === "background") {
    color = BACKGROUND_ROAD;
    opacity = 0.08;
  } else if (visibility === "route") {
    color = ROUTE_ROAD;
    opacity = 0.42;
  }
  if (road.is_junction) {
    color =
      visibility === "route"
        ? JUNCTION_ROUTE
        : visibility === "background"
          ? JUNCTION_BACKGROUND
          : JUNCTION_ADJACENT;
    opacity = visibility === "route" ? 0.34 : visibility === "background" ? 0.07 : 0.14;
  }
  if (road.is_closed) {
    color = CLOSED_ROAD;
    opacity = Math.max(opacity, 0.26);
  }
  return new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

export default function WorldLayerSurface() {
  const groupRef = useRef<THREE.Group>(null);
  const lastTickRef = useRef<number | null>(null);
  const lastSignatureRef = useRef<string>("");

  useFrame(() => {
    const frame = useFrameStore.getState().currentFrame;
    const group = groupRef.current;
    if (!group) return;
    if (!frame || lastTickRef.current === frame.tick_id) return;
    lastTickRef.current = frame.tick_id;

    const worldLayer = frame["visualization/world_layer"];
    const signature = worldLayer?.signature ?? "";
    if (!worldLayer || signature === lastSignatureRef.current) {
      return;
    }
    lastSignatureRef.current = signature;

    disposeObject3D(group);

    for (const road of worldLayer.roads ?? []) {
      const shape = polygonShape(road.polygon_world);
      if (!shape) continue;
      const geometry = new THREE.ShapeGeometry(shape);
      const mesh = new THREE.Mesh(geometry, roadMaterial(road));
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = ROAD_SURFACE_Y;
      group.add(mesh);
    }

    for (const sidewalk of worldLayer.sidewalks ?? []) {
      if (sidewalk.visibility_class === "background") continue;
      const shape = polygonShape(sidewalk.polygon_world);
      if (!shape) continue;
      const geometry = new THREE.ExtrudeGeometry(shape, {
        depth: SIDEWALK_HEIGHT,
        bevelEnabled: false,
      });
      const material = new THREE.MeshBasicMaterial({
        color: sidewalk.is_route_adjacent ? SIDEWALK_ROUTE_COLOR : SIDEWALK_COLOR,
        transparent: true,
        opacity: sidewalk.is_route_adjacent ? 0.3 : 0.18,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.y = SIDEWALK_Y;
      group.add(mesh);

      if (sidewalk.edge_world && sidewalk.edge_world.length >= 2) {
        const edgePoints = sidewalk.edge_world.map((point) => {
          const scene = worldToScene(point);
          return new THREE.Vector3(scene.x, CURB_Y, scene.z);
        });
        const edgeGeometry = new THREE.BufferGeometry().setFromPoints(edgePoints);
        const edgeMaterial = new THREE.LineBasicMaterial({
          color: CURB_COLOR,
          transparent: true,
          opacity: sidewalk.is_route_adjacent ? 0.45 : 0.28,
        });
        group.add(new THREE.Line(edgeGeometry, edgeMaterial));
      }
    }

    for (const marker of worldLayer.lane_markers ?? []) {
      if (!marker.polyline_world || marker.polyline_world.length < 2) continue;
      if (marker.visibility_class === "background") continue;
      const markerPoints = marker.polyline_world.map((point) => {
        const scene = worldToScene(point);
        return new THREE.Vector3(scene.x, MARKER_Y, scene.z);
      });
      const markerGeometry = new THREE.BufferGeometry().setFromPoints(markerPoints);
      const markerMaterial = new THREE.LineBasicMaterial({
        color: marker.is_route ? ROUTE_MARKER : LANE_MARKER,
        transparent: true,
        opacity: marker.is_route ? 0.9 : 0.35,
      });
      group.add(new THREE.Line(markerGeometry, markerMaterial));
    }
  });

  return <group ref={groupRef} />;
}
