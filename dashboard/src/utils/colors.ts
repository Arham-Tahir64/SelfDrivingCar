import * as THREE from "three";

export const COLORS = {
  background: "#0a0a0f",
  ground: "#1a1a24",
  gridLine: "#2a2a3a",

  egoCyan: "#00E5FF",
  egoBody: "#ffffff",
  trajectory: "#00BCD4",

  vehicle: new THREE.Color(0.71, 0.71, 0.78),
  vehicleOpacity: 0.4,
  pedestrian: new THREE.Color(0.13, 0.77, 0.37),
  pedestrianOpacity: 0.3,
  cyclist: new THREE.Color(0.98, 0.75, 0.14),
  cyclistOpacity: 0.3,
  emergencyVehicle: new THREE.Color(1.0, 0.2, 0.2),
  emergencyOpacity: 0.5,

  cone: "#FF6D00",
  laneLine: "#ffffff",
  laneLineOpacity: 0.5,
  closedLane: "#ff4444",
  temporaryBoundary: "#ffffff",
  temporaryBoundaryOpacity: 0.8,

  predictionDefault: "#888899",
  predictionOpacity: 0.45,

  trafficRed: "#ff0000",
  trafficAmber: "#ffc800",
  trafficGreen: "#00ff00",
} as const;

export function classColor(objectClass: string): THREE.Color {
  switch (objectClass) {
    case "vehicle":
      return COLORS.vehicle;
    case "pedestrian":
      return COLORS.pedestrian;
    case "cyclist":
      return COLORS.cyclist;
    case "emergency_vehicle":
      return COLORS.emergencyVehicle;
    default:
      return new THREE.Color(0.5, 0.5, 0.5);
  }
}

export function classOpacity(objectClass: string): number {
  switch (objectClass) {
    case "vehicle":
      return COLORS.vehicleOpacity;
    case "pedestrian":
      return COLORS.pedestrianOpacity;
    case "cyclist":
      return COLORS.cyclistOpacity;
    case "emergency_vehicle":
      return COLORS.emergencyOpacity;
    default:
      return 0.3;
  }
}
