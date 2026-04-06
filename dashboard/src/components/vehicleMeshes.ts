import * as THREE from "three";

export type VehicleVariant = "hero" | "sedan" | "suv" | "van";

type VehicleProfile = {
  bodyPoints: Array<[number, number]>;
  cabinSize: [number, number, number];
  cabinPosition: [number, number, number];
  roofSize: [number, number, number];
  roofPosition: [number, number, number];
  noseSize: [number, number, number];
  nosePosition: [number, number, number];
  tailSize: [number, number, number];
  tailPosition: [number, number, number];
  wheelRadius: number;
  wheelWidth: number;
  wheelPositions: Array<[number, number, number]>;
  headLightPositions: Array<[number, number, number]>;
  tailLightPositions: Array<[number, number, number]>;
  windshieldSize: [number, number, number];
  windshieldPosition: [number, number, number];
  shadowScale: [number, number, number];
  roofLightBar?: [number, number, number];
};

type VehicleBuildOptions = {
  variant: VehicleVariant;
  bodyColor: THREE.ColorRepresentation;
  accentColor: THREE.ColorRepresentation;
  bodyOpacity?: number;
  bodyRoughness?: number;
  bodyMetalness?: number;
  glassColor?: THREE.ColorRepresentation;
  glassOpacity?: number;
  trimColor?: THREE.ColorRepresentation;
  shadowOpacity?: number;
  outlineOpacity?: number;
  emissiveIntensity?: number;
  headLightColor?: THREE.ColorRepresentation;
  tailLightColor?: THREE.ColorRepresentation;
};

const VEHICLE_PROFILES: Record<VehicleVariant, VehicleProfile> = {
  hero: {
    bodyPoints: [
      [-0.58, -0.18],
      [-0.54, -0.25],
      [-0.28, -0.31],
      [0.22, -0.31],
      [0.46, -0.23],
      [0.58, -0.12],
      [0.58, 0.12],
      [0.46, 0.23],
      [0.22, 0.31],
      [-0.28, 0.31],
      [-0.54, 0.25],
      [-0.58, 0.18],
    ],
    cabinSize: [0.58, 0.24, 0.42],
    cabinPosition: [0.0, 0.24, 0.0],
    roofSize: [0.34, 0.12, 0.34],
    roofPosition: [0.02, 0.38, 0.0],
    noseSize: [0.16, 0.06, 0.44],
    nosePosition: [0.5, 0.08, 0.0],
    tailSize: [0.14, 0.06, 0.46],
    tailPosition: [-0.52, 0.08, 0.0],
    wheelRadius: 0.08,
    wheelWidth: 0.06,
    wheelPositions: [
      [0.32, 0.09, 0.24],
      [0.32, 0.09, -0.24],
      [-0.3, 0.09, 0.24],
      [-0.3, 0.09, -0.24],
    ],
    headLightPositions: [
      [0.58, 0.11, 0.14],
      [0.58, 0.11, -0.14],
    ],
    tailLightPositions: [
      [-0.58, 0.11, 0.16],
      [-0.58, 0.11, -0.16],
    ],
    windshieldSize: [0.18, 0.09, 0.34],
    windshieldPosition: [0.2, 0.34, 0.0],
    shadowScale: [1.2, 1, 0.78],
    roofLightBar: [0.0, 0.47, 0.0],
  },
  sedan: {
    bodyPoints: [
      [-0.56, -0.17],
      [-0.52, -0.23],
      [-0.24, -0.28],
      [0.2, -0.28],
      [0.44, -0.2],
      [0.56, -0.11],
      [0.56, 0.11],
      [0.44, 0.2],
      [0.2, 0.28],
      [-0.24, 0.28],
      [-0.52, 0.23],
      [-0.56, 0.17],
    ],
    cabinSize: [0.5, 0.22, 0.38],
    cabinPosition: [0.0, 0.22, 0.0],
    roofSize: [0.28, 0.1, 0.3],
    roofPosition: [0.0, 0.34, 0.0],
    noseSize: [0.14, 0.05, 0.4],
    nosePosition: [0.48, 0.075, 0.0],
    tailSize: [0.12, 0.05, 0.42],
    tailPosition: [-0.5, 0.075, 0.0],
    wheelRadius: 0.076,
    wheelWidth: 0.055,
    wheelPositions: [
      [0.31, 0.085, 0.22],
      [0.31, 0.085, -0.22],
      [-0.29, 0.085, 0.22],
      [-0.29, 0.085, -0.22],
    ],
    headLightPositions: [
      [0.56, 0.1, 0.13],
      [0.56, 0.1, -0.13],
    ],
    tailLightPositions: [
      [-0.56, 0.1, 0.14],
      [-0.56, 0.1, -0.14],
    ],
    windshieldSize: [0.15, 0.08, 0.3],
    windshieldPosition: [0.18, 0.3, 0.0],
    shadowScale: [1.14, 1, 0.72],
  },
  suv: {
    bodyPoints: [
      [-0.55, -0.19],
      [-0.5, -0.26],
      [-0.2, -0.31],
      [0.2, -0.31],
      [0.45, -0.24],
      [0.56, -0.12],
      [0.56, 0.12],
      [0.45, 0.24],
      [0.2, 0.31],
      [-0.2, 0.31],
      [-0.5, 0.26],
      [-0.55, 0.19],
    ],
    cabinSize: [0.54, 0.26, 0.42],
    cabinPosition: [0.0, 0.25, 0.0],
    roofSize: [0.34, 0.12, 0.34],
    roofPosition: [0.0, 0.39, 0.0],
    noseSize: [0.14, 0.06, 0.42],
    nosePosition: [0.49, 0.09, 0.0],
    tailSize: [0.14, 0.06, 0.44],
    tailPosition: [-0.5, 0.09, 0.0],
    wheelRadius: 0.082,
    wheelWidth: 0.058,
    wheelPositions: [
      [0.3, 0.09, 0.235],
      [0.3, 0.09, -0.235],
      [-0.3, 0.09, 0.235],
      [-0.3, 0.09, -0.235],
    ],
    headLightPositions: [
      [0.56, 0.11, 0.14],
      [0.56, 0.11, -0.14],
    ],
    tailLightPositions: [
      [-0.56, 0.11, 0.15],
      [-0.56, 0.11, -0.15],
    ],
    windshieldSize: [0.16, 0.09, 0.32],
    windshieldPosition: [0.16, 0.33, 0.0],
    shadowScale: [1.18, 1, 0.76],
  },
  van: {
    bodyPoints: [
      [-0.58, -0.2],
      [-0.58, -0.28],
      [-0.2, -0.31],
      [0.28, -0.31],
      [0.52, -0.24],
      [0.58, -0.15],
      [0.58, 0.15],
      [0.52, 0.24],
      [0.28, 0.31],
      [-0.2, 0.31],
      [-0.58, 0.28],
      [-0.58, 0.2],
    ],
    cabinSize: [0.62, 0.28, 0.44],
    cabinPosition: [0.02, 0.28, 0.0],
    roofSize: [0.42, 0.12, 0.36],
    roofPosition: [0.04, 0.42, 0.0],
    noseSize: [0.12, 0.06, 0.42],
    nosePosition: [0.5, 0.1, 0.0],
    tailSize: [0.08, 0.08, 0.46],
    tailPosition: [-0.56, 0.1, 0.0],
    wheelRadius: 0.082,
    wheelWidth: 0.06,
    wheelPositions: [
      [0.28, 0.09, 0.24],
      [0.28, 0.09, -0.24],
      [-0.3, 0.09, 0.24],
      [-0.3, 0.09, -0.24],
    ],
    headLightPositions: [
      [0.58, 0.11, 0.14],
      [0.58, 0.11, -0.14],
    ],
    tailLightPositions: [
      [-0.58, 0.12, 0.16],
      [-0.58, 0.12, -0.16],
    ],
    windshieldSize: [0.13, 0.1, 0.32],
    windshieldPosition: [0.22, 0.35, 0.0],
    shadowScale: [1.22, 1, 0.8],
  },
};

function createBodyShape(points: Array<[number, number]>): THREE.Shape {
  const shape = new THREE.Shape();
  shape.moveTo(points[0][0], points[0][1]);
  for (let index = 1; index < points.length; index += 1) {
    shape.lineTo(points[index][0], points[index][1]);
  }
  shape.closePath();
  return shape;
}

function addOutlinedMesh(
  group: THREE.Group,
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  outlineColor: THREE.Color,
  outlineOpacity: number,
  position?: [number, number, number],
  rotation?: [number, number, number],
) {
  const mesh = new THREE.Mesh(geometry, material);
  if (position) {
    mesh.position.set(position[0], position[1], position[2]);
  }
  if (rotation) {
    mesh.rotation.set(rotation[0], rotation[1], rotation[2]);
  }
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(geometry),
    new THREE.LineBasicMaterial({
      color: outlineColor,
      transparent: true,
      opacity: outlineOpacity,
    }),
  );
  mesh.add(edges);
  group.add(mesh);
  return mesh;
}

function standardMaterial(
  color: THREE.ColorRepresentation,
  opacity: number,
  emissiveColor: THREE.ColorRepresentation,
  emissiveIntensity: number,
  roughness: number,
  metalness: number,
) {
  return new THREE.MeshStandardMaterial({
    color,
    transparent: opacity < 1,
    opacity,
    depthWrite: opacity >= 1,
    emissive: emissiveColor,
    emissiveIntensity,
    roughness,
    metalness,
  });
}

export function isVehicleLikeClass(objectClass: string): boolean {
  const normalized = objectClass.trim().toLowerCase();
  return normalized === "vehicle"
    || normalized === "emergency_vehicle"
    || normalized === "car"
    || normalized === "truck"
    || normalized === "bus"
    || normalized === "van";
}

export function pickVehicleVariant(trackId: number, objectClass: string): VehicleVariant {
  const normalized = objectClass.trim().toLowerCase();
  if (normalized === "emergency_vehicle") return "suv";
  if (normalized === "truck" || normalized === "bus" || normalized === "van") return "van";
  const variants: VehicleVariant[] = ["sedan", "suv", "van"];
  return variants[Math.abs(trackId) % variants.length];
}

export function buildVehicleMeshGroup({
  variant,
  bodyColor,
  accentColor,
  bodyOpacity = 1,
  bodyRoughness = 0.38,
  bodyMetalness = 0.12,
  glassColor = "#0f172a",
  glassOpacity = 0.88,
  trimColor = "#dbe7f3",
  shadowOpacity = 0.18,
  outlineOpacity = 0.92,
  emissiveIntensity = 0.12,
  headLightColor = "#d6fbff",
  tailLightColor = "#ff5d66",
}: VehicleBuildOptions): THREE.Group {
  const profile = VEHICLE_PROFILES[variant];
  const group = new THREE.Group();
  const accent = new THREE.Color(accentColor);

  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(0.6, 28),
    new THREE.MeshBasicMaterial({
      color: "#02040a",
      transparent: true,
      opacity: shadowOpacity,
      depthWrite: false,
    }),
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.01;
  shadow.scale.set(profile.shadowScale[0], profile.shadowScale[1], profile.shadowScale[2]);
  group.add(shadow);

  addOutlinedMesh(
    group,
    new THREE.ShapeGeometry(createBodyShape(profile.bodyPoints)),
    standardMaterial(bodyColor, bodyOpacity, accent, emissiveIntensity, bodyRoughness, bodyMetalness),
    accent,
    outlineOpacity,
    [0, 0.055, 0],
    [-Math.PI / 2, 0, 0],
  );

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(...profile.cabinSize),
    standardMaterial(trimColor, Math.min(bodyOpacity, 0.96), accent, emissiveIntensity * 0.8, 0.24, 0.16),
    accent,
    outlineOpacity * 0.72,
    profile.cabinPosition,
  );

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(...profile.roofSize),
    standardMaterial(glassColor, glassOpacity, accent, emissiveIntensity * 0.65, 0.14, 0.22),
    accent,
    outlineOpacity * 0.64,
    profile.roofPosition,
  );

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(...profile.noseSize),
    standardMaterial(bodyColor, bodyOpacity, accent, emissiveIntensity * 0.8, bodyRoughness, bodyMetalness),
    accent,
    outlineOpacity * 0.8,
    profile.nosePosition,
  );

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(...profile.tailSize),
    standardMaterial(bodyColor, bodyOpacity, accent, emissiveIntensity * 0.8, bodyRoughness, bodyMetalness),
    accent,
    outlineOpacity * 0.8,
    profile.tailPosition,
  );

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(...profile.windshieldSize),
    standardMaterial("#9ec5e5", 0.55, accent, emissiveIntensity * 0.48, 0.1, 0.26),
    accent,
    outlineOpacity * 0.54,
    profile.windshieldPosition,
  );

  for (const [x, y, z] of profile.headLightPositions) {
    const light = new THREE.Mesh(
      new THREE.BoxGeometry(0.045, 0.035, 0.08),
      new THREE.MeshStandardMaterial({
        color: headLightColor,
        emissive: headLightColor,
        emissiveIntensity: variant === "hero" ? 1.0 : 0.72,
        transparent: true,
        opacity: 0.95,
      }),
    );
    light.position.set(x, y, z);
    group.add(light);
  }

  for (const [x, y, z] of profile.tailLightPositions) {
    const light = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.04, 0.09),
      new THREE.MeshStandardMaterial({
        color: tailLightColor,
        emissive: tailLightColor,
        emissiveIntensity: variant === "hero" ? 1.08 : 0.82,
        transparent: true,
        opacity: 0.95,
      }),
    );
    light.position.set(x, y, z);
    group.add(light);
  }

  if (profile.roofLightBar) {
    const lightBar = new THREE.Mesh(
      new THREE.BoxGeometry(0.18, 0.04, 0.08),
      new THREE.MeshStandardMaterial({
        color: accent,
        emissive: accent,
        emissiveIntensity: 0.9,
        transparent: true,
        opacity: 0.95,
      }),
    );
    lightBar.position.set(profile.roofLightBar[0], profile.roofLightBar[1], profile.roofLightBar[2]);
    group.add(lightBar);
  }

  const wheelGeometry = new THREE.CylinderGeometry(profile.wheelRadius, profile.wheelRadius, profile.wheelWidth, 14);
  for (const [x, y, z] of profile.wheelPositions) {
    const wheel = new THREE.Mesh(
      wheelGeometry.clone(),
      new THREE.MeshStandardMaterial({
        color: "#090b10",
        roughness: 0.95,
        metalness: 0.04,
      }),
    );
    wheel.rotation.z = Math.PI * 0.5;
    wheel.position.set(x, y, z);
    group.add(wheel);
  }

  return group;
}
