import * as THREE from "three";

function addOutlinedMesh(
  group: THREE.Group,
  geometry: THREE.BufferGeometry,
  material: THREE.MeshStandardMaterial,
  outlineColor: THREE.Color,
  outlineOpacity = 0.95,
) {
  const mesh = new THREE.Mesh(geometry, material);
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
  color: THREE.Color,
  opacity: number,
  outlineColor: THREE.Color,
  emissiveIntensity: number,
) {
  return new THREE.MeshStandardMaterial({
    color,
    transparent: true,
    opacity,
    depthWrite: false,
    emissive: outlineColor,
    emissiveIntensity,
    roughness: 0.45,
    metalness: 0.08,
  });
}

function buildVehicleProxy(
  bodyColor: THREE.Color,
  outlineColor: THREE.Color,
  opacity: number,
) {
  const group = new THREE.Group();

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(0.94, 0.26, 0.42),
    standardMaterial(bodyColor, opacity, outlineColor, 0.16),
    outlineColor,
  );

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(0.42, 0.2, 0.36),
    standardMaterial(new THREE.Color("#dbe7f3"), Math.min(opacity + 0.05, 0.85), outlineColor, 0.12),
    outlineColor,
  ).position.set(0.05, 0.46, 0.0);

  const windshield = new THREE.Mesh(
    new THREE.BoxGeometry(0.18, 0.12, 0.34),
    new THREE.MeshStandardMaterial({
      color: new THREE.Color("#8ab4d6"),
      transparent: true,
      opacity: 0.52,
      roughness: 0.12,
      metalness: 0.25,
      emissive: outlineColor,
      emissiveIntensity: 0.08,
    }),
  );
  windshield.position.set(0.18, 0.52, 0.0);
  group.add(windshield);

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(0.1, 0.08, 0.38),
    standardMaterial(bodyColor, opacity, outlineColor, 0.12),
    outlineColor,
  ).position.set(0.46, 0.18, 0.0);

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(0.08, 0.08, 0.38),
    standardMaterial(bodyColor, opacity, outlineColor, 0.12),
    outlineColor,
  ).position.set(-0.46, 0.18, 0.0);

  const wheelGeometry = new THREE.CylinderGeometry(0.09, 0.09, 0.06, 14);
  const wheelMaterial = new THREE.MeshStandardMaterial({
    color: new THREE.Color("#0b0d12"),
    roughness: 0.95,
    metalness: 0.02,
    emissive: outlineColor,
    emissiveIntensity: 0.02,
  });
  for (const [x, z] of [
    [0.32, 0.23],
    [0.32, -0.23],
    [-0.32, 0.23],
    [-0.32, -0.23],
  ]) {
    const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial.clone());
    wheel.rotation.z = Math.PI * 0.5;
    wheel.position.set(x, 0.11, z);
    group.add(wheel);
  }

  return group;
}

function buildPedestrianProxy(
  bodyColor: THREE.Color,
  outlineColor: THREE.Color,
  opacity: number,
) {
  const group = new THREE.Group();

  addOutlinedMesh(
    group,
    new THREE.CylinderGeometry(0.12, 0.14, 0.42, 10),
    standardMaterial(bodyColor, opacity, outlineColor, 0.14),
    outlineColor,
  ).position.set(0.0, 0.46, 0.0);

  addOutlinedMesh(
    group,
    new THREE.SphereGeometry(0.13, 14, 14),
    standardMaterial(new THREE.Color("#f3d9c3"), Math.min(opacity + 0.05, 0.9), outlineColor, 0.08),
    outlineColor,
  ).position.set(0.0, 0.84, 0.0);

  for (const x of [-0.07, 0.07]) {
    const leg = new THREE.Mesh(
      new THREE.CylinderGeometry(0.04, 0.05, 0.34, 8),
      new THREE.MeshStandardMaterial({
        color: bodyColor,
        transparent: true,
        opacity,
        roughness: 0.6,
        metalness: 0.05,
        emissive: outlineColor,
        emissiveIntensity: 0.12,
      }),
    );
    leg.position.set(x, 0.17, 0.0);
    group.add(leg);
  }

  for (const [x, angle] of [
    [-0.16, 0.28],
    [0.16, -0.28],
  ]) {
    const arm = new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.04, 0.24, 8),
      new THREE.MeshStandardMaterial({
        color: bodyColor,
        transparent: true,
        opacity,
        roughness: 0.6,
        metalness: 0.05,
        emissive: outlineColor,
        emissiveIntensity: 0.12,
      }),
    );
    arm.rotation.z = angle;
    arm.position.set(x, 0.54, 0.0);
    group.add(arm);
  }

  return group;
}

function buildCyclistProxy(
  bodyColor: THREE.Color,
  outlineColor: THREE.Color,
  opacity: number,
) {
  const group = new THREE.Group();
  const wheelGeometry = new THREE.CylinderGeometry(0.11, 0.11, 0.04, 14);
  const wheelMaterial = new THREE.MeshStandardMaterial({
    color: new THREE.Color("#0b0d12"),
    roughness: 0.96,
    metalness: 0.02,
    emissive: outlineColor,
    emissiveIntensity: 0.02,
  });

  for (const x of [-0.25, 0.25]) {
    const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial.clone());
    wheel.rotation.z = Math.PI * 0.5;
    wheel.position.set(x, 0.12, 0.0);
    group.add(wheel);
  }

  for (const [startX, startY, startZ, endX, endY, endZ] of [
    [-0.18, 0.17, 0.0, 0.0, 0.32, 0.0],
    [0.18, 0.17, 0.0, 0.0, 0.32, 0.0],
    [-0.18, 0.17, 0.0, 0.24, 0.22, 0.0],
    [0.0, 0.32, 0.0, 0.24, 0.22, 0.0],
    [0.0, 0.32, 0.0, 0.0, 0.55, 0.0],
  ]) {
    const start = new THREE.Vector3(startX, startY, startZ);
    const end = new THREE.Vector3(endX, endY, endZ);
    const direction = new THREE.Vector3().subVectors(end, start);
    const length = direction.length();
    const geom = new THREE.CylinderGeometry(0.018, 0.018, Math.max(length, 0.02), 8);
    const mesh = new THREE.Mesh(
      geom,
      new THREE.MeshStandardMaterial({
        color: bodyColor,
        transparent: true,
        opacity,
        roughness: 0.55,
        metalness: 0.05,
        emissive: outlineColor,
        emissiveIntensity: 0.13,
      }),
    );
    mesh.position.copy(start).add(end).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      direction.clone().normalize(),
    );
    group.add(mesh);
  }

  addOutlinedMesh(
    group,
    new THREE.CylinderGeometry(0.08, 0.09, 0.22, 8),
    standardMaterial(bodyColor, opacity, outlineColor, 0.13),
    outlineColor,
  ).position.set(0.0, 0.48, 0.0);

  addOutlinedMesh(
    group,
    new THREE.SphereGeometry(0.1, 12, 12),
    standardMaterial(new THREE.Color("#f3d9c3"), Math.min(opacity + 0.05, 0.9), outlineColor, 0.08),
    outlineColor,
  ).position.set(0.0, 0.66, 0.0);

  return group;
}

function buildTrafficLightProxy(
  bodyColor: THREE.Color,
  outlineColor: THREE.Color,
  opacity: number,
) {
  const group = new THREE.Group();
  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.025, 0.03, 0.55, 8),
    new THREE.MeshStandardMaterial({
      color: new THREE.Color("#39414f"),
      transparent: true,
      opacity,
      roughness: 0.7,
      metalness: 0.08,
      emissive: outlineColor,
      emissiveIntensity: 0.05,
    }),
  );
  pole.position.set(0.0, 0.28, 0.0);
  group.add(pole);

  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(0.11, 0.22, 0.11),
    standardMaterial(bodyColor, opacity, outlineColor, 0.13),
    outlineColor,
  ).position.set(0.0, 0.65, 0.0);

  for (const y of [0.72, 0.65, 0.58]) {
    const light = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 10, 10),
      new THREE.MeshStandardMaterial({
        color: outlineColor,
        emissive: outlineColor,
        emissiveIntensity: 1.0,
      }),
    );
    light.position.set(0.0, y, 0.06);
    group.add(light);
  }

  return group;
}

function buildFallbackProxy(
  bodyColor: THREE.Color,
  outlineColor: THREE.Color,
  opacity: number,
) {
  const group = new THREE.Group();
  addOutlinedMesh(
    group,
    new THREE.BoxGeometry(1.0, 1.0, 0.6),
    standardMaterial(bodyColor, opacity, outlineColor, 0.12),
    outlineColor,
  );
  return group;
}

export function buildAgentProxy(
  objectClass: string,
  bodyColor: THREE.Color,
  outlineColor: THREE.Color,
  opacity: number,
) {
  const normalized = objectClass.trim().toLowerCase();
  switch (normalized) {
    case "vehicle":
    case "emergency_vehicle":
    case "car":
    case "truck":
    case "bus":
    case "van":
      return buildVehicleProxy(bodyColor, outlineColor, opacity);
    case "pedestrian":
    case "person":
    case "walker":
      return buildPedestrianProxy(bodyColor, outlineColor, opacity);
    case "cyclist":
    case "bike":
    case "bicycle":
    case "motorbike":
    case "motobike":
      return buildCyclistProxy(bodyColor, outlineColor, opacity);
    case "traffic_light":
    case "traffic_light_red":
    case "traffic_light_orange":
    case "traffic_light_green":
      return buildTrafficLightProxy(bodyColor, outlineColor, opacity);
    default:
      return buildFallbackProxy(bodyColor, outlineColor, opacity);
  }
}
