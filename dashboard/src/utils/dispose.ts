import * as THREE from "three";

function disposeMaterial(material: THREE.Material | THREE.Material[] | undefined) {
  if (!material) {
    return;
  }
  if (Array.isArray(material)) {
    for (const entry of material) {
      entry.dispose();
    }
    return;
  }
  material.dispose();
}

export function disposeObject3D(root: THREE.Object3D) {
  const objects: THREE.Object3D[] = [];
  root.traverse((obj) => {
    if (obj !== root) {
      objects.push(obj);
    }
  });

  for (const obj of objects) {
    const maybeMesh = obj as THREE.Mesh;
    if (maybeMesh.geometry && "dispose" in maybeMesh.geometry) {
      maybeMesh.geometry.dispose();
    }
    disposeMaterial((maybeMesh as unknown as { material?: THREE.Material | THREE.Material[] }).material);
  }

  root.clear();
}
