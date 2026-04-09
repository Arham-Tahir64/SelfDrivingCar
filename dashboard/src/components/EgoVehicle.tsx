import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useFrameStore } from "../store/frameStore";
import { frameInterpolationAlpha, interpolateEgoPose } from "../utils/interpolation";
import { dampAngle, worldToScene, yawToScene } from "../utils/scene";
import { buildVehicleMeshGroup } from "./vehicleMeshes";

const POSITION_SMOOTHING = 0.22;
const ROTATION_SMOOTHING = 0.18;

export default function EgoVehicle() {
  const groupRef = useRef<THREE.Group>(null);
  const targetPosition = useRef(new THREE.Vector3());
  const initialized = useRef(false);
  const vehicleModel = useMemo(
    () =>
      buildVehicleMeshGroup({
        variant: "hero",
        bodyColor: "#f5f7fa",
        accentColor: "#00e5ff",
        trimColor: "#0f172a",
        glassColor: "#08111c",
        bodyOpacity: 1,
        bodyRoughness: 0.34,
        bodyMetalness: 0.14,
        glassOpacity: 0.9,
        shadowOpacity: 0.24,
        emissiveIntensity: 0.18,
        headLightColor: "#d8fcff",
        tailLightColor: "#ff5768",
      }),
    [],
  );

  useFrame(() => {
    const state = useFrameStore.getState();
    const alpha = frameInterpolationAlpha(
      state.currentFrameReceivedAtMs,
      state.frameIntervalMs,
      performance.now(),
    );
    const ego = interpolateEgoPose(
      state.previousFrame?.["localization/ego_pose"],
      state.currentFrame?.["localization/ego_pose"],
      alpha,
    );
    if (!ego || !groupRef.current) return;

    targetPosition.current.copy(worldToScene(ego.world_xyz));

    if (!initialized.current) {
      groupRef.current.position.copy(targetPosition.current);
      groupRef.current.rotation.y = yawToScene(ego.yaw_rad);
      initialized.current = true;
    } else {
      groupRef.current.position.lerp(targetPosition.current, POSITION_SMOOTHING);
      groupRef.current.rotation.y = dampAngle(
        groupRef.current.rotation.y,
        yawToScene(ego.yaw_rad),
        ROTATION_SMOOTHING,
      );
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={vehicleModel} />
    </group>
  );
}
