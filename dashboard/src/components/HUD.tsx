import { modalityColorCss } from "../utils/colors";
import { useFrameStore } from "../store/frameStore";

const BEHAVIOR_COLORS: Record<string, string> = {
  LANE_KEEP: "#00E5FF",
  PREPARE_MERGE: "#FFC107",
  MERGING: "#FF9800",
  INTERSECTION_APPROACH: "#AB47BC",
  STOPPING_FOR_RED: "#F44336",
  PEDESTRIAN_YIELD: "#66BB6A",
  EMERGENCY_YIELD: "#FF1744",
  CONSTRUCTION_NAVIGATE: "#FF6D00",
  GOAL_REACHED: "#00C853",
};

function mpsToMph(mps: number): string {
  return (mps * 2.237).toFixed(1);
}

function perceptionFallbackColor(fallbackState: string | undefined): string {
  switch (fallbackState) {
    case "fused":
      return modalityColorCss("fused");
    case "lidar_only":
      return modalityColorCss("lidar");
    case "camera_only":
      return modalityColorCss("camera");
    case "bootstrap":
      return modalityColorCss("bootstrap");
    default:
      return "#888";
  }
}

export default function HUD() {
  const frame = useFrameStore((s) => s.currentFrame);
  const connected = useFrameStore((s) => s.connected);

  const ego = frame?.["localization/ego_pose"];
  const traj = frame?.["planning/ego_trajectory"];
  const control = frame?.["control/vehicle_command"];
  const scenario = frame?.["system/scenario_info"];
  const perception = frame?.["perception/status"];
  const behaviorState = traj?.behavior_state ?? "--";
  const behaviorColor = BEHAVIOR_COLORS[behaviorState] ?? "#888";
  const fallbackColor = perceptionFallbackColor(perception?.fallback_state);
  const modalityCounts = perception
    ? Object.entries(perception.counts_by_modality)
        .map(([key, value]) => `${key}:${value}`)
        .join(" ")
    : "";

  return (
    <div style={styles.container}>
      <div style={styles.connectionDot}>
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            backgroundColor: connected ? "#00C853" : "#F44336",
            marginRight: 6,
          }}
        />
        <span style={{ color: "#888", fontSize: 11 }}>
          {connected ? "LIVE" : "DISCONNECTED"}
        </span>
      </div>

      <div style={styles.topLeft}>
        {scenario && (
          <div style={{ color: "#aaa", fontSize: 13, marginBottom: 4 }}>
            {scenario.scenario_id}: {scenario.name}
          </div>
        )}
        {frame && (
          <div style={{ color: "#666", fontSize: 11 }}>
            Tick {frame.tick_id} &middot; {frame.sim_time_s.toFixed(1)}s
          </div>
        )}
        {perception && (
          <div style={{ display: "flex", gap: 8, marginTop: 10, alignItems: "center", flexWrap: "wrap" }}>
            <div
              style={{
                padding: "4px 10px",
                borderRadius: 999,
                border: `1px solid ${fallbackColor}`,
                color: fallbackColor,
                backgroundColor: `${fallbackColor}22`,
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: 0.4,
              }}
            >
              {perception.active_mode.toUpperCase()}
            </div>
            <div style={{ color: "#8b93a7", fontSize: 11 }}>
              {perception.fallback_state}
              {modalityCounts ? ` | ${modalityCounts}` : ""}
            </div>
          </div>
        )}
      </div>

      <div style={styles.bottomLeft}>
        <div style={{ fontSize: 42, fontWeight: 700, color: "#fff", lineHeight: 1 }}>
          {ego ? mpsToMph(ego.speed_mps) : "--"}
        </div>
        <div style={{ fontSize: 12, color: "#888", marginTop: 2 }}>MPH</div>
        <div
          style={{
            marginTop: 8,
            padding: "4px 10px",
            borderRadius: 4,
            backgroundColor: behaviorColor + "22",
            border: `1px solid ${behaviorColor}`,
            color: behaviorColor,
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: 0.5,
          }}
        >
          {behaviorState}
        </div>
      </div>

      {control && (
        <div style={styles.bottomRight}>
          <ControlBar label="THR" value={control.throttle} color="#00E5FF" />
          <ControlBar label="BRK" value={control.brake} color="#F44336" />
          <ControlBar label="STR" value={(control.steer + 1) / 2} color="#FFC107" />
        </div>
      )}

      {control?.emergency_override && (
        <div style={styles.emergencyBanner}>EMERGENCY BRAKE</div>
      )}
    </div>
  );
}

function ControlBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 10, color: "#888", marginBottom: 2 }}>{label}</div>
      <div
        style={{
          width: 80,
          height: 6,
          backgroundColor: "#222",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.min(Math.max(value, 0), 1) * 100}%`,
            height: "100%",
            backgroundColor: color,
            borderRadius: 3,
            transition: "width 50ms",
          }}
        />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
    fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
  },
  connectionDot: {
    position: "absolute",
    top: 12,
    right: 16,
    display: "flex",
    alignItems: "center",
  },
  topLeft: {
    position: "absolute",
    top: 16,
    left: 20,
  },
  bottomLeft: {
    position: "absolute",
    bottom: 24,
    left: 24,
  },
  bottomRight: {
    position: "absolute",
    bottom: 24,
    right: 24,
  },
  emergencyBanner: {
    position: "absolute",
    top: 16,
    left: "50%",
    transform: "translateX(-50%)",
    backgroundColor: "#F44336",
    color: "white",
    padding: "8px 24px",
    borderRadius: 6,
    fontSize: 16,
    fontWeight: 700,
    letterSpacing: 1,
    animation: "pulse 0.5s infinite alternate",
  },
};
