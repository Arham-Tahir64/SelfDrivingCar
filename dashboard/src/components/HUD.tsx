import type { CSSProperties } from "react";
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

const LATENCY_BUDGET_MS = 100;
const GAUGE_START = 160;
const GAUGE_SWEEP = 220;
const MAX_SPEED_MPH = 80;
const SPEED_ZONES = [
  { startMph: 0, endMph: 35, color: "#5CFF95" },
  { startMph: 35, endMph: 55, color: "#FFC145" },
  { startMph: 55, endMph: MAX_SPEED_MPH, color: "#FF5D5D" },
] as const;

const HUD_GLASS_CARD: CSSProperties = {
  background: "rgba(10,14,24,0.75)",
  backdropFilter: "blur(10px)",
  WebkitBackdropFilter: "blur(10px)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 10,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function polarToXY(cx: number, cy: number, r: number, angleDeg: number) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, sweep: number): string {
  if (sweep <= 0) return "";
  const s = polarToXY(cx, cy, r, startAngle);
  const e = polarToXY(cx, cy, r, startAngle + sweep);
  const large = sweep > 180 ? 1 : 0;
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`;
}

function zoneFillPercent(mph: number, startMph: number, endMph: number): number {
  const zoneRange = endMph - startMph;
  if (zoneRange <= 0) return 0;
  return clamp(((mph - startMph) / zoneRange) * 100, 0, 100);
}

function SpeedGauge({ speedMps }: { speedMps: number }) {
  const mph = clamp(speedMps * 2.237, 0, MAX_SPEED_MPH);
  const cx = 80;
  const cy = 78;
  const r = 58;
  const trackPath = describeArc(cx, cy, r, GAUGE_START, GAUGE_SWEEP);
  const startMarker = polarToXY(cx, cy, r + 18, GAUGE_START);
  const endMarker = polarToXY(cx, cy, r + 18, GAUGE_START + GAUGE_SWEEP);

  return (
    <svg width={160} height={132} viewBox="0 0 160 132" style={{ overflow: "visible", display: "block" }}>
      <path d={trackPath} fill="none" stroke="rgba(18,24,36,0.95)" strokeWidth={12} strokeLinecap="round" />

      {SPEED_ZONES.map((zone) => {
        const zoneStartAngle = GAUGE_START + (zone.startMph / MAX_SPEED_MPH) * GAUGE_SWEEP;
        const zoneSweep = ((zone.endMph - zone.startMph) / MAX_SPEED_MPH) * GAUGE_SWEEP;
        const path = describeArc(cx, cy, r, zoneStartAngle, zoneSweep);
        const fillPercent = zoneFillPercent(mph, zone.startMph, zone.endMph);

        return (
          <g key={`${zone.startMph}-${zone.endMph}`}>
            <path
              d={path}
              fill="none"
              stroke={zone.color}
              strokeWidth={8}
              strokeLinecap="round"
              opacity={0.24}
            />
            {fillPercent > 0 ? (
              <path
                d={path}
                fill="none"
                stroke={zone.color}
                strokeWidth={8}
                strokeLinecap="round"
                pathLength={100}
                strokeDasharray={`${fillPercent} 100`}
                style={{
                  filter: `drop-shadow(0 0 8px ${zone.color}aa)`,
                  transition: "stroke-dasharray 240ms ease, stroke 240ms ease, filter 240ms ease",
                }}
              />
            ) : null}
          </g>
        );
      })}

      <circle cx={cx} cy={cy} r={40} fill="rgba(7,10,16,0.76)" stroke="rgba(255,255,255,0.06)" />

      <text
        x={cx}
        y={cy - 4}
        textAnchor="middle"
        dominantBaseline="central"
        fill="#F5F7FB"
        fontSize={30}
        fontWeight={700}
        fontFamily="'Segoe UI', 'Inter', sans-serif"
      >
        {Math.round(mph)}
      </text>
      <text
        x={cx}
        y={cy + 22}
        textAnchor="middle"
        fill="#79849B"
        fontSize={9}
        fontWeight={700}
        letterSpacing={2}
        fontFamily="'Segoe UI', 'Inter', sans-serif"
      >
        MPH
      </text>

      <text
        x={startMarker.x}
        y={startMarker.y + 3}
        textAnchor="middle"
        fill="#46536B"
        fontSize={9}
        fontWeight={700}
        fontFamily="monospace"
      >
        0
      </text>
      <text
        x={endMarker.x}
        y={endMarker.y + 3}
        textAnchor="middle"
        fill="#46536B"
        fontSize={9}
        fontWeight={700}
        fontFamily="monospace"
      >
        {MAX_SPEED_MPH}
      </text>
    </svg>
  );
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

function latencyColor(ms: number): string {
  if (ms <= 15) return "#00C853";
  if (ms <= 40) return "#FFC107";
  return "#F44336";
}

export default function HUD() {
  const frame = useFrameStore((s) => s.currentFrame);
  const connected = useFrameStore((s) => s.connected);

  const ego = frame?.["localization/ego_pose"];
  const traj = frame?.["planning/ego_trajectory"];
  const control = frame?.["control/vehicle_command"];
  const scenario = frame?.["system/scenario_info"];
  const perception = frame?.["perception/status"];
  const latency = frame?.["pipeline/latency"];
  const detections = frame?.["perception/detections"];
  const laneCount =
    frame?.["perception/lanes"]?.length ?? frame?.["map/local_map"]?.perceived_lanes?.length ?? 0;
  const behaviorState = traj?.behavior_state ?? "--";
  const behaviorColor = BEHAVIOR_COLORS[behaviorState] ?? "#888";
  const fallbackColor = perceptionFallbackColor(perception?.fallback_state);
  const laneBadgeColor = laneCount > 0 ? modalityColorCss("camera") : "#F44336";
  const modalityCounts = perception
    ? Object.entries(perception.counts_by_modality)
        .map(([key, value]) => `${key}:${value}`)
        .join(" ")
    : "";

  const detectionCounts: Record<string, number> = {};
  if (detections) {
    for (const d of detections) {
      detectionCounts[d.object_class] = (detectionCounts[d.object_class] ?? 0) + 1;
    }
  }
  const totalDetections = detections?.length ?? 0;

  return (
    <div style={styles.container}>
      <div style={styles.topLeft}>
        {scenario && (
          <div style={{ color: "#B2BAC8", fontSize: 13, marginBottom: 4 }}>
            {scenario.scenario_id}: {scenario.name}
          </div>
        )}
        {frame && (
          <div style={{ color: "#6E7A90", fontSize: 11 }}>
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
            <div style={{ color: "#8B93A7", fontSize: 11 }}>
              {perception.fallback_state}
              {modalityCounts ? ` | ${modalityCounts}` : ""}
            </div>
            <div
              style={{
                padding: "4px 10px",
                borderRadius: 999,
                border: `1px solid ${laneBadgeColor}`,
                color: laneBadgeColor,
                backgroundColor: `${laneBadgeColor}22`,
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: 0.4,
              }}
            >
              {laneCount > 0 ? "CAMERA LANES" : "LANE DEGRADED"}
            </div>
          </div>
        )}
        {totalDetections > 0 && (
          <div style={{ marginTop: 10, display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ color: "#687487", fontSize: 10, fontWeight: 700, letterSpacing: 0.5 }}>
              OBJECTS
            </span>
            {Object.entries(detectionCounts).map(([cls, count]) => (
              <div
                key={cls}
                style={{
                  padding: "2px 8px",
                  borderRadius: 4,
                  backgroundColor: "#ffffff0d",
                  border: "1px solid #ffffff18",
                  color: "#C2C9D5",
                  fontSize: 10,
                }}
              >
                {cls}: {count}
              </div>
            ))}
            <div style={{ color: "#687487", fontSize: 10 }}>({totalDetections} total)</div>
          </div>
        )}
      </div>

      <div style={styles.bottomLeft}>
        <SpeedGauge speedMps={ego?.speed_mps ?? 0} />
        <div
          style={{
            ...styles.behaviorBadge,
            backgroundColor: `${behaviorColor}22`,
            borderColor: behaviorColor,
            color: behaviorColor,
            boxShadow: `0 0 12px ${behaviorColor}33`,
          }}
        >
          {behaviorState}
        </div>
      </div>

      {latency && (
        <div style={styles.latencyPanel}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
            <span style={{ fontSize: 10, color: "#6E7A90", fontWeight: 700, letterSpacing: 0.5 }}>LATENCY</span>
            <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  backgroundColor: connected ? "#00C853" : "#F44336",
                }}
              />
              <span style={{ color: "#8B93A7", fontSize: 9 }}>{connected ? "LIVE" : "DISC"}</span>
            </div>
          </div>
          {(["perception", "localization", "mapping", "prediction", "planning", "control"] as const).map(
            (module) => {
              const ms = latency[module] ?? 0;
              return <LatencyBar key={module} label={module.slice(0, 5).toUpperCase()} ms={ms} />;
            },
          )}
          {(latency.perception_aux_total ?? 0) > 0 && (
            <LatencyBar label="P-AUX" ms={latency.perception_aux_total ?? 0} />
          )}
          <div
            style={{
              marginTop: 4,
              paddingTop: 4,
              borderTop: "1px solid rgba(255,255,255,0.06)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span style={{ fontSize: 10, color: "#8B93A7" }}>TOTAL</span>
            <span style={{ fontSize: 11, fontWeight: 700, color: latencyColor(latency.total ?? 0) }}>
              {(latency.total ?? 0).toFixed(1)}ms
            </span>
          </div>
          <div style={{ marginTop: 4, width: "100%", height: 3, backgroundColor: "#111520", borderRadius: 2 }}>
            <div
              style={{
                width: `${Math.min(100, ((latency.total ?? 0) / LATENCY_BUDGET_MS) * 100)}%`,
                height: "100%",
                backgroundColor: latencyColor(latency.total ?? 0),
                borderRadius: 2,
                transition: "width 80ms",
                boxShadow: `0 0 4px ${latencyColor(latency.total ?? 0)}88`,
              }}
            />
          </div>
          <div style={{ fontSize: 9, color: "#49576B", marginTop: 2, textAlign: "right" }}>
            /{LATENCY_BUDGET_MS}ms budget
          </div>
        </div>
      )}

      {control && (
        <div style={styles.bottomRight}>
          <ControlBar label="THR" value={control.throttle} color="#00E5FF" />
          <ControlBar label="BRK" value={control.brake} color="#F44336" />
          <ControlBar label="STR" value={(control.steer + 1) / 2} color="#FFC107" />
        </div>
      )}

      {control?.emergency_override && <div style={styles.emergencyBanner}>EMERGENCY BRAKE</div>}
    </div>
  );
}

function LatencyBar({ label, ms }: { label: string; ms: number }) {
  const maxWidth = 60;
  const barWidth = Math.min(maxWidth, (ms / 60) * maxWidth);
  const color = latencyColor(ms);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
      <span style={{ fontSize: 9, color: "#667285", width: 30, textAlign: "right" }}>{label}</span>
      <div style={{ width: maxWidth, height: 4, backgroundColor: "#111520", borderRadius: 2 }}>
        <div
          style={{
            width: barWidth,
            height: "100%",
            backgroundColor: color,
            borderRadius: 2,
            transition: "width 80ms",
          }}
        />
      </div>
      <span style={{ fontSize: 9, color: "#8B93A7", width: 36 }}>{ms.toFixed(1)}ms</span>
    </div>
  );
}

function ControlBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 10, color: "#888", marginBottom: 2 }}>{label}</div>
      <div style={{ width: 80, height: 6, backgroundColor: "#222", borderRadius: 3, overflow: "hidden" }}>
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

const styles: Record<string, CSSProperties> = {
  container: {
    position: "absolute",
    inset: 0,
    pointerEvents: "none",
    fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
  },
  topLeft: {
    ...HUD_GLASS_CARD,
    position: "absolute",
    top: 16,
    left: 16,
    padding: "10px 14px",
    minWidth: 200,
  },
  bottomLeft: {
    position: "absolute",
    bottom: 44,
    left: 12,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 6,
  },
  behaviorBadge: {
    padding: "4px 12px",
    borderRadius: 4,
    border: "1px solid transparent",
    fontSize: 11,
    fontWeight: 600,
    letterSpacing: 0.5,
    textAlign: "center",
    transition: "background 400ms ease, border-color 400ms ease, color 400ms ease",
    minWidth: 112,
  },
  latencyPanel: {
    ...HUD_GLASS_CARD,
    position: "absolute",
    top: 16,
    right: 16,
    padding: "10px 14px",
    minWidth: 160,
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
