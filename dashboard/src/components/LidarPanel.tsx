import { useRef, useEffect } from "react";
import { useFrameStore } from "../store/frameStore";
import type { LidarPanelObject } from "../utils/types";

const CANVAS_W = 400;
const CANVAS_H = 300;
const RANGE_M = 50;
const EGO_Y = CANVAS_H * 0.76;
const POINT_RADIUS = 1.0;

function toCanvas(forwardM: number, lateralM: number): [number, number] {
  const x = CANVAS_W / 2 + (lateralM / RANGE_M) * (CANVAS_W / 2);
  const y = EGO_Y - (forwardM / RANGE_M) * (CANVAS_H * 0.9);
  return [x, y];
}

function centroid(points: number[][]): [number, number] {
  if (!points.length) return [0, 0];
  let sx = 0;
  let sy = 0;
  for (const [x, y] of points) {
    sx += x;
    sy += y;
  }
  return [sx / points.length, sy / points.length];
}

function objectStroke(object: LidarPanelObject, threatIds: number[]): string {
  if (threatIds.includes(object.track_id)) {
    return object.threat_rank === 1 ? "#FF6B57" : "#FFB347";
  }
  if (object.is_path_relevant) return "#49D7FF";
  if (object.track_state !== "CONFIRMED") return "rgba(167, 179, 198, 0.45)";
  return "rgba(154, 164, 178, 0.7)";
}

function objectFill(object: LidarPanelObject, threatIds: number[]): string {
  if (threatIds.includes(object.track_id)) {
    return object.threat_rank === 1 ? "rgba(255, 107, 87, 0.18)" : "rgba(255, 179, 71, 0.16)";
  }
  if (object.is_path_relevant) return "rgba(73, 215, 255, 0.12)";
  if (object.track_state !== "CONFIRMED") return "rgba(124, 134, 150, 0.07)";
  return "rgba(124, 134, 150, 0.1)";
}

function drawArrow(
  ctx: CanvasRenderingContext2D,
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  color: string,
  width: number,
) {
  const angle = Math.atan2(endY - startY, endX - startX);
  const headLength = 7;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
  ctx.lineTo(endX, endY);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(
    endX - headLength * Math.cos(angle - Math.PI / 6),
    endY - headLength * Math.sin(angle - Math.PI / 6),
  );
  ctx.lineTo(
    endX - headLength * Math.cos(angle + Math.PI / 6),
    endY - headLength * Math.sin(angle + Math.PI / 6),
  );
  ctx.closePath();
  ctx.fill();
}

function labelForObject(object: LidarPanelObject): string {
  const prefix = object.object_class?.[0]?.toUpperCase() ?? "O";
  return `${prefix}${object.track_id}`;
}

export default function LidarPanel() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lidarData = useFrameStore((s) => s.currentFrame?.["visualization/lidar_preview"]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
    ctx.fillStyle = "#091019";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    const forwardCone = lidarData?.forward_cone ?? { length_m: 28, half_angle_deg: 18 };
    const pathPolyline = lidarData?.path_polyline_xy ?? [];
    const objects = lidarData?.objects ?? [];
    const threatIds = lidarData?.threat_ids ?? [];
    const status = lidarData?.status;

    ctx.strokeStyle = "#162030";
    ctx.lineWidth = 0.7;
    for (let r = 10; r <= RANGE_M; r += 10) {
      const radiusX = (r / RANGE_M) * (CANVAS_W * 0.48);
      const radiusY = (r / RANGE_M) * (CANVAS_H * 0.42);
      ctx.beginPath();
      ctx.ellipse(CANVAS_W / 2, EGO_Y, radiusX, radiusY, 0, Math.PI, 0, true);
      ctx.stroke();
    }

    ctx.strokeStyle = "#101924";
    ctx.beginPath();
    ctx.moveTo(CANVAS_W / 2, 0);
    ctx.lineTo(CANVAS_W / 2, CANVAS_H);
    ctx.moveTo(0, EGO_Y);
    ctx.lineTo(CANVAS_W, EGO_Y);
    ctx.stroke();

    const coneLeft = toCanvas(
      forwardCone.length_m,
      Math.tan((forwardCone.half_angle_deg * Math.PI) / 180) * forwardCone.length_m,
    );
    const coneRight = toCanvas(
      forwardCone.length_m,
      -Math.tan((forwardCone.half_angle_deg * Math.PI) / 180) * forwardCone.length_m,
    );
    ctx.fillStyle = "rgba(16, 123, 160, 0.08)";
    ctx.beginPath();
    ctx.moveTo(CANVAS_W / 2, EGO_Y);
    ctx.lineTo(coneLeft[0], coneLeft[1]);
    ctx.lineTo(coneRight[0], coneRight[1]);
    ctx.closePath();
    ctx.fill();

    if (pathPolyline.length >= 2) {
      ctx.strokeStyle = "rgba(73, 215, 255, 0.22)";
      ctx.lineWidth = 8;
      ctx.beginPath();
      pathPolyline.forEach(([forwardM, lateralM], index) => {
        const [x, y] = toCanvas(forwardM, lateralM);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.strokeStyle = "rgba(73, 215, 255, 0.55)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      pathPolyline.forEach(([forwardM, lateralM], index) => {
        const [x, y] = toCanvas(forwardM, lateralM);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    if (lidarData?.points?.length) {
      for (const point of lidarData.points) {
        const forwardM = point[0] ?? 0;
        const lateralM = point[1] ?? 0;
        const z = point[2] ?? 0;
        if (forwardM < -8 || forwardM > RANGE_M || Math.abs(lateralM) > RANGE_M) continue;
        const [x, y] = toCanvas(forwardM, lateralM);
        const alpha = Math.max(0.18, Math.min(0.42, 0.24 + ((z + 1.0) * 0.03)));
        ctx.fillStyle = `rgba(95, 183, 255, ${alpha})`;
        ctx.beginPath();
        ctx.arc(x, y, POINT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    for (const object of objects) {
      const footprint = object.footprint_xy ?? [];
      if (footprint.length < 3) continue;
      const polygon = footprint.map(([forwardM, lateralM]) => toCanvas(forwardM, lateralM));
      const stroke = objectStroke(object, threatIds);
      const fill = objectFill(object, threatIds);

      ctx.fillStyle = fill;
      ctx.strokeStyle = stroke;
      ctx.lineWidth = object.track_state === "CONFIRMED" ? 1.8 : 1.0;
      ctx.beginPath();
      polygon.forEach(([x, y], index) => {
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      const [centroidForward, centroidLateral] = object.centroid_xy ?? centroid(footprint);
      const [cx, cy] = toCanvas(centroidForward, centroidLateral);

      if (object.track_state === "CONFIRMED" && (object.speed_mps ?? 0) > 0.6) {
        const [vx, vy] = object.velocity_xy ?? [0, 0];
        const arrowForward = Math.max(-3.0, Math.min(6.0, vx * 0.7));
        const arrowLateral = Math.max(-4.0, Math.min(4.0, vy * 0.7));
        const [ex, ey] = toCanvas(centroidForward + arrowForward, centroidLateral + arrowLateral);
        drawArrow(ctx, cx, cy, ex, ey, stroke, threatIds.includes(object.track_id) ? 2 : 1.3);
      }

      if (
        object.ghost_xy &&
        threatIds.includes(object.track_id) &&
        object.track_state === "CONFIRMED"
      ) {
        const [gx, gy] = toCanvas(object.ghost_xy[0] ?? 0, object.ghost_xy[1] ?? 0);
        ctx.strokeStyle = "rgba(255, 208, 122, 0.8)";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(gx, gy, 4.5, 0, Math.PI * 2);
        ctx.stroke();
      }

      const shouldLabel =
        object.track_state === "CONFIRMED" &&
        (threatIds.includes(object.track_id) ||
          (object.is_path_relevant && (object.relevance_score ?? 0) >= 0.72));
      if (shouldLabel) {
        ctx.font = "10px Consolas, monospace";
        ctx.textAlign = "left";
        ctx.fillStyle = threatIds.includes(object.track_id) ? "#FFE0B2" : "#C8F7FF";
        ctx.fillText(labelForObject(object), cx + 5, cy - 6);
      }
    }

    ctx.fillStyle = "#00E5FF";
    ctx.beginPath();
    ctx.moveTo(CANVAS_W / 2, EGO_Y - 10);
    ctx.lineTo(CANVAS_W / 2 - 6, EGO_Y + 6);
    ctx.lineTo(CANVAS_W / 2 + 6, EGO_Y + 6);
    ctx.closePath();
    ctx.fill();

    ctx.font = "10px Consolas, monospace";
    ctx.textAlign = "left";
    ctx.fillStyle = "#8BA3BD";
    ctx.fillText(`${status?.point_count ?? 0} pts`, 12, CANVAS_H - 12);
    ctx.fillText(`${status?.confirmed_track_count ?? 0} confirmed`, 74, CANVAS_H - 12);
    ctx.fillText(`${threatIds.length} threats`, 190, CANVAS_H - 12);

    if (!lidarData?.points?.length && objects.length === 0) {
      ctx.fillStyle = "#5A6472";
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for LiDAR...", CANVAS_W / 2, CANVAS_H / 2);
    }
  }, [lidarData]);

  const modeLabel = lidarData?.status?.mode?.toUpperCase() ?? "LIDAR";
  const degraded = lidarData?.status?.degraded ?? false;

  return (
    <div style={styles.container}>
      <div style={styles.label}>LIDAR — PERCEPTION</div>
      <div
        style={{
          ...styles.badge,
          color: degraded ? "#FF8A80" : "#8CEBFF",
          borderColor: degraded ? "#FF8A80" : "#2AC7E3",
          backgroundColor: degraded ? "#FF8A8022" : "#2AC7E322",
        }}
      >
        {degraded ? `${modeLabel} DEGRADED` : modeLabel}
      </div>
      <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H} style={styles.canvas} />
      <div style={styles.rangeLabel}>{RANGE_M}m forward range</div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: "100%",
    height: "100%",
    backgroundColor: "#091019",
    borderLeft: "1px solid #1e2a3b",
    position: "relative",
    overflow: "hidden",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  label: {
    position: "absolute",
    top: 8,
    left: 12,
    color: "#FFB300",
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.2,
    zIndex: 1,
    textShadow: "0 1px 4px rgba(0,0,0,0.8)",
  },
  badge: {
    position: "absolute",
    top: 24,
    left: 12,
    padding: "3px 8px",
    borderRadius: 999,
    border: "1px solid",
    fontSize: 9,
    fontWeight: 700,
    letterSpacing: 0.5,
    zIndex: 1,
  },
  canvas: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
  },
  rangeLabel: {
    position: "absolute",
    bottom: 6,
    right: 10,
    color: "#556272",
    fontSize: 10,
  },
};
