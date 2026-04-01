import { useRef, useEffect } from "react";
import { useFrameStore } from "../store/frameStore";

const CANVAS_W = 400;
const CANVAS_H = 300;
const RANGE_M = 50;
const POINT_RADIUS = 1.2;

function heightToColor(z: number): string {
  // blue (ground) -> cyan -> green -> yellow -> red (high)
  const t = Math.max(0, Math.min(1, (z + 1) / 5));
  if (t < 0.25) return `rgb(${Math.round(t * 4 * 60)}, ${Math.round(t * 4 * 120 + 80)}, 255)`;
  if (t < 0.5) return `rgb(0, ${Math.round(180 + (t - 0.25) * 4 * 75)}, ${Math.round(255 - (t - 0.25) * 4 * 200)})`;
  if (t < 0.75) return `rgb(${Math.round((t - 0.5) * 4 * 255)}, 255, ${Math.round(55 - (t - 0.5) * 4 * 55)})`;
  return `rgb(255, ${Math.round(255 - (t - 0.75) * 4 * 200)}, 0)`;
}

export default function LidarPanel() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lidarData = useFrameStore((s) => s.currentFrame?.["visualization/lidar_preview"]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#0a0a14";
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    // Draw grid
    ctx.strokeStyle = "#1a1a2a";
    ctx.lineWidth = 0.5;
    const gridStep = 10; // meters
    for (let r = gridStep; r <= RANGE_M; r += gridStep) {
      const rx = (r / RANGE_M) * (CANVAS_W / 2);
      const ry = (r / RANGE_M) * (CANVAS_H / 2);
      ctx.beginPath();
      ctx.ellipse(CANVAS_W / 2, CANVAS_H / 2, rx, ry, 0, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw crosshairs
    ctx.strokeStyle = "#1f1f30";
    ctx.beginPath();
    ctx.moveTo(CANVAS_W / 2, 0);
    ctx.lineTo(CANVAS_W / 2, CANVAS_H);
    ctx.moveTo(0, CANVAS_H / 2);
    ctx.lineTo(CANVAS_W, CANVAS_H / 2);
    ctx.stroke();

    if (!lidarData?.points?.length) {
      ctx.fillStyle = "#444";
      ctx.font = "12px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for LiDAR...", CANVAS_W / 2, CANVAS_H / 2);
      return;
    }

    for (const pt of lidarData.points) {
      // CARLA ray-cast LiDAR points arrive in ego/sensor-local coordinates:
      // x = forward, y = lateral, z = height.
      const localX = pt[0] ?? 0;
      const localY = pt[1] ?? 0;
      const z = pt[2] ?? 0;

      // Skip points outside range
      if (Math.abs(localX) > RANGE_M || Math.abs(localY) > RANGE_M) continue;

      // Map to canvas (forward = up)
      const cx = CANVAS_W / 2 + (localY / RANGE_M) * (CANVAS_W / 2);
      const cy = CANVAS_H / 2 - (localX / RANGE_M) * (CANVAS_H / 2);

      ctx.fillStyle = heightToColor(z);
      ctx.beginPath();
      ctx.arc(cx, cy, POINT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw ego marker
    ctx.fillStyle = "#00E5FF";
    ctx.beginPath();
    const ex = CANVAS_W / 2;
    const ey = CANVAS_H / 2;
    ctx.moveTo(ex, ey - 6);
    ctx.lineTo(ex - 4, ey + 4);
    ctx.lineTo(ex + 4, ey + 4);
    ctx.closePath();
    ctx.fill();
  }, [lidarData]);

  return (
    <div style={styles.container}>
      <div style={styles.label}>LIDAR — BIRD'S EYE</div>
      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        style={styles.canvas}
      />
      <div style={styles.rangeLabel}>{RANGE_M}m range</div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: "100%",
    height: "100%",
    backgroundColor: "#0a0a14",
    borderLeft: "1px solid #2a2a3a",
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
  canvas: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
  },
  rangeLabel: {
    position: "absolute",
    bottom: 6,
    right: 10,
    color: "#555",
    fontSize: 10,
  },
};
