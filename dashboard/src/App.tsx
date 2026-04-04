import { useEffect, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { useWebSocket } from "./hooks/useWebSocket";
import BEVScene from "./components/BEVScene";
import MapScene from "./components/MapScene";
import HUD from "./components/HUD";
import CameraPanel from "./components/CameraPanel";
import LidarPanel from "./components/LidarPanel";
import PlaybackControls from "./components/PlaybackControls";
import { useFrameStore } from "./store/frameStore";

export default function App() {
  useWebSocket();
  const [viewMode, setViewMode] = useState<"bev" | "map">("bev");

  // Keyboard shortcut: Space to toggle pause
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        useFrameStore.getState().togglePause();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div style={styles.root}>
      {/* Left: BEV 3D scene (55%) */}
      <div style={styles.bevPanel}>
        <Canvas
          gl={{ antialias: true, alpha: false }}
          style={{ background: "#0a0a0f" }}
        >
          {viewMode === "bev" ? <BEVScene /> : <MapScene />}
        </Canvas>
        <div style={styles.viewToggle}>
          <button
            type="button"
            onClick={() => setViewMode("bev")}
            style={{
              ...styles.viewToggleButton,
              ...(viewMode === "bev" ? styles.viewToggleButtonActive : {}),
            }}
          >
            BEV
          </button>
          <button
            type="button"
            onClick={() => setViewMode("map")}
            style={{
              ...styles.viewToggleButton,
              ...(viewMode === "map" ? styles.viewToggleButtonActive : {}),
            }}
          >
            MAP
          </button>
        </div>
        <HUD />
        <PlaybackControls />
      </div>

      {/* Right: Camera (top 60%) + LiDAR (bottom 40%) */}
      <div style={styles.rightPanel}>
        <div style={styles.cameraPanel}>
          <CameraPanel />
        </div>
        <div style={styles.lidarPanel}>
          <LidarPanel />
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    width: "100vw",
    height: "100vh",
    display: "flex",
    flexDirection: "row",
    overflow: "hidden",
    backgroundColor: "#0a0a0f",
  },
  bevPanel: {
    flex: "0 0 55%",
    position: "relative",
    height: "100%",
  },
  viewToggle: {
    position: "absolute",
    top: 18,
    left: "50%",
    transform: "translateX(-50%)",
    display: "flex",
    gap: 8,
    padding: 6,
    borderRadius: 14,
    background: "rgba(10, 14, 24, 0.86)",
    border: "1px solid rgba(96, 110, 140, 0.28)",
    backdropFilter: "blur(10px)",
    boxShadow: "0 10px 30px rgba(0, 0, 0, 0.28)",
    zIndex: 6,
  },
  viewToggleButton: {
    border: "none",
    borderRadius: 10,
    padding: "8px 14px",
    background: "rgba(19, 24, 36, 0.94)",
    color: "rgba(215, 222, 236, 0.82)",
    fontSize: 12,
    fontWeight: 700,
    letterSpacing: "0.08em",
    cursor: "pointer",
    transition: "background 160ms ease, color 160ms ease, box-shadow 160ms ease",
  },
  viewToggleButtonActive: {
    background: "linear-gradient(180deg, rgba(17, 190, 255, 0.24), rgba(9, 111, 185, 0.36))",
    color: "#eaf9ff",
    boxShadow: "inset 0 0 0 1px rgba(55, 214, 255, 0.5), 0 0 18px rgba(17, 190, 255, 0.16)",
  },
  rightPanel: {
    flex: "0 0 45%",
    display: "flex",
    flexDirection: "column",
    height: "100%",
  },
  cameraPanel: {
    flex: "0 0 60%",
    overflow: "hidden",
  },
  lidarPanel: {
    flex: "0 0 40%",
    overflow: "hidden",
  },
};
