import { Canvas } from "@react-three/fiber";
import { useWebSocket } from "./hooks/useWebSocket";
import BEVScene from "./components/BEVScene";
import HUD from "./components/HUD";
import CameraPanel from "./components/CameraPanel";
import LidarPanel from "./components/LidarPanel";

export default function App() {
  useWebSocket();

  return (
    <div style={styles.root}>
      {/* Left: BEV 3D scene (55%) */}
      <div style={styles.bevPanel}>
        <Canvas
          gl={{ antialias: true, alpha: false }}
          style={{ background: "#0a0a0f" }}
        >
          <BEVScene />
        </Canvas>
        <HUD />
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
