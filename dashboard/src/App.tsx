import { Canvas } from "@react-three/fiber";
import { useWebSocket } from "./hooks/useWebSocket";
import BEVScene from "./components/BEVScene";
import HUD from "./components/HUD";

export default function App() {
  useWebSocket();

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative" }}>
      <Canvas
        gl={{ antialias: true, alpha: false }}
        style={{ background: "#0a0a0f" }}
      >
        <BEVScene />
      </Canvas>
      <HUD />
    </div>
  );
}
