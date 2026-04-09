import type { CSSProperties } from "react";
import { useState } from "react";
import PanelHeader from "./PanelHeader";
import { useFrameStore } from "../store/frameStore";

type ViewMode = "perception" | "depth";

export default function CameraPanel() {
  const overlay = useFrameStore(
    (s) => s.currentFrame?.["visualization/camera_overlay"]
  );
  const depth = useFrameStore((s) => s.currentFrame?.["perception/depth"]);
  const [viewMode, setViewMode] = useState<ViewMode>("perception");

  const activeImage = viewMode === "depth" && depth ? depth : overlay;
  const altLabel = viewMode === "depth" ? "Depth map" : "Camera overlay";

  return (
    <div style={styles.container}>
      <PanelHeader
        title="Camera"
        accentColor="#4DD0E1"
        rightContent={
          <div style={styles.toggleShell}>
            <button
              type="button"
              onClick={() => setViewMode("perception")}
              style={{
                ...styles.toggleButton,
                ...(viewMode === "perception" ? styles.toggleButtonActive : {}),
              }}
            >
              SEG
            </button>
            <button
              type="button"
              onClick={() => setViewMode("depth")}
              style={{
                ...styles.toggleButton,
                ...(viewMode === "depth" ? styles.toggleButtonActive : {}),
                ...(!depth ? styles.toggleButtonDisabled : {}),
              }}
              disabled={!depth}
            >
              DEPTH
            </button>
          </div>
        }
      />

      <div style={styles.content}>
        {activeImage ? (
          <img
            src={`data:image/jpeg;base64,${activeImage}`}
            alt={altLabel}
            style={styles.image}
          />
        ) : (
          <div style={styles.placeholder}>
            <div style={styles.placeholderIcon}>&#x1F3A5;</div>
            <div style={styles.placeholderText}>Waiting for camera feed...</div>
          </div>
        )}
      </div>
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    width: "100%",
    height: "100%",
    backgroundColor: "#0d0d14",
    borderLeft: "1px solid #2a2a3a",
    borderBottom: "1px solid #2a2a3a",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
  },
  content: {
    position: "relative",
    flex: 1,
    minHeight: 0,
    display: "flex",
  },
  toggleShell: {
    display: "flex",
    gap: 4,
    padding: 2,
    borderRadius: 999,
    background: "rgba(13, 18, 30, 0.85)",
    border: "1px solid rgba(96, 110, 140, 0.24)",
    boxShadow: "inset 0 0 0 1px rgba(255,255,255,0.03)",
  },
  toggleButton: {
    border: "1px solid transparent",
    borderRadius: 999,
    padding: "3px 9px",
    minWidth: 48,
    background: "transparent",
    color: "rgba(215, 222, 236, 0.7)",
    fontSize: 9,
    fontWeight: 700,
    letterSpacing: "0.12em",
    lineHeight: 1.1,
    textTransform: "uppercase",
    cursor: "pointer",
    transition: "background 120ms ease, color 120ms ease, border-color 120ms ease, opacity 120ms ease",
  },
  toggleButtonActive: {
    background:
      "linear-gradient(180deg, rgba(17, 190, 255, 0.24), rgba(9, 111, 185, 0.36))",
    color: "#eaf9ff",
    borderColor: "rgba(55, 214, 255, 0.5)",
    boxShadow: "0 0 12px rgba(17, 190, 255, 0.16)",
  },
  toggleButtonDisabled: {
    opacity: 0.35,
  },
  image: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    display: "block",
  },
  placeholder: {
    width: "100%",
    height: "100%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    color: "#444",
  },
  placeholderIcon: {
    fontSize: 32,
    marginBottom: 8,
    opacity: 0.4,
  },
  placeholderText: {
    fontSize: 12,
    letterSpacing: 0.5,
  },
};
