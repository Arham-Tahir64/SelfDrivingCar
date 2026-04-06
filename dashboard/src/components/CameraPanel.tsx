import { useState } from "react";
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
  const headerLabel =
    viewMode === "depth"
      ? "CAMERA \u2014 DEPTH (DepthAnything V2)"
      : "CAMERA \u2014 PERCEPTION OVERLAY";

  return (
    <div style={styles.container}>
      <div style={styles.label}>{headerLabel}</div>

      {/* View toggle */}
      <div style={styles.toggle}>
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
            opacity: depth ? 1 : 0.35,
          }}
          disabled={!depth}
        >
          DEPTH
        </button>
      </div>

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
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    width: "100%",
    height: "100%",
    backgroundColor: "#0d0d14",
    borderLeft: "1px solid #2a2a3a",
    borderBottom: "1px solid #2a2a3a",
    position: "relative",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
  },
  label: {
    position: "absolute",
    top: 8,
    left: 12,
    color: "#4DD0E1",
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.2,
    zIndex: 1,
    textShadow: "0 1px 4px rgba(0,0,0,0.8)",
  },
  toggle: {
    position: "absolute",
    top: 6,
    right: 12,
    display: "flex",
    gap: 4,
    zIndex: 2,
  },
  toggleButton: {
    border: "1px solid rgba(96, 110, 140, 0.3)",
    borderRadius: 6,
    padding: "4px 10px",
    background: "rgba(19, 24, 36, 0.9)",
    color: "rgba(215, 222, 236, 0.7)",
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: "0.08em",
    cursor: "pointer",
    transition: "background 120ms ease, color 120ms ease",
  },
  toggleButtonActive: {
    background:
      "linear-gradient(180deg, rgba(17, 190, 255, 0.24), rgba(9, 111, 185, 0.36))",
    color: "#eaf9ff",
    borderColor: "rgba(55, 214, 255, 0.5)",
    boxShadow: "0 0 12px rgba(17, 190, 255, 0.16)",
  },
  image: {
    width: "100%",
    height: "100%",
    objectFit: "contain",
    display: "block",
  },
  placeholder: {
    flex: 1,
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
