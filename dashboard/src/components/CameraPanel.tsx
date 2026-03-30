import { useFrameStore } from "../store/frameStore";

export default function CameraPanel() {
  const overlay = useFrameStore((s) => s.currentFrame?.["visualization/camera_overlay"]);

  return (
    <div style={styles.container}>
      <div style={styles.label}>CAMERA — PERCEPTION OVERLAY</div>
      {overlay ? (
        <img
          src={`data:image/jpeg;base64,${overlay}`}
          alt="Camera overlay"
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
