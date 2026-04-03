import { useFrameStore } from "../store/frameStore";

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4];

export default function PlaybackControls() {
  const paused = useFrameStore((s) => s.paused);
  const speed = useFrameStore((s) => s.playbackSpeed);
  const togglePause = useFrameStore((s) => s.togglePause);
  const setSpeed = useFrameStore((s) => s.setPlaybackSpeed);

  return (
    <div style={styles.bar}>
      <button
        style={{ ...styles.btn, ...(paused ? styles.btnActive : {}) }}
        onClick={togglePause}
        title={paused ? "Resume (Space)" : "Pause (Space)"}
      >
        {paused ? "\u25B6" : "\u23F8"}
      </button>

      <div style={styles.divider} />

      {SPEED_OPTIONS.map((s) => (
        <button
          key={s}
          style={{
            ...styles.speedBtn,
            ...(Math.abs(speed - s) < 0.01 ? styles.speedBtnActive : {}),
          }}
          onClick={() => setSpeed(s)}
        >
          {s}x
        </button>
      ))}

      {paused && <span style={styles.pauseLabel}>PAUSED</span>}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  bar: {
    position: "absolute",
    bottom: 12,
    left: "50%",
    transform: "translateX(-50%)",
    display: "flex",
    alignItems: "center",
    gap: 6,
    background: "rgba(10, 10, 18, 0.88)",
    border: "1px solid #2a2a3a",
    borderRadius: 8,
    padding: "6px 14px",
    zIndex: 20,
    backdropFilter: "blur(6px)",
  },
  btn: {
    background: "none",
    border: "1px solid #3a3a4a",
    borderRadius: 6,
    color: "#ccc",
    fontSize: 16,
    width: 36,
    height: 30,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "background 0.15s",
  },
  btnActive: {
    background: "rgba(77, 208, 225, 0.18)",
    borderColor: "#4DD0E1",
    color: "#4DD0E1",
  },
  divider: {
    width: 1,
    height: 20,
    background: "#2a2a3a",
    margin: "0 4px",
  },
  speedBtn: {
    background: "none",
    border: "1px solid transparent",
    borderRadius: 4,
    color: "#888",
    fontSize: 11,
    fontWeight: 600,
    padding: "3px 8px",
    cursor: "pointer",
    letterSpacing: 0.3,
    transition: "all 0.15s",
  },
  speedBtnActive: {
    color: "#4DD0E1",
    borderColor: "#4DD0E1",
    background: "rgba(77, 208, 225, 0.12)",
  },
  pauseLabel: {
    color: "#FF6B6B",
    fontSize: 10,
    fontWeight: 700,
    letterSpacing: 1.5,
    marginLeft: 8,
  },
};
