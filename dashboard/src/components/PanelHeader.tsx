import type { CSSProperties, ReactNode } from "react";

type PanelHeaderProps = {
  title: string;
  accentColor: string;
  rightContent?: ReactNode;
};

export default function PanelHeader({ title, accentColor, rightContent }: PanelHeaderProps) {
  return (
    <div style={styles.container}>
      <div style={{ ...styles.title, color: accentColor }}>{title}</div>
      {rightContent ? <div style={styles.rightContent}>{rightContent}</div> : null}
      <div
        style={{
          ...styles.divider,
          background: `linear-gradient(90deg, transparent 0%, ${accentColor} 18%, ${accentColor} 82%, transparent 100%)`,
          boxShadow: `0 0 12px ${accentColor}99`,
        }}
      />
    </div>
  );
}

const styles: Record<string, CSSProperties> = {
  container: {
    position: "relative",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
    width: "100%",
    height: 30,
    padding: "0 10px 0 12px",
    boxSizing: "border-box",
    flexShrink: 0,
    background: "rgba(8,12,20,0.9)",
    backdropFilter: "blur(10px)",
    WebkitBackdropFilter: "blur(10px)",
    overflow: "hidden",
  },
  title: {
    fontSize: 10,
    fontWeight: 800,
    letterSpacing: "0.16em",
    textTransform: "uppercase",
    textShadow: "0 1px 4px rgba(0,0,0,0.6)",
    whiteSpace: "nowrap",
  },
  rightContent: {
    display: "flex",
    alignItems: "center",
    justifyContent: "flex-end",
    minWidth: 0,
    zIndex: 1,
  },
  divider: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    height: 1,
    opacity: 0.95,
  },
};
