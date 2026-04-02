export const DEFAULT_BEV_RENDER_MODE = "navigation-first" as const;

export function isNavigationFirstMode(): boolean {
  return DEFAULT_BEV_RENDER_MODE === "navigation-first";
}
