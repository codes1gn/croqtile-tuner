export type Decision = "keep" | "reject" | "unknown";

// Within 0.5% of best is treated as noise (KEEP) rather than a regression.
// TODO: calibrate against real benchmark variance once we have live data.
export const ACCEPT_TOLERANCE = 0.995;

export function decide(tflops: number | undefined, best: number | undefined): Decision {
  if (tflops === undefined) return "unknown";
  if (best === undefined) return "keep";
  return tflops >= best * ACCEPT_TOLERANCE ? "keep" : "reject";
}
