import { describe, expect, it } from "vitest";
import type { IterationLogData } from "../../api";
import { isBaselineEntry } from "../TflopsChart";

function makeLog(overrides: Partial<IterationLogData> = {}): IterationLogData {
  return {
    id: 1,
    task_id: 1,
    iteration: 1,
    request_number: 1,
    kernel_path: "iter001_candidate",
    tflops: 100,
    decision: "KEEP",
    bottleneck: "memory",
    idea_summary: null,
    logged_at: new Date().toISOString(),
    ...overrides,
  };
}

describe("isBaselineEntry fuzz properties", () => {
  it("never classifies iter001_baseline_* custom kernels as baseline", () => {
    for (let i = 0; i < 300; i += 1) {
      const suffix = `${Math.random().toString(36).slice(2)}_${i}`;
      const log = makeLog({
        kernel_path: `iter001_baseline_${suffix}`,
        decision: "KEEP",
        bottleneck: "memory",
      });
      expect(isBaselineEntry(log)).toBe(false);
    }
  });

  it("always treats iter000 + baseline markers as baseline", () => {
    const markers = ["cublas", "torch", "baseline"];
    for (let i = 0; i < 300; i += 1) {
      const marker = markers[i % markers.length];
      const suffix = `${Math.random().toString(36).slice(2)}_${i}`;
      const log = makeLog({
        kernel_path: `iter000_${marker}_${suffix}`,
        decision: "DISCARD",
        bottleneck: "memory",
      });
      expect(isBaselineEntry(log)).toBe(true);
    }
  });

  it("always treats bottleneck baseline markers as baseline", () => {
    for (let i = 0; i < 300; i += 1) {
      const kernelPath = `${Math.random().toString(36).slice(2)}_${i}`;
      const log = makeLog({
        kernel_path: kernelPath,
        bottleneck: "baseline_profile",
        decision: "DISCARD",
      });
      expect(isBaselineEntry(log)).toBe(true);
    }
  });
});
