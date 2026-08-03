import type { Decision } from "./decide.ts";

export interface TuneTask {
  name: string;
  cwd: string;
  kernelPath: string;
  buildCmd: string;
  profileCmd: string;
  dsl?: string; // DSL name (croqtile, cuda, ...) → injects DSL contract into prompt
  gpu?: string; // e.g. sm86_NVIDIA_GeForce_RTX_3070; auto-detected when storing
  shapeKey?: string; // tuning/ directory shape key; defaults to task name
}

export interface TuneResult {
  round: number;
  success: boolean;
  decision: Decision;
  tflops?: number;
  errorMessage?: string;
}
