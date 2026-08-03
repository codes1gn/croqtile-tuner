import type { Decision } from "./decide.ts";

export interface TuneTask {
  name: string;
  cwd: string;
  kernelPath: string;
  buildCmd: string;
  profileCmd: string;
}

export interface TuneResult {
  round: number;
  success: boolean;
  decision: Decision;
  tflops?: number;
  errorMessage?: string;
}
