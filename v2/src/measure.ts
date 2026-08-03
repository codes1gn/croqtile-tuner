import { spawn } from "node:child_process";

export interface MeasureResult {
  ok: boolean;
  tflops?: number;
  output: string;
  error?: string;
}

const MEASURE_TIMEOUT_MS = 300_000; // TODO(Iter 5): configurable timeout + process-group kill
const OUTPUT_CAP = 100_000; // chars kept from combined stdout+stderr

export function runMeasure(cmd: string, cwd: string): Promise<MeasureResult> {
  return new Promise(resolve => {
    const child = spawn(cmd, { shell: true, cwd });
    let out = "";
    let capped = false;

    const append = (chunk: Buffer | string) => {
      if (capped) return;
      out += chunk.toString();
      if (out.length > OUTPUT_CAP * 4) capped = true;
    };
    const tail = () => out.slice(-OUTPUT_CAP);

    const timer = setTimeout(() => {
      child.kill();
      resolve({ ok: false, output: tail(), error: `timed out after ${MEASURE_TIMEOUT_MS / 1000}s` });
    }, MEASURE_TIMEOUT_MS);

    child.stdout.on("data", append);
    child.stderr.on("data", append);
    child.on("error", err => {
      clearTimeout(timer);
      resolve({ ok: false, output: tail(), error: err.message });
    });
    child.on("close", code => {
      clearTimeout(timer);
      const output = tail();
      const tflops = parseTflops(output);
      if (code !== 0) resolve({ ok: false, output, error: `exit code ${code}` });
      else if (tflops === undefined) resolve({ ok: false, output, error: "no TFLOPS found in output" });
      else resolve({ ok: true, tflops, output });
    });
  });
}

// Accepts both "TFLOPS: 36.12" (DSL contract) and "277.65 TFLOPS (67.4% cuBLAS)".
// First occurrence wins — benchmark binaries print once.
export function parseTflops(text: string): number | undefined {
  const patterns = [
    /TFLOPS:\s*([0-9]*\.?[0-9]+)/i,
    /([0-9]*\.?[0-9]+)\s*TFLOPS/i,
  ];
  for (const re of patterns) {
    const match = text.match(re);
    const value = match?.[1];
    if (value !== undefined) return parseFloat(value);
  }
  return undefined;
}
