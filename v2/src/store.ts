import { spawnSync } from "child_process";
import { resolve } from "path";
import { REPO_ROOT } from "./repo.ts";
import type { TuneTask } from "./task.ts";
import type { Decision } from "./decide.ts";
import { iterTag } from "./iters.ts";

const STORE_SCRIPT = resolve(REPO_ROOT, ".claude", "skills", "croq-tune", "tools", "store_round.sh");
const DETECT_GPU = resolve(REPO_ROOT, ".claude", "skills", "croq-tune", "tools", "detect_gpu.sh");

const DECISION_TO_STORE: Record<Decision, string | undefined> = {
  keep: "KEEP",
  reject: "DISCARD",
  unknown: undefined, // no measurement → nothing meaningful to store
};

export interface StoreOptions {
  task: TuneTask;
  model: string;
  round: number; // 0-based round index
  tflops: number;
  decision: Decision;
  idea: string;
}

// Persists a round via the skill system's store_round.sh so results land in
// tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/ per existing conventions.
// Failures are warnings, never round failures.
export function storeRound(opts: StoreOptions): boolean {
  const { task, round } = opts;
  const decision = DECISION_TO_STORE[opts.decision];
  if (decision === undefined || task.dsl === undefined) return false;

  const result = spawnSync("bash", [
    STORE_SCRIPT,
    "--gpu", task.gpu ?? detectGpu(),
    "--dsl", task.dsl,
    "--shape-key", task.shapeKey ?? task.name,
    "--model", opts.model,
    "--iter", iterTag(round + 1),
    "--kernel", `${iterTag(round + 1)}_auto`,
    "--tflops", String(opts.tflops),
    "--decision", decision,
    "--bottleneck", "unknown",
    "--idea", opts.idea.slice(0, 120),
    "--round", String(round + 1),
    "--category", "general",
  ], { cwd: task.cwd, encoding: "utf-8" });

  if (result.status !== 0) {
    const errLine = (result.stderr ?? result.stdout ?? "")
      .split("\n").find(l => l.includes("ERROR"));
    console.warn(`  [store] store_round.sh failed: ${errLine ?? `exit ${result.status}`}`);
    return false;
  }
  return true;
}

let cachedGpu: string | undefined; // the GPU never changes mid-session

function detectGpu(): string {
  if (cachedGpu !== undefined) return cachedGpu;
  const out = spawnSync("bash", [DETECT_GPU], { encoding: "utf-8" });
  cachedGpu = out.status === 0 && out.stdout.trim() ? out.stdout.trim() : "sm00_unknown";
  return cachedGpu;
}
