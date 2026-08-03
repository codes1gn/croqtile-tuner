import { loadEnv } from "./env.ts";
import { tune } from "./tuner.ts";
import type { TuneTask, TuneResult } from "./task.ts";

loadEnv();

const args = process.argv.slice(2);

if (args.length === 0 || args.includes("--help")) {
  console.log(`croqtile-tuner v2

Usage:
  croqtile-tuner --kernel <path> --build <cmd> --profile <cmd> [--rounds N] [--cwd <dir>]

Options:
  --kernel    Path to kernel source file
  --build     Build/compile command
  --profile   Benchmark command that prints "TFLOPS: <value>"
  --rounds    Number of optimization rounds (default: 3)
  --cwd       Working directory (default: current)
  --provider  LLM provider (default: from .env)
  --model     Model ID (default: from .env)
  --dsl       DSL name (croqtile, cuda, ...) — injects the DSL contract into the agent prompt
  --gpu       GPU tag for stored results (default: detect_gpu.sh)
  --shape-key Tuning directory shape key (default: kernel file name)
  --store     Persist each round via store_round.sh into tuning/

Per-round kernel snapshots are saved to <cwd>/iters/ (iter000.* = baseline).
The tuner benchmarks after each round; regressions are rejected and the
kernel is reverted to the best-known iteration.`);
  process.exit(0);
}

function getArg(name: string): string | undefined {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 ? args[idx + 1] : undefined;
}

const kernel = getArg("kernel");
const build = getArg("build");
const profile = getArg("profile");

if (!kernel || !build || !profile) {
  console.error("Error: --kernel, --build, and --profile are required");
  process.exit(1);
}

const task: TuneTask = {
  name: kernel.replace(/^.*\//, "").replace(/\.[^.]+$/, ""),
  cwd: getArg("cwd") ?? process.cwd(),
  kernelPath: kernel,
  buildCmd: build,
  profileCmd: profile,
  dsl: getArg("dsl"),
  gpu: getArg("gpu"),
  shapeKey: getArg("shape-key"),
};

const rounds = parseInt(getArg("rounds") ?? "3", 10);
const provider = getArg("provider");
const modelId = getArg("model");
const store = args.includes("--store");

console.log(`Tuning: ${task.name} (${rounds} rounds)`);
console.log(`  kernel:  ${task.kernelPath}`);
console.log(`  build:   ${task.buildCmd}`);
console.log(`  profile: ${task.profileCmd}`);
if (task.dsl) console.log(`  dsl:     ${task.dsl}`);
if (store) console.log(`  store:   results → tuning/`);

let results: TuneResult[];
try {
  results = await tune({ task, rounds, provider, modelId, dsl: task.dsl, store });
} catch (err) {
  console.error(`Error: ${err instanceof Error ? err.message : err}`);
  process.exit(1);
}

console.log("\n=== Summary ===");
for (const r of results) {
  const perf = r.tflops !== undefined ? ` ${r.tflops} TFLOPS` : "";
  const err = r.errorMessage ? ` — ${r.errorMessage}` : "";
  console.log(`  Round ${r.round + 1}: ${r.success ? "✓" : "✗"} ${r.decision}${perf}${err}`);
}

const passed = results.filter(r => r.success).length;
console.log(`\n${passed}/${results.length} rounds completed successfully.`);
process.exit(passed === results.length ? 0 : 1);
