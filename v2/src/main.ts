import { loadEnv } from "./env.ts";
import { tune } from "./tuner.ts";
import type { TuneTask, TuneResult } from "./task.ts";
import { loadConfig, type Config, DEFAULT_ROUNDS, DEFAULT_STORE, DEFAULT_ROUND_TIMEOUT_S } from "./config.ts";
import { errMsg } from "./util.ts";

loadEnv();

const VERSION = "0.2.0"; // TODO: source from package.json at build time

const args = process.argv.slice(2);

if (args.length === 0 || args.includes("--help")) {
  console.log(`croqtile-tuner v${VERSION}

Usage:
  croqtile-tuner --config config.yaml
  croqtile-tuner --kernel <path> --build <cmd> --profile <cmd> [options]

Options:
  --config    Path to YAML config file (see config.example.yaml)
  --kernel    Path to kernel source file
  --build     Build/compile command
  --profile   Benchmark command that prints "TFLOPS: <value>"
  --rounds    Number of optimization rounds (default: 3)
  --cwd       Working directory (default: current)
  --provider  LLM provider (default: from config/.env)
  --model     Model ID (default: from config/.env)
  --dsl       DSL name (croqtile, cuda, ...) — injects the DSL contract into the agent prompt
  --gpu       GPU tag for stored results (default: detect_gpu.sh)
  --shape-key Tuning directory shape key (default: kernel file name)
  --store     Persist each round via store_round.sh into tuning/
  --version   Print version

Per-round kernel snapshots are saved to <cwd>/iters/ (iter000.* = baseline).
The tuner benchmarks after each round; regressions are rejected and the
kernel is reverted to the best-known iteration.
Agent trajectory (every tool call + response) → <cwd>/iters/trajectory.jsonl
Ctrl+C stops after the current round; state is saved in <cwd>/iters/.`);
  process.exit(0);
}

if (args.includes("--version")) {
  console.log(`croqtile-tuner v${VERSION}`);
  process.exit(0);
}

function getArg(name: string): string | undefined {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 ? args[idx + 1] : undefined;
}

const configFile = getArg("config");
let cfg: Config | undefined;
if (configFile) {
  const loaded = loadConfig(configFile);
  if (!loaded.ok) {
    console.error(`Error: ${loaded.error}`);
    process.exit(1);
  }
  cfg = loaded.value;
}

const kernel = getArg("kernel") ?? cfg?.task.kernel;
const build = getArg("build") ?? cfg?.task.build;
const profile = getArg("profile") ?? cfg?.task.profile;

if (!kernel || !build || !profile) {
  console.error("Error: --kernel, --build and --profile are required (or --config with a task section)");
  process.exit(1);
}

const task: TuneTask = {
  name: cfg?.task.name ?? kernel.replace(/^.*\//, "").replace(/\.[^.]+$/, ""),
  cwd: getArg("cwd") ?? cfg?.task.cwd ?? process.cwd(),
  kernelPath: kernel,
  buildCmd: build,
  profileCmd: profile,
  dsl: getArg("dsl") ?? cfg?.task.dsl,
  gpu: getArg("gpu") ?? cfg?.task.gpu,
  shapeKey: getArg("shape-key") ?? cfg?.task.shape_key,
};

const rounds = parseInt(getArg("rounds") ?? String(cfg?.orchestrator.rounds ?? DEFAULT_ROUNDS), 10);
const provider = getArg("provider") ?? cfg?.model.provider;
const modelId = getArg("model") ?? cfg?.model.model;
const store = args.includes("--store") || (cfg?.orchestrator.store ?? DEFAULT_STORE);
const roundTimeoutMs = Math.round((cfg?.orchestrator.round_timeout_s ?? DEFAULT_ROUND_TIMEOUT_S) * 1000);

console.log(`Tuning: ${task.name} (${rounds} rounds)`);
console.log(`  kernel:  ${task.kernelPath}`);
console.log(`  build:   ${task.buildCmd}`);
console.log(`  profile: ${task.profileCmd}`);
if (task.dsl) console.log(`  dsl:     ${task.dsl}`);
if (store) console.log(`  store:   results → tuning/`);

// Ctrl+C → stop after the current round (per-round state is already saved).
const interrupt = new AbortController();
process.on("SIGINT", () => {
  console.log("\nInterrupt received — stopping after the current round.");
  interrupt.abort();
});

let results: TuneResult[];
try {
  results = await tune({ task, rounds, provider, modelId, apiKey: cfg?.model.api_key, store, roundTimeoutMs, signal: interrupt.signal });
} catch (err) {
  console.error(`Error: ${errMsg(err)}`);
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
