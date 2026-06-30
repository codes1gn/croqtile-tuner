import { loadEnv } from "./env.ts";
import { tune } from "./tuner.ts";
import type { TuneTask } from "./task.ts";

loadEnv();

const args = process.argv.slice(2);

if (args.length === 0 || args.includes("--help")) {
  console.log(`croqtile-tuner v2

Usage:
  croqtile-tuner --kernel <path> --build <cmd> --profile <cmd> [--rounds N] [--cwd <dir>]

Options:
  --kernel    Path to kernel source file
  --build     Build/compile command
  --profile   Profile/benchmark command
  --rounds    Number of optimization rounds (default: 3)
  --cwd       Working directory (default: current)
  --provider  LLM provider (default: from .env)
  --model     Model ID (default: from .env)`);
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
};

const rounds = parseInt(getArg("rounds") ?? "3", 10);
const provider = getArg("provider");
const modelId = getArg("model");

console.log(`Tuning: ${task.name} (${rounds} rounds)`);
console.log(`  kernel:  ${task.kernelPath}`);
console.log(`  build:   ${task.buildCmd}`);
console.log(`  profile: ${task.profileCmd}\n`);

const results = await tune({ task, rounds, provider, modelId });

console.log("\n=== Summary ===");
for (const r of results) {
  console.log(`  Round ${r.round + 1}: ${r.success ? "✓" : "✗"} ${r.errorMessage ?? ""}`);
}

const passed = results.filter(r => r.success).length;
console.log(`\n${passed}/${results.length} rounds completed successfully.`);
process.exit(passed === results.length ? 0 : 1);
