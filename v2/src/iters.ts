import { copyFileSync, mkdirSync } from "fs";
import { extname, resolve } from "path";
import type { TuneTask } from "./task.ts";

// Per-round kernel snapshots in <cwd>/iters/, matching the iter000... convention.
function iterPath(task: TuneTask, n: number): string {
  const ext = extname(task.kernelPath);
  return resolve(task.cwd, "iters", `iter${String(n).padStart(3, "0")}${ext}`);
}

export function saveIter(task: TuneTask, n: number): void {
  mkdirSync(resolve(task.cwd, "iters"), { recursive: true });
  copyFileSync(resolve(task.cwd, task.kernelPath), iterPath(task, n));
}

export function restoreIter(task: TuneTask, n: number): void {
  copyFileSync(iterPath(task, n), resolve(task.cwd, task.kernelPath));
}
