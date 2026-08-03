import { copyFileSync, mkdirSync } from "fs";
import { extname, resolve } from "path";
import type { TuneTask } from "./task.ts";

// Per-round kernel snapshots in <cwd>/iters/, matching the iter000... convention.
export function iterTag(n: number): string {
  return `iter${String(n).padStart(3, "0")}`;
}

export function itersDir(task: TuneTask): string {
  return resolve(task.cwd, "iters");
}

function iterPath(task: TuneTask, n: number): string {
  return resolve(itersDir(task), `${iterTag(n)}${extname(task.kernelPath)}`);
}

export function saveIter(task: TuneTask, n: number): void {
  mkdirSync(itersDir(task), { recursive: true });
  copyFileSync(resolve(task.cwd, task.kernelPath), iterPath(task, n));
}

export function restoreIter(task: TuneTask, n: number): void {
  copyFileSync(iterPath(task, n), resolve(task.cwd, task.kernelPath));
}
