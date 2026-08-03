import { readFileSync, existsSync } from "fs";
import { resolve } from "path";
import { REPO_ROOT } from "./repo.ts";

export function loadEnv(dir?: string): void {
  // Explicit dir, or every plausible .env location: the cwd, the module's dir
  // (dev runs), the module's parent (v2/ when started from repo root), and
  // v2/ via REPO_ROOT (standalone binary run from the repo root).
  const candidates = dir
    ? [dir]
    : [process.cwd(), import.meta.dirname, resolve(import.meta.dirname, ".."), resolve(REPO_ROOT, "v2")];
  for (const d of candidates) {
    const envPath = resolve(d, ".env");
    if (existsSync(envPath)) loadEnvFile(envPath);
  }
}

function loadEnvFile(envPath: string): void {
  const content = readFileSync(envPath, "utf-8");
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx < 0) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    let value = trimmed.slice(eqIdx + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}
