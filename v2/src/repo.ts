import { existsSync } from "fs";
import { resolve } from "path";

// Repo root of croqtile-tuner. Skill files (DSL knowledge, shell tools) live
// there during the migration. In a standalone binary the module path doesn't
// contain .claude — fall back to the working directory (run from repo root).
function findRepoRoot(): string {
  const fromModule = resolve(import.meta.dirname, "..", "..");
  return existsSync(resolve(fromModule, ".claude")) ? fromModule : process.cwd();
}

export const REPO_ROOT = findRepoRoot();
