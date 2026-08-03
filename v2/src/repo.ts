import { resolve } from "path";

// Repo root of croqtile-tuner (v2/src → up two levels).
// Skill files (DSL knowledge, shell tools) live there during the migration.
export const REPO_ROOT = resolve(import.meta.dirname, "..", "..");
