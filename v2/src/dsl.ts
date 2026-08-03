import { readFileSync } from "fs";
import { resolve } from "path";
import { stripFrontmatter } from "@earendil-works/pi-coding-agent";
import { REPO_ROOT } from "./repo.ts";

// Loads the DSL contract from .claude/skills/croq-dsl-<dsl>/SKILL.md
// (knowledge transfer, not rewrite — same content, new carrier).
// Returns the body with YAML frontmatter stripped, or undefined if unknown.
export function loadDslKnowledge(dsl: string): string | undefined {
  const path = resolve(REPO_ROOT, ".claude", "skills", `croq-dsl-${dsl}`, "SKILL.md");
  try {
    return stripFrontmatter(readFileSync(path, "utf-8"));
  } catch {
    return undefined;
  }
}
