import { test } from "node:test";
import assert from "node:assert/strict";
import { stripFrontmatter, loadDslKnowledge } from "../src/dsl.ts";
import { REPO_ROOT } from "../src/repo.ts";

test("stripFrontmatter: removes YAML header", () => {
  const text = "---\nname: x\ndescription: y\n---\n# Body\ncontent";
  assert.equal(stripFrontmatter(text), "# Body\ncontent");
});

test("stripFrontmatter: no frontmatter → unchanged", () => {
  const text = "# Body\ncontent";
  assert.equal(stripFrontmatter(text), text);
});

test("stripFrontmatter: unterminated frontmatter → unchanged", () => {
  const text = "---\nname: x";
  assert.equal(stripFrontmatter(text), text);
});

test("loadDslKnowledge: loads the real croqtile contract body", () => {
  const knowledge = loadDslKnowledge("croqtile");
  assert.ok(knowledge);
  assert.ok(knowledge.includes("# Croq-DSL: CroqTile (Choreo)"));
  assert.ok(knowledge.includes("## IDEA Menu"));
  assert.ok(!knowledge.includes("argument-hint")); // frontmatter stripped
});

test("loadDslKnowledge: unknown DSL → undefined", () => {
  assert.equal(loadDslKnowledge("nonexistent-dsl"), undefined);
});
