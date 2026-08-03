import { test } from "node:test";
import assert from "node:assert/strict";
import { loadDslKnowledge } from "../src/dsl.ts";

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
