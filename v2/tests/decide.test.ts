import { test } from "node:test";
import assert from "node:assert/strict";
import { decide } from "../src/decide.ts";

test("keep when improved vs best", () => {
  assert.equal(decide(12.4, 12.0), "keep");
});

test("keep when within noise tolerance of best", () => {
  assert.equal(decide(11.99, 12.0), "keep");
});

test("reject when regressed beyond tolerance", () => {
  assert.equal(decide(10.5, 12.0), "reject");
});

test("unknown when no measurement", () => {
  assert.equal(decide(undefined, 12.0), "unknown");
});

test("keep when no best to compare against", () => {
  assert.equal(decide(12.0, undefined), "keep");
});
