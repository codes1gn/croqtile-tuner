import { after, test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "fs";
import { resolve } from "path";
import { cleanDir, cleanStoreTraces } from "./helpers.ts";
import { storeRound } from "../src/store.ts";

after(() => cleanStoreTraces("sm00_store_test")); // store_round.sh leaves activity traces in repo tuning/

const CWD = "/tmp/croqtile-tuner-test/store";

const TASK = {
  name: "matmul_f16_2048",
  cwd: CWD,
  kernelPath: "iter000_swizzle.co",
  buildCmd: "true",
  profileCmd: "true",
  dsl: "croqtile",
  gpu: "sm00_store_test",
};

function tsvPath(model: string): string {
  return resolve(CWD, "tuning", "sm00_store_test", "croqtile", "logs", "matmul_f16_2048", model, "results.tsv");
}

test("storeRound: writes results.tsv + idea-log.jsonl for KEEP", () => {
  cleanDir(CWD);
  const stored = storeRound({ task: TASK, model: "test-model", round: 0, tflops: 0.122, decision: "keep", idea: "swizzle tile scheduling" });
  assert.ok(stored);

  const tsv = readFileSync(tsvPath("test-model"), "utf-8");
  assert.match(tsv, /^iter001\titer001_auto\t0\.122\tKEEP\tunknown\tswizzle tile scheduling$/m);

  const log = readFileSync(resolve(CWD, "tuning", "sm00_store_test", "croqtile", "logs", "matmul_f16_2048", "test-model", "idea-log.jsonl"), "utf-8");
  assert.match(log, /"round": 1/);
  assert.match(log, /"tflops": 0\.122/);
  assert.match(log, /"decision": "KEEP"/);
});

test("storeRound: DISCARD for rejected rounds", () => {
  cleanDir(CWD);
  const stored = storeRound({ task: TASK, model: "m", round: 1, tflops: 0.1, decision: "reject", idea: "worse" });
  assert.ok(stored);
  const tsv = readFileSync(tsvPath("m"), "utf-8");
  assert.match(tsv, /^iter002\titer002_auto\t0\.1\tDISCARD\tunknown\tworse$/m);
});

test("storeRound: skips when decision unknown (no measurement)", () => {
  cleanDir(CWD);
  assert.equal(storeRound({ task: TASK, model: "m", round: 0, tflops: 0, decision: "unknown", idea: "" }), false);
});

test("storeRound: skips when task has no dsl", () => {
  cleanDir(CWD);
  assert.equal(storeRound({ task: { ...TASK, dsl: undefined }, model: "m", round: 0, tflops: 1, decision: "keep", idea: "" }), false);
});
