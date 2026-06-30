import { test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync } from "fs";
import { createFauxSession, cleanDir, fauxAssistantMessage, fauxToolCall } from "./helpers.ts";
import { tune } from "../src/tuner.ts";

const CWD = "/tmp/croqtile-tuner-test/tuner";

const TASK = {
  name: "kernel",
  cwd: CWD,
  kernelPath: "kernel.cu",
  buildCmd: "echo 'build ok'",
  profileCmd: "echo 'time: 1.0ms'",
};

test("single round: profile → edit → rebuild", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("bash", { command: "echo 'time: 1.2ms'" })]),
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* optimized */ }\n" })]),
      fauxAssistantMessage([fauxToolCall("bash", { command: "echo 'build ok'" })]),
      fauxAssistantMessage("Optimized: 1.2ms → 0.8ms."),
    ],
  });

  const results = await tune({ task: TASK, rounds: 1, session });
  session.dispose();

  assert.equal(results.length, 1);
  assert.ok(results[0].success);
  assert.ok(readFileSync(`${CWD}/kernel.cu`, "utf-8").includes("optimized"));
});

test("multi-round: completes all rounds", async () => {
  cleanDir(CWD);

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage("Round 1 profiled and optimized."),
      fauxAssistantMessage("Round 2 profiled and optimized."),
    ],
  });

  const results = await tune({ task: TASK, rounds: 2, session });
  session.dispose();

  assert.equal(results.length, 2);
  assert.ok(results.every(r => r.success));
});
