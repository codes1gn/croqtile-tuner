import { after, test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync, existsSync } from "fs";
import { resolve } from "path";
import { createFauxSession, cleanDir, cleanStoreTraces, fauxAssistantMessage, fauxToolCall } from "./helpers.ts";
import { tune } from "../src/tuner.ts";

after(cleanStoreTraces); // store_round.sh leaves activity traces in repo tuning/

const CWD = "/tmp/croqtile-tuner-test/tuner";
const SCORE = "/tmp/croqtile-tuner-test/score";

const TASK = {
  name: "kernel",
  cwd: CWD,
  kernelPath: "kernel.cu",
  buildCmd: "echo 'build ok'",
  profileCmd: "echo 'TFLOPS: 1.0'",
};

test("single round: edit kernel, measure, keep", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("bash", { command: "echo 'profile for analysis'" })]),
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* optimized */ }\n" })]),
      fauxAssistantMessage([fauxToolCall("bash", { command: "echo 'build ok'" })]),
      fauxAssistantMessage("Optimized: 1.2ms → 0.8ms."),
    ],
  });

  const results = await tune({ task: TASK, rounds: 1, session });
  session.dispose();

  assert.equal(results.length, 1);
  assert.equal(results[0].decision, "keep");
  assert.equal(results[0].tflops, 1.0);
  assert.ok(results[0].success);
  assert.ok(readFileSync(`${CWD}/kernel.cu`, "utf-8").includes("optimized"));
});

test("iteration artifacts: iter000 = baseline, iter001 = round 1 kernel", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v1 */ }\n" })]),
      fauxAssistantMessage("done"),
    ],
  });

  await tune({ task: TASK, rounds: 1, session });
  session.dispose();

  assert.ok(existsSync(`${CWD}/iters/iter000.cu`));
  assert.ok(existsSync(`${CWD}/iters/iter001.cu`));
  assert.match(readFileSync(`${CWD}/iters/iter000.cu`, "utf-8"), /v0/);
  assert.match(readFileSync(`${CWD}/iters/iter001.cu`, "utf-8"), /v1/);
});

test("multi-round: completes all rounds", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage("Round 1 optimized."),
      fauxAssistantMessage("Round 2 optimized."),
    ],
  });

  const results = await tune({ task: TASK, rounds: 2, session });
  session.dispose();

  assert.equal(results.length, 2);
  assert.ok(results.every(r => r.success));
  assert.ok(results.every(r => r.decision === "keep"));
});

test("regression round is rejected and kernel reverted to best", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");
  writeFileSync(SCORE, "1.0"); // baseline score

  // profileCmd reads a counter the faux agent's bash calls control
  const counterTask = {
    ...TASK,
    profileCmd: `echo "TFLOPS: $(cat ${SCORE})"`,
  };

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      // Round 1: agent claims 2.0 TFLOPS kernel, writes it
      fauxAssistantMessage([fauxToolCall("bash", { command: `echo 2.0 > ${SCORE}` })]),
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v2 */ }\n" })]),
      fauxAssistantMessage([fauxToolCall("bash", { command: "echo 'build ok'" })]),
      fauxAssistantMessage("Round 1 optimized."),
      // Round 2: agent's kernel is slower (1.0)
      fauxAssistantMessage([fauxToolCall("bash", { command: `echo 1.0 > ${SCORE}` })]),
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v1 */ }\n" })]),
      fauxAssistantMessage([fauxToolCall("bash", { command: "echo 'build ok'" })]),
      fauxAssistantMessage("Round 2 optimized."),
    ],
  });

  const results = await tune({ task: counterTask, rounds: 2, session });
  session.dispose();

  assert.equal(results.length, 2);
  assert.equal(results[0].decision, "keep");
  assert.equal(results[0].tflops, 2.0);
  assert.equal(results[1].decision, "reject");
  assert.equal(results[1].tflops, 1.0);
  assert.ok(!results[1].success);
  assert.match(results[1].errorMessage ?? "", /-50\.0% vs best/);
  // kernel reverted to the best-known version (round 1's v2)
  assert.match(readFileSync(`${CWD}/kernel.cu`, "utf-8"), /v2/);
  assert.ok(!readFileSync(`${CWD}/kernel.cu`, "utf-8").includes("v1"));
});

test("store: round results persist via store_round.sh into tuning/", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  const storeTask = {
    ...TASK,
    dsl: "croqtile",
    gpu: "sm00_test",
  };

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v1 */ }\n" })]),
      fauxAssistantMessage("Increased tile size."),
    ],
  });

  const results = await tune({ task: storeTask, rounds: 1, session, dsl: "croqtile", store: true });
  session.dispose();

  assert.equal(results[0].decision, "keep");
  const tsv = readFileSync(resolve(CWD, "tuning", "sm00_test", "croqtile", "logs", "kernel", "auto", "results.tsv"), "utf-8");
  assert.match(tsv, /^iter001\titer001_auto\t1\tKEEP\tunknown\tIncreased tile size\.$/m);
});
