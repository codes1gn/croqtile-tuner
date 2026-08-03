import { after, test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync, existsSync } from "fs";
import { resolve } from "path";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { createFauxSession, cleanDir, cleanStoreTraces, fauxAssistantMessage, fauxToolCall } from "./helpers.ts";
import { tune } from "../src/tuner.ts";

after(() => cleanStoreTraces("sm00_tuner_test")); // store_round.sh leaves activity traces in repo tuning/

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
    gpu: "sm00_tuner_test",
  };

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v1 */ }\n" })]),
      fauxAssistantMessage("Increased tile size."),
    ],
  });

  const results = await tune({ task: storeTask, rounds: 1, session, store: true });
  session.dispose();

  assert.equal(results[0].decision, "keep");
  const tsv = readFileSync(resolve(CWD, "tuning", "sm00_tuner_test", "croqtile", "logs", "kernel", "auto", "results.tsv"), "utf-8");
  assert.match(tsv, /^iter001\titer001_auto\t1\tKEEP\tunknown\tIncreased tile size\.$/m);
});

test("round timeout: hanging agent is killed and round fails", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  // a session whose prompt never settles
  const hanging = {
    messages: [],
    prompt: () => new Promise<void>(() => {}),
  } as unknown as AgentSession;

  const results = await tune({ task: TASK, rounds: 2, session: hanging, roundTimeoutMs: 100 });

  assert.equal(results.length, 1);
  assert.equal(results[0].success, false);
  assert.equal(results[0].decision, "unknown");
  assert.match(results[0].errorMessage ?? "", /timed out after/);
});

test("build gate: failed compile fails the round but tuning continues from best", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  // fails only while the kernel contains the "broken" marker
  const brokenTask = { ...TASK, buildCmd: "grep -q broken kernel.cu && exit 2; echo 'build ok'" };

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* broken */ }\n" })]),
      fauxAssistantMessage("Round 1 done."),
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v2 */ }\n" })]),
      fauxAssistantMessage("Round 2 done."),
    ],
  });

  const results = await tune({ task: brokenTask, rounds: 2, session });
  session.dispose();

  assert.equal(results.length, 2); // the loop survived the bad round
  assert.equal(results[0].success, false);
  assert.equal(results[0].decision, "unknown");
  assert.match(results[0].errorMessage ?? "", /build failed: exit code 2/);
  assert.equal(results[1].success, true);
  assert.equal(results[1].decision, "keep");
  // round 2's kernel is on disk
  assert.match(readFileSync(`${CWD}/kernel.cu`, "utf-8"), /v2/);
});

test("unmeasurable round restores best kernel and continues", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  const unmeasurableTask = { ...TASK, profileCmd: "echo 'no numbers here'" };

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v1 */ }\n" })]),
      fauxAssistantMessage("Round 1 done."),
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v2 */ }\n" })]),
      fauxAssistantMessage("Round 2 done."),
    ],
  });

  const results = await tune({ task: unmeasurableTask, rounds: 2, session });
  session.dispose();

  assert.equal(results.length, 2);
  assert.ok(results.every(r => r.decision === "unknown")); // no measurement, not a failure
  assert.ok(results.every(r => r.success));
  // kernel on disk is the best-known version (baseline v0), not the last agent write
  assert.match(readFileSync(`${CWD}/kernel.cu`, "utf-8"), /v0/);
});

test("keep within tolerance restores the best kernel for the next round", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");
  writeFileSync(SCORE, "2.0"); // baseline

  const counterTask = {
    ...TASK,
    profileCmd: `echo "TFLOPS: $(cat ${SCORE})"`,
  };

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("bash", { command: `echo 1.99 > ${SCORE}` })]),
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v1 */ }\n" })]),
      fauxAssistantMessage([fauxToolCall("bash", { command: "echo 'build ok'" })]),
      fauxAssistantMessage("Round 1 done."),
    ],
  });

  const results = await tune({ task: counterTask, rounds: 1, session });
  session.dispose();

  assert.equal(results[0].decision, "keep"); // 1.99 is within 0.5% of 2.0
  assert.equal(results[0].tflops, 1.99);
  assert.ok(results[0].success);
  // kept-but-not-best: disk holds the best-known kernel, not the round's
  assert.match(readFileSync(`${CWD}/kernel.cu`, "utf-8"), /v0/);
});
