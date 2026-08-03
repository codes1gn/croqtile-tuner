import { after, test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync } from "fs";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import { createFauxSession, cleanDir, cleanStoreTraces, fauxAssistantMessage, fauxToolCall } from "./helpers.ts";
import { tune } from "../src/tuner.ts";
import { buildTrajectoryRecord } from "../src/trajectory.ts";

after(() => cleanStoreTraces());

const CWD = "/tmp/croqtile-tuner-test/trajectory";

const TASK = {
  name: "kernel",
  cwd: CWD,
  kernelPath: "kernel.cu",
  buildCmd: "echo 'build ok'",
  profileCmd: "echo 'TFLOPS: 1.0'",
};

test("buildTrajectoryRecord: serializes tool calls and tool results", () => {
  const session = {
    messages: [
      { role: "user", content: [{ type: "text", text: "optimize" }] },
      {
        role: "assistant",
        content: [
          { type: "text", text: "let me try tiling" },
          { type: "toolCall", id: "c1", name: "write", arguments: { path: "kernel.cu" } },
        ],
      },
      {
        role: "toolResult",
        toolCallId: "c1",
        toolName: "write",
        isError: false,
        content: [{ type: "text", text: "written" }],
      },
    ],
  } as unknown as AgentSession;

  const rec = buildTrajectoryRecord(session, { round: 0, success: true, decision: "keep", tflops: 1.5 }) as {
    round: number;
    tflops: number;
    messages: { role: string; content: unknown[] }[];
  };

  assert.equal(rec.round, 0);
  assert.equal(rec.tflops, 1.5);
  assert.equal(rec.messages.length, 3);

  const assistant = rec.messages[1];
  assert.ok(assistant);
  const blocks = assistant.content as { type: string }[];
  assert.ok(blocks.some(b => b.type === "text"));
  assert.ok(blocks.some(b => b.type === "tool_call"));

  const toolResult = rec.messages[2] as { role: string; tool_call_id: string; tool_name: string; content: { type: string; text: string }[] };
  assert.equal(toolResult.role, "toolResult");
  assert.equal(toolResult.tool_call_id, "c1");
  assert.equal(toolResult.tool_name, "write");
  assert.equal(toolResult.content[0]?.text, "written");
});

test("buildTrajectoryRecord: caps long tool outputs to the tail", () => {
  const long = "x".repeat(6000);
  const session = {
    messages: [
      {
        role: "toolResult",
        toolCallId: "c1",
        toolName: "bash",
        isError: false,
        content: [{ type: "text", text: long }],
      },
    ],
  } as unknown as AgentSession;

  const rec = buildTrajectoryRecord(session, { round: 0, success: true, decision: "keep" }) as {
    messages: { content: { type: string; text: string }[] }[];
  };
  const out = rec.messages[0]?.content[0]?.text ?? "";
  assert.equal(out.length, 5000);
  assert.equal(out, long.slice(-5000));
});

test("recordTrajectory: one JSONL line per round during tune()", async () => {
  cleanDir(CWD);
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  const session = await createFauxSession({
    cwd: CWD,
    responses: [
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v1 */ }\n" })]),
      fauxAssistantMessage("Tiled the inner loop."),
      fauxAssistantMessage([fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* v2 */ }\n" })]),
      fauxAssistantMessage("Swizzled the tile order."),
    ],
  });

  const results = await tune({ task: TASK, rounds: 2, session });
  session.dispose();

  assert.equal(results.length, 2);
  const lines = readFileSync(`${CWD}/iters/trajectory.jsonl`, "utf-8").trim().split("\n");
  assert.equal(lines.length, 2);

  const round1 = JSON.parse(lines[0]!) as { round: number; decision: string; messages: { role: string; content: unknown[] }[] };
  assert.equal(round1.round, 0);
  assert.equal(round1.decision, "keep");

  const blocks = round1.messages.flatMap(m => m.content);
  assert.ok(blocks.some(b => (b as { type?: string }).type === "tool_call"));
  assert.ok(round1.messages.some(m => m.role === "toolResult"));

  const round2 = JSON.parse(lines[1]!) as { round: number; messages: unknown[] };
  assert.equal(round2.round, 1);
  // delta recording: round 2's record excludes round 1's messages
  assert.ok(!JSON.stringify(round2.messages).includes("Tiled the inner loop."));
});
