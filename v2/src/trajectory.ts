import { appendFileSync, mkdirSync } from "fs";
import { resolve } from "path";
import type { AgentSession } from "@earendil-works/pi-coding-agent";
import type { TuneTask, TuneResult } from "./task.ts";

const CONTENT_CAP = 5000; // chars kept per content block — tool outputs are the noisy ones

// Pure: one trajectory record from the session state + round outcome.
// Every tool call and response is captured so "why did the agent do X"
// is answerable later (PRD: Trajectory Recorder → RL/grey-testing data).
export function buildTrajectoryRecord(session: AgentSession, result: TuneResult): unknown {
  return {
    ts: new Date().toISOString(),
    round: result.round,
    success: result.success,
    decision: result.decision,
    tflops: result.tflops,
    errorMessage: result.errorMessage,
    messages: session.messages.map(serializeMessage),
  };
}

// IO: append one JSONL line to <cwd>/iters/trajectory.jsonl (next to kernel snapshots).
export function recordTrajectory(task: TuneTask, session: AgentSession, result: TuneResult): void {
  const file = resolve(task.cwd, "iters", "trajectory.jsonl");
  mkdirSync(resolve(task.cwd, "iters"), { recursive: true });
  appendFileSync(file, JSON.stringify(buildTrajectoryRecord(session, result)) + "\n");
}

type TracedMessage = {
  role: string;
  content?: unknown;
  summary?: unknown;
  toolCallId?: unknown;
  toolName?: unknown;
  isError?: unknown;
};

function serializeMessage(m: TracedMessage): unknown {
  return {
    role: m.role,
    ...(m.content !== undefined ? { content: serializeContent(m.content) } : {}),
    ...(m.summary !== undefined ? { summary: cap(m.summary) } : {}),
    // toolResult data lives on the message, not in a content block
    ...(m.role === "toolResult" ? { tool_call_id: m.toolCallId, tool_name: m.toolName, is_error: m.isError } : {}),
  };
}

function serializeContent(content: unknown): unknown {
  if (Array.isArray(content)) return content.map(serializeBlock);
  if (typeof content === "string") return content.slice(-CONTENT_CAP);
  return content;
}

function serializeBlock(block: unknown): unknown {
  if (typeof block !== "object" || block === null) return block;
  const o = block as Record<string, unknown>;
  switch (o.type) {
    case "text":
    case "thinking":
      return { type: o.type, text: cap(o.text) };
    case "toolCall":
      return { type: "tool_call", id: o.id, name: o.name, arguments: o.arguments };
    default:
      return { type: o.type, content: cap(o.content ?? o.text) };
  }
}

function cap(v: unknown): unknown {
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s === undefined || s.length <= CONTENT_CAP ? v : s.slice(-CONTENT_CAP);
}
