import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync, readFileSync, rmSync } from "fs";
import {
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { createFauxCore, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { tune } from "../src/tuner.ts";

const CWD = "/tmp/croqtile-tuner-test/tuner";
const AGENT_DIR = "/tmp/croqtile-tuner-test";

async function createFauxSession(responses: unknown[]) {
  const authStorage = AuthStorage.create(`${AGENT_DIR}/auth.json`);
  const modelRegistry = ModelRegistry.inMemory(authStorage);
  const faux = createFauxCore({ provider: "faux", models: [{ id: "m" }] });

  authStorage.setRuntimeApiKey("faux", "k");
  modelRegistry.registerProvider("faux", {
    name: "Faux",
    baseUrl: "http://localhost:0",
    apiKey: "k",
    models: [{
      id: "m", name: "M", reasoning: false, input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 100000, maxTokens: 8192,
    }],
    api: faux.api,
    streamSimple: faux.streamSimple,
  });

  faux.setResponses(responses);

  const sm = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
  const rl = new DefaultResourceLoader({ cwd: CWD, agentDir: AGENT_DIR, settingsManager: sm });
  await rl.reload();

  const { session } = await createAgentSession({
    cwd: CWD,
    agentDir: AGENT_DIR,
    model: modelRegistry.find("faux", "m")!,
    thinkingLevel: "off",
    authStorage,
    modelRegistry,
    resourceLoader: rl,
    tools: ["read", "write", "bash"],
    sessionManager: SessionManager.inMemory(CWD),
    settingsManager: sm,
  });

  return session;
}

test("tuner: single round with tool calls", async () => {
  rmSync(CWD, { recursive: true, force: true });
  mkdirSync(CWD, { recursive: true });
  writeFileSync(`${CWD}/kernel.cu`, "__global__ void k() { /* v0 */ }\n");

  const session = await createFauxSession([
    fauxAssistantMessage([
      fauxToolCall("bash", { command: "echo 'time: 1.2ms'" }),
    ]),
    fauxAssistantMessage([
      fauxToolCall("write", { path: "kernel.cu", content: "__global__ void k() { /* optimized */ }\n" }),
    ]),
    fauxAssistantMessage([
      fauxToolCall("bash", { command: "echo 'build ok'" }),
    ]),
    fauxAssistantMessage("Optimized: 1.2ms → 0.8ms (33% improvement)."),
  ]);

  const results = await tune({
    task: {
      name: "kernel",
      cwd: CWD,
      kernelPath: "kernel.cu",
      buildCmd: "echo 'build ok'",
      profileCmd: "echo 'time: 1.0ms'",
    },
    rounds: 1,
    session,
  });

  session.dispose();

  assert.equal(results.length, 1);
  assert.ok(results[0].success);
  const content = readFileSync(`${CWD}/kernel.cu`, "utf-8");
  assert.ok(content.includes("optimized"));
});

test("tuner: error stops loop", async () => {
  rmSync(CWD, { recursive: true, force: true });
  mkdirSync(CWD, { recursive: true });

  const session = await createFauxSession([
    fauxAssistantMessage("Baseline profiled."),
    fauxAssistantMessage("Round 2 done."),
  ]);

  const results = await tune({
    task: {
      name: "kernel",
      cwd: CWD,
      kernelPath: "kernel.cu",
      buildCmd: "echo ok",
      profileCmd: "echo ok",
    },
    rounds: 2,
    session,
  });

  session.dispose();
  assert.equal(results.length, 2);
  assert.ok(results.every(r => r.success));
});
