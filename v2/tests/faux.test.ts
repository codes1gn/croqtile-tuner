import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, existsSync, rmSync } from "fs";
import {
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { createFauxCore, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";

const CWD = "/tmp/croqtile-tuner-test/workspace";
const AGENT_DIR = "/tmp/croqtile-tuner-test";

function setupFaux() {
  const authStorage = AuthStorage.create(`${AGENT_DIR}/auth.json`);
  const modelRegistry = ModelRegistry.inMemory(authStorage);
  const faux = createFauxCore({ provider: "faux", models: [{ id: "faux-model" }] });

  authStorage.setRuntimeApiKey("faux", "faux-key");
  modelRegistry.registerProvider("faux", {
    name: "Faux",
    baseUrl: "http://localhost:0",
    apiKey: "faux-key",
    models: [{
      id: "faux-model",
      name: "Faux",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 100000,
      maxTokens: 8192,
    }],
    api: faux.api,
    streamSimple: faux.streamSimple,
  });

  return { authStorage, modelRegistry, faux };
}

test("faux: write tool creates file", async () => {
  mkdirSync(CWD, { recursive: true });
  rmSync(`${CWD}/hello.cu`, { force: true });

  const { authStorage, modelRegistry, faux } = setupFaux();
  faux.setResponses([
    fauxAssistantMessage([
      fauxToolCall("write", { path: "hello.cu", content: "__global__ void k() {}\n" }),
    ]),
    fauxAssistantMessage("Done."),
  ]);

  const model = modelRegistry.find("faux", "faux-model")!;
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
  const resourceLoader = new DefaultResourceLoader({ cwd: CWD, agentDir: AGENT_DIR, settingsManager });
  await resourceLoader.reload();

  const { session } = await createAgentSession({
    cwd: CWD,
    agentDir: AGENT_DIR,
    model,
    thinkingLevel: "off",
    authStorage,
    modelRegistry,
    resourceLoader,
    tools: ["write"],
    sessionManager: SessionManager.inMemory(CWD),
    settingsManager,
  });

  try {
    await session.prompt("Write hello.cu");
    assert.ok(existsSync(`${CWD}/hello.cu`), "hello.cu should exist");
  } finally {
    session.dispose();
  }
});

test("faux: empty response ends gracefully", async () => {
  mkdirSync(CWD, { recursive: true });
  const { authStorage, modelRegistry, faux } = setupFaux();
  faux.setResponses([fauxAssistantMessage("Hi.")]);

  const model = modelRegistry.find("faux", "faux-model")!;
  const settingsManager = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
  const resourceLoader = new DefaultResourceLoader({ cwd: CWD, agentDir: AGENT_DIR, settingsManager });
  await resourceLoader.reload();

  const { session } = await createAgentSession({
    cwd: CWD,
    agentDir: AGENT_DIR,
    model,
    thinkingLevel: "off",
    authStorage,
    modelRegistry,
    resourceLoader,
    tools: [],
    sessionManager: SessionManager.inMemory(CWD),
    settingsManager,
  });

  try {
    await session.prompt("Say hi");
    const last = session.messages[session.messages.length - 1];
    assert.equal(last.role, "assistant");
    assert.ok(last.content.length > 0, "should have content");
  } finally {
    session.dispose();
  }
});
