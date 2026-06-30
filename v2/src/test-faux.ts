/**
 * Faux provider test — verifies the full agent loop without network.
 * Run: bun run src/test-faux.ts
 */
import { mkdirSync, existsSync, rmSync } from "fs";
import {
  AuthStorage,
  createAgentSession,
  createExtensionRuntime,
  ModelRegistry,
  type ResourceLoader,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { createFauxCore, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";

const authStorage = AuthStorage.create("/tmp/croqtile-tuner/auth.json");
const modelRegistry = ModelRegistry.inMemory(authStorage);

const faux = createFauxCore({ provider: "faux", models: [{ id: "faux-model" }] });
authStorage.setRuntimeApiKey("faux", "faux-key");
modelRegistry.registerProvider("faux", {
  name: "Faux",
  baseUrl: "http://localhost:0",
  apiKey: "faux-key",
  models: [{
    id: "faux-model",
    name: "Faux Test Model",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 100000,
    maxTokens: 8192,
  }],
  api: faux.api,
  streamSimple: faux.streamSimple,
});

faux.setResponses([
  fauxAssistantMessage([
    fauxToolCall("write", {
      path: "hello.cu",
      content: `#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d\\n", threadIdx.x);
}

int main() {
    hello_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
`,
    }),
  ]),
  fauxAssistantMessage("Done."),
]);

const cwd = "/tmp/croqtile-tuner/workspace";
mkdirSync(cwd, { recursive: true });
rmSync(cwd + "/hello.cu", { force: true });

const model = modelRegistry.find("faux", "faux-model")!;

const resourceLoader: ResourceLoader = {
  getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
  getSkills: () => ({ skills: [], diagnostics: [] }),
  getPrompts: () => ({ prompts: [], diagnostics: [] }),
  getThemes: () => ({ themes: [], diagnostics: [] }),
  getAgentsFiles: () => ({ agentsFiles: [] }),
  getSystemPrompt: () => "You are a test assistant. Tools: write.",
  getAppendSystemPrompt: () => [],
  extendResources: () => {},
  reload: async () => {},
};

const { session } = await createAgentSession({
  cwd,
  agentDir: "/tmp/croqtile-tuner",
  model,
  thinkingLevel: "off",
  authStorage,
  modelRegistry,
  resourceLoader,
  tools: ["read", "write", "bash"],
  sessionManager: SessionManager.inMemory(cwd),
  settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }),
});

try {
  await session.prompt("Write hello.cu");
  const ok = existsSync(cwd + "/hello.cu");
  console.log(ok ? "PASS: hello.cu created" : "FAIL: hello.cu missing");
  process.exit(ok ? 0 : 1);
} finally {
  session.dispose();
}
