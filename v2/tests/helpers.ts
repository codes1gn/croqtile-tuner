import { mkdirSync, rmSync } from "fs";
import { resolve } from "path";
import {
  type AgentSession,
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { createFauxCore, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai/providers/faux";
import { REPO_ROOT } from "../src/repo.ts";

export { fauxAssistantMessage, fauxToolCall };

const AGENT_DIR = "/tmp/croqtile-tuner-test";

export interface FauxSessionOptions {
  cwd: string;
  responses: unknown[];
  tools?: string[];
}

export async function createFauxSession(opts: FauxSessionOptions): Promise<AgentSession> {
  mkdirSync(opts.cwd, { recursive: true });

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

  faux.setResponses(opts.responses);

  const sm = SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } });
  const rl = new DefaultResourceLoader({ cwd: opts.cwd, agentDir: AGENT_DIR, settingsManager: sm });
  await rl.reload();

  const { session } = await createAgentSession({
    cwd: opts.cwd,
    agentDir: AGENT_DIR,
    model: modelRegistry.find("faux", "m")!,
    thinkingLevel: "off",
    authStorage,
    modelRegistry,
    resourceLoader: rl,
    tools: opts.tools ?? ["read", "write", "bash"],
    sessionManager: SessionManager.inMemory(opts.cwd),
    settingsManager: sm,
  });

  return session;
}

export function cleanDir(path: string): void {
  rmSync(path, { recursive: true, force: true });
  mkdirSync(path, { recursive: true });
}

// store_round.sh writes activity traces into the repo's tuning/ — activity_trace.sh
// resolves project_root from the script location, not cwd. Tests remove the residue.
// Distinct gpu tags per test file avoid races between parallel test files.
export function cleanStoreTraces(gpu = "sm00_test"): void {
  rmSync(resolve(REPO_ROOT, "tuning", gpu), { recursive: true, force: true });
}
