import {
  type AgentSession,
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { providerEnvName } from "./env.ts";

export interface SessionConfig {
  cwd: string;
  provider?: string;
  modelId?: string;
  systemPrompt?: string;
  agentDir?: string;
  apiKey?: string; // overrides the <PROVIDER>_API_KEY env resolution
}

export async function createSession(config: SessionConfig): Promise<AgentSession> {
  const {
    cwd,
    provider = process.env.CROQTILE_PROVIDER ?? "anthropic",
    modelId = process.env.CROQTILE_MODEL ?? "claude-sonnet-4-20250514",
    systemPrompt = DEFAULT_SYSTEM_PROMPT,
    agentDir = process.env.CROQTILE_AGENT_DIR ?? "/tmp/croqtile-tuner",
  } = config;

  const authStorage = AuthStorage.create(`${agentDir}/auth.json`);
  const modelRegistry = ModelRegistry.inMemory(authStorage);

  if (provider === "ollama") {
    registerOpenAiCompatible(modelRegistry, "ollama", modelId, {
      baseUrl: process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434/v1",
      apiKey: "ollama", // local server, no key
      contextWindow: 32768,
      maxTokens: 8192, // full-file writes need headroom beyond the default 4096
      supportsDeveloperRole: false,
    });
    authStorage.setRuntimeApiKey("ollama", "ollama");
  } else {
    const apiKey = config.apiKey ?? resolveApiKey(provider);
    if (!apiKey) {
      throw new Error(`API key required. Set <PROVIDER>_API_KEY (or model.api_key in config)`);
    }
    authStorage.setRuntimeApiKey(provider, apiKey);

    // <PROVIDER>_BASE_URL override → OpenAI-compatible endpoint (local gateway, proxy, self-hosted)
    const baseUrl = process.env[providerEnvName(provider, "BASE_URL")];
    if (baseUrl) {
      registerOpenAiCompatible(modelRegistry, provider, modelId, {
        baseUrl,
        apiKey,
        contextWindow: 200_000,
        maxTokens: 16_384,
        supportsDeveloperRole: true,
      });
    }
  }

  const model = modelRegistry.find(provider, modelId);
  if (!model) {
    const available = modelRegistry.getAll()
      .filter(m => m.provider === provider)
      .map(m => m.id)
      .slice(0, 5);
    throw new Error(`Model ${provider}/${modelId} not found. Available: ${available.join(", ")}`);
  }

  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: false },
    retry: { enabled: true, maxRetries: 2 },
  });

  const resourceLoader = new DefaultResourceLoader({
    cwd,
    agentDir,
    settingsManager,
    systemPromptOverride: () => systemPrompt,
  });
  await resourceLoader.reload();

  const { session } = await createAgentSession({
    cwd,
    agentDir,
    model,
    thinkingLevel: "off",
    authStorage,
    modelRegistry,
    resourceLoader,
    tools: ["read", "write", "bash"],
    sessionManager: SessionManager.inMemory(cwd),
    settingsManager,
  });

  return session;
}

// One registration path for OpenAI-compatible endpoints (ollama, <PROVIDER>_BASE_URL
// overrides). Per-provider differences live in the options, not in the control flow.
interface CompatOptions {
  baseUrl: string;
  apiKey: string;
  contextWindow: number;
  maxTokens: number;
  supportsDeveloperRole: boolean;
}

function registerOpenAiCompatible(registry: ModelRegistry, provider: string, modelId: string, opts: CompatOptions): void {
  registry.registerProvider(provider, {
    baseUrl: opts.baseUrl,
    apiKey: opts.apiKey,
    api: "openai-completions",
    models: [{
      id: modelId,
      name: modelId,
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: opts.contextWindow,
      maxTokens: opts.maxTokens,
      compat: { supportsDeveloperRole: opts.supportsDeveloperRole, maxTokensField: "max_tokens" },
    }],
  });
}

function resolveApiKey(provider: string): string | undefined {
  return process.env[providerEnvName(provider, "API_KEY")];
}

const DEFAULT_SYSTEM_PROMPT = `You are a GPU kernel engineer assistant.
Available tools: read, write, bash.
Write files when asked. Be concise. Do not explain unless asked.`;
