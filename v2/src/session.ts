import {
  type AgentSession,
  AuthStorage,
  createAgentSession,
  DefaultResourceLoader,
  ModelRegistry,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";

export interface SessionConfig {
  cwd: string;
  provider?: string;
  modelId?: string;
  systemPrompt?: string;
  agentDir?: string;
}

export interface SessionResult {
  session: AgentSession;
  model: { provider: string; id: string };
}

export async function createSession(config: SessionConfig): Promise<SessionResult> {
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
    const baseUrl = process.env.OLLAMA_BASE_URL ?? "http://127.0.0.1:11434/v1";
    modelRegistry.registerProvider("ollama", {
      baseUrl,
      apiKey: "ollama",
      api: "openai-completions",
      models: [{
        id: modelId,
        name: modelId,
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 32768,
        maxTokens: 4096,
        compat: { supportsDeveloperRole: false, maxTokensField: "max_tokens" },
      }],
    });
    authStorage.setRuntimeApiKey("ollama", "ollama");
  } else {
    const apiKey = resolveApiKey(provider);
    if (!apiKey) {
      throw new Error(`API key required. Set one of: ANTHROPIC_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY, or OPENAI_API_KEY`);
    }
    authStorage.setRuntimeApiKey(provider, apiKey);
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

  return { session, model: { provider: model.provider, id: model.id } };
}

function resolveApiKey(provider: string): string | undefined {
  const envMap: Record<string, string> = {
    anthropic: "ANTHROPIC_API_KEY",
    groq: "GROQ_API_KEY",
    google: "GOOGLE_API_KEY",
    openai: "OPENAI_API_KEY",
    openrouter: "OPENROUTER_API_KEY",
    deepseek: "DEEPSEEK_API_KEY",
    together: "TOGETHER_API_KEY",
  };
  const envVar = envMap[provider] ?? `${provider.toUpperCase().replace(/-/g, "_")}_API_KEY`;
  return process.env[envVar];
}

const DEFAULT_SYSTEM_PROMPT = `You are a GPU kernel engineer assistant.
Available tools: read, write, bash.
Write files when asked. Be concise. Do not explain unless asked.`;
