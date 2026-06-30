import type { AgentSession } from "@earendil-works/pi-coding-agent";
import type { TuneTask, TuneResult } from "./task.ts";
import { createSession } from "./session.ts";

export interface TuneConfig {
  task: TuneTask;
  rounds: number;
  provider?: string;
  modelId?: string;
  session?: AgentSession;
}

export async function tune(config: TuneConfig): Promise<TuneResult[]> {
  const { task, rounds } = config;
  const results: TuneResult[] = [];
  const ownsSession = !config.session;

  const session = config.session ?? (await createSession({
    cwd: task.cwd,
    provider: config.provider,
    modelId: config.modelId,
    systemPrompt: buildSystemPrompt(task),
  })).session;

  try {
    for (let round = 0; round < rounds; round++) {
      console.log(`\n=== Round ${round + 1}/${rounds} ===\n`);
      const prompt = round === 0 ? buildFirstRoundPrompt(task) : buildNextRoundPrompt(task, round);
      await session.prompt(prompt);

      const result = extractResult(session, round);
      results.push(result);
      console.log(`Round ${round + 1}: ${result.success ? "OK" : "FAIL"}`);

      if (!result.success) break;
    }
  } finally {
    if (ownsSession) session.dispose();
  }

  return results;
}

function buildSystemPrompt(task: TuneTask): string {
  return `You are a GPU kernel performance engineer.
Your goal: maximize throughput of the kernel at ${task.kernelPath}.

Tools available: read, write, bash.
Working directory: ${task.cwd}

Build: ${task.buildCmd}
Profile: ${task.profileCmd}

Rules:
- Profile first, then optimize, then verify.
- Make one focused change per round.
- After editing, always rebuild and re-profile to confirm improvement.
- If a change regresses performance, revert it.`;
}

function buildFirstRoundPrompt(task: TuneTask): string {
  return `Start optimizing ${task.kernelPath}.

1. Read the current kernel source
2. Profile the current baseline: ${task.profileCmd}
3. Identify the top bottleneck from the profile output
4. Make ONE targeted optimization to address it
5. Rebuild: ${task.buildCmd}
6. Re-profile and report the before/after comparison`;
}

function buildNextRoundPrompt(task: TuneTask, round: number): string {
  return `Continue optimization (round ${round + 1}).

1. Profile the current state: ${task.profileCmd}
2. Identify the next bottleneck
3. Make ONE targeted change
4. Rebuild and re-profile
5. Report before/after`;
}

function extractResult(session: AgentSession, round: number): TuneResult {
  const messages = session.messages;
  const last = messages[messages.length - 1];

  if (last?.role === "assistant") {
    const asst = last as { stopReason?: string; errorMessage?: string; content: unknown[] };
    if (asst.stopReason === "error") {
      return { round, success: false, profileOutput: "", errorMessage: asst.errorMessage };
    }
  }

  const textContent = messages
    .filter(m => m.role === "assistant")
    .flatMap(m => m.content)
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map(c => c.text)
    .join("\n");

  return { round, success: true, profileOutput: textContent.slice(-2000) };
}
