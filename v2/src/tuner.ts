import type { AgentSession } from "@earendil-works/pi-coding-agent";
import type { TuneTask, TuneResult } from "./task.ts";
import { createSession } from "./session.ts";
import { runMeasure } from "./measure.ts";
import { decide } from "./decide.ts";
import { saveIter, restoreIter } from "./iters.ts";
import { loadDslKnowledge } from "./dsl.ts";
import { storeRound } from "./store.ts";

export interface TuneConfig {
  task: TuneTask;
  rounds: number;
  provider?: string;
  modelId?: string;
  session?: AgentSession;
  dsl?: string;
  store?: boolean; // persist each measured round via store_round.sh
}

const PROMPT_OUTPUT_CAP = 1500; // chars of the last measurement fed back to the agent

export async function tune(config: TuneConfig): Promise<TuneResult[]> {
  const { task, rounds } = config;
  const results: TuneResult[] = [];
  const ownsSession = !config.session;

  const dslKnowledge = config.dsl !== undefined ? loadDslKnowledge(config.dsl) : undefined;
  if (config.dsl !== undefined && dslKnowledge === undefined) {
    console.warn(`Warning: no DSL knowledge found for '${config.dsl}' — continuing with generic prompt`);
  }

  const session = config.session ?? (await createSession({
    cwd: task.cwd,
    provider: config.provider,
    modelId: config.modelId,
    systemPrompt: buildSystemPrompt(task, dslKnowledge),
  })).session;

  try {
    await saveIter(task, 0); // baseline snapshot (iter000)
    const baseline = await runMeasure(task.profileCmd, task.cwd);
    if (baseline.tflops !== undefined) {
      console.log(`Baseline: ${baseline.tflops} TFLOPS`);
    } else {
      console.log(`Warning: baseline measurement failed (${baseline.error ?? "unknown error"}) — continuing without comparisons`);
    }

    let best = baseline.tflops;
    let bestIter = 0;
    let lastOutput = "";

    for (let round = 0; round < rounds; round++) {
      console.log(`\n=== Round ${round + 1}/${rounds} ===\n`);
      const prompt = round === 0
        ? buildFirstRoundPrompt(task, baseline)
        : buildNextRoundPrompt(task, round, best, bestIter, results.at(-1), lastOutput);
      await session.prompt(prompt);

      const agentError = agentErrorOf(session);
      if (agentError) {
        results.push({ round, success: false, decision: "unknown", errorMessage: agentError });
        break;
      }

      try {
        await saveIter(task, round + 1); // iteration artifact (iter00N)
      } catch (err) {
        results.push({ round, success: false, decision: "unknown", errorMessage: `failed to save iteration artifact: ${errMsg(err)}` });
        break;
      }

      const measured = await runMeasure(task.profileCmd, task.cwd);
      const tflops = measured.tflops;
      const decision = decide(tflops, best);
      const prev = results.at(-1);

      if (decision === "keep" && tflops !== undefined && (best === undefined || tflops > best)) {
        best = tflops;
        bestIter = round + 1;
      }

      let errorMessage = measured.error;
      if (decision === "reject") {
        try {
          await restoreIter(task, bestIter);
          errorMessage = `regressed ${fmtPct(tflops, best)} vs best — reverted to iter${String(bestIter).padStart(3, "0")}`;
        } catch (err) {
          errorMessage = `regressed ${fmtPct(tflops, best)} vs best, restore failed: ${errMsg(err)}`;
        }
      }

      console.log(`  Measured: ${fmtTflops(tflops)}, ${fmtPct(tflops, baseline.tflops)} vs baseline, ${fmtPct(tflops, prev?.tflops)} vs prev → ${decision.toUpperCase()}`);
      if (decision === "reject") console.log(`  Rejected — kernel restored to iter${String(bestIter).padStart(3, "0")}`);
      if (tflops !== undefined && measured.ok) lastOutput = measured.output.slice(-PROMPT_OUTPUT_CAP);

      if (config.store) {
        const stored = storeRound({
          task, model: config.modelId ?? process.env.CROQTILE_MODEL ?? "auto",
          round, tflops: tflops ?? 0, decision, idea: agentIdea(session),
        });
        if (stored) console.log(`  Stored: ${iterTag(round + 1)} (${decision})`);
      }

      results.push({ round, success: decision !== "reject", tflops, decision, errorMessage });
    }
  } finally {
    if (ownsSession) session.dispose();
  }

  return results;
}

function buildSystemPrompt(task: TuneTask, dslKnowledge?: string): string {
  return `You are a GPU kernel performance engineer.
Your goal: maximize throughput of the kernel at ${task.kernelPath}.

Tools available: read, write, bash.
Working directory: ${task.cwd}

Build: ${task.buildCmd}

Rules:
- Make one focused change per round.
- After editing, always rebuild to confirm the kernel still compiles.
- The tuner benchmarks after your round — you only need to make the kernel fast and correct.
- If unsure what to change, run the profile command for analysis, but don't rely on it as the official measurement.
${dslKnowledge !== undefined ? `
--- DSL CONTRACT (croq-dsl-${task.dsl ?? "?"}) ---
${dslKnowledge}
--- END DSL CONTRACT ---
` : ""}`;
}

function buildFirstRoundPrompt(task: TuneTask, baseline: { tflops?: number }): string {
  return `Start optimizing ${task.kernelPath}.

1. Read the current kernel source
2. Make ONE targeted optimization (tiling, pipeline, memory layout, launch config, ...)
3. Rebuild to verify it compiles: ${task.buildCmd}

Baseline measured by the tuner: ${baseline.tflops ?? "unknown"} TFLOPS.
The tuner benchmarks automatically after your round.`;
}

function buildNextRoundPrompt(
  task: TuneTask,
  round: number,
  best: number | undefined,
  bestIter: number,
  prev: TuneResult | undefined,
  lastOutput: string,
): string {
  const prevLine = prev?.tflops !== undefined
    ? ` Previous round: ${prev.tflops} TFLOPS (${prev.decision}).`
    : "";
  return `Continue optimization (round ${round + 1}).

Best so far: ${best ?? "unknown"} TFLOPS (iter${String(bestIter).padStart(3, "0")}).
${prevLine}
The current kernel is the best-known version — start from it.

1. Make ONE targeted change to beat ${best ?? "the current kernel"}
2. Rebuild to verify it compiles: ${task.buildCmd}

${lastOutput ? `Last benchmark output:\n${lastOutput}\n` : ""}`;
}

function agentErrorOf(session: AgentSession): string | undefined {
  const last = session.messages.at(-1);
  if (last?.role !== "assistant") return undefined;
  const asst = last as { stopReason?: string; errorMessage?: string };
  return asst.stopReason === "error" ? (asst.errorMessage ?? "agent stopped with error") : undefined;
}

// Last assistant text — used as the "idea" summary when storing a round.
function agentIdea(session: AgentSession): string {
  const texts = session.messages
    .filter(m => m.role === "assistant")
    .flatMap(m => m.content)
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map(c => c.text);
  return texts.at(-1) ?? "no idea";
}

function iterTag(n: number): string {
  return `iter${String(n).padStart(3, "0")}`;
}

function fmtTflops(tflops: number | undefined): string {
  return tflops !== undefined ? `${tflops} TFLOPS` : "no measurement";
}

function fmtPct(a: number | undefined, b: number | undefined): string {
  if (a === undefined || b === undefined) return "n/a";
  const delta = ((a - b) / b) * 100;
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%`;
}

function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
