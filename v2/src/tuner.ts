import type { AgentSession } from "@earendil-works/pi-coding-agent";
import type { TuneTask, TuneResult } from "./task.ts";
import { createSession } from "./session.ts";
import { runMeasure, runCommand } from "./measure.ts";
import { decide } from "./decide.ts";
import { saveIter, restoreIter } from "./iters.ts";
import { loadDslKnowledge } from "./dsl.ts";
import { storeRound } from "./store.ts";
import { recordTrajectory } from "./trajectory.ts";

export interface TuneConfig {
  task: TuneTask;
  rounds: number;
  provider?: string;
  modelId?: string;
  session?: AgentSession;
  dsl?: string;
  store?: boolean; // persist each measured round via store_round.sh
  roundTimeoutMs?: number; // per-round agent timeout; default 600s
  signal?: AbortSignal; // aborts between rounds (SIGINT graceful shutdown)
}

const PROMPT_OUTPUT_CAP = 1500; // chars of the last measurement fed back to the agent
const DEFAULT_ROUND_TIMEOUT_MS = 600_000; // TODO: make configurable via config file (Iter 6)

export async function tune(config: TuneConfig): Promise<TuneResult[]> {
  const { task, rounds } = config;
  const roundTimeoutMs = config.roundTimeoutMs ?? DEFAULT_ROUND_TIMEOUT_MS;
  const results: TuneResult[] = [];
  const ownsSession = !config.session;
  let sessionAlive = true;

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
      if (config.signal?.aborted) {
        console.log("Tuning interrupted — state saved to <cwd>/iters/.");
        break;
      }
      console.log(`\n=== Round ${round + 1}/${rounds} ===\n`);
      const prompt = round === 0
        ? buildFirstRoundPrompt(task, baseline)
        : buildNextRoundPrompt(task, round, best, bestIter, results.at(-1), lastOutput);

      const timedOut = !(await withTimeout(session.prompt(prompt), roundTimeoutMs));
      if (timedOut) {
        const errorMessage = `agent timed out after ${Math.round(roundTimeoutMs / 1000)}s`;
        pushResult(results, task, session, { round, success: false, decision: "unknown", errorMessage });
        if (ownsSession) {
          session.dispose(); // kill the agent so in-flight work stops
          sessionAlive = false;
        }
        // TODO: retry the round once with a fresh session (Iter 5.1 "kill and retry")
        break;
      }

      const agentError = agentErrorOf(session);
      if (agentError) {
        pushResult(results, task, session, { round, success: false, decision: "unknown", errorMessage: agentError });
        break;
      }

      // Deterministic compile gate (PRD dual-layer robustness): the orchestrator
      // verifies the round's kernel actually builds — otherwise the benchmark
      // could measure a stale binary. A broken kernel fails the round but the
      // tuning continues from the best-known version.
      const build = await runCommand(task.buildCmd, task.cwd, false);
      if (!build.ok) {
        pushResult(results, task, session, { round, success: false, decision: "unknown", errorMessage: `build failed: ${build.error}` });
        try {
          await restoreIter(task, bestIter); // kernel on disk is broken — start next round from best
        } catch { /* iter000 baseline always exists */ }
        continue;
      }

      try {
        await saveIter(task, round + 1); // iteration artifact (iter00N)
      } catch (err) {
        pushResult(results, task, session, { round, success: false, decision: "unknown", errorMessage: `failed to save iteration artifact: ${errMsg(err)}` });
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
      if (decision === "reject" || tflops === undefined) {
        // Rejected or unmeasurable → the kernel on disk is not the best-known;
        // restore so the next round starts from it (the prompt says so).
        try {
          await restoreIter(task, bestIter);
          if (decision === "reject") {
            errorMessage = `regressed ${fmtPct(tflops, best)} vs best — reverted to iter${String(bestIter).padStart(3, "0")}`;
          } else {
            console.log(`  Kernel unmeasurable — restored iter${String(bestIter).padStart(3, "0")} (best-known)`);
          }
        } catch (err) {
          errorMessage = `restore failed: ${errMsg(err)}`;
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

      pushResult(results, task, session, { round, success: decision !== "reject", tflops, decision, errorMessage });
    }
  } finally {
    if (ownsSession && sessionAlive) session.dispose();
  }

  return results;
}

// Resolves true when the promise settles (success or error), false on timeout.
// The underlying promise is abandoned on timeout — dispose() stops in-flight work.
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<boolean> {
  return new Promise(resolve => {
    const timer = setTimeout(() => resolve(false), ms);
    promise.then(
      () => { clearTimeout(timer); resolve(true); },
      () => { clearTimeout(timer); resolve(true); },
    );
  });
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

function pushResult(results: TuneResult[], task: TuneTask, session: AgentSession, result: TuneResult): void {
  results.push(result);
  recordTrajectory(task, session, result);
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
