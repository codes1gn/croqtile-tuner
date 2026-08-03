import type { AgentSession } from "@earendil-works/pi-coding-agent";
import type { TuneTask, TuneResult } from "./task.ts";
import { createSession } from "./session.ts";
import { runMeasure, runCommand } from "./measure.ts";
import { decide, becomesReference } from "./decide.ts";
import { saveIter, restoreIter, iterTag } from "./iters.ts";
import { loadDslKnowledge } from "./dsl.ts";
import { storeRound } from "./store.ts";
import { recordTrajectory } from "./trajectory.ts";
import { errMsg, tailCap } from "./util.ts";
import { DEFAULT_ROUND_TIMEOUT_S } from "./config.ts";

export interface TuneConfig {
  task: TuneTask;
  rounds: number;
  provider?: string;
  modelId?: string;
  apiKey?: string; // passed through to the agent session
  session?: AgentSession;
  store?: boolean; // persist each measured round via store_round.sh
  roundTimeoutMs?: number; // per-round agent timeout; default 600s
  signal?: AbortSignal; // aborts between rounds (SIGINT graceful shutdown)
}

const PROMPT_OUTPUT_CAP = 1500; // chars of the last measurement fed back to the agent

export async function tune(config: TuneConfig): Promise<TuneResult[]> {
  const { task, rounds } = config;
  const roundTimeoutMs = config.roundTimeoutMs ?? DEFAULT_ROUND_TIMEOUT_S * 1000;
  const results: TuneResult[] = [];
  const ownsSession = !config.session;
  let lastTrajectoryFrom = 0;

  const dslKnowledge = task.dsl !== undefined ? loadDslKnowledge(task.dsl) : undefined;
  if (task.dsl !== undefined && dslKnowledge === undefined) {
    console.warn(`Warning: no DSL knowledge found for '${task.dsl}' — continuing with generic prompt`);
  }

  let session: AgentSession | undefined = config.session ?? await createSession({
    cwd: task.cwd,
    provider: config.provider,
    modelId: config.modelId,
    apiKey: config.apiKey,
    systemPrompt: buildSystemPrompt(task, dslKnowledge),
  });

  const pushResult = (sess: AgentSession, result: TuneResult): void => {
    results.push(result);
    recordTrajectory(task, sess, result, lastTrajectoryFrom);
    lastTrajectoryFrom = sess.messages.length;
  };

  try {
    await saveIter(task, 0); // baseline snapshot (iter000)
    // Rebuild before measuring so the baseline reflects the kernel as written,
    // not a stale binary left in the workspace by a previous session.
    const baselineBuild = await runCommand(task.buildCmd, task.cwd, false);
    if (!baselineBuild.ok) {
      console.warn(`Warning: baseline build failed (${baselineBuild.error}) — continuing without comparisons`);
    }
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
      if (session === undefined) break; // disposed by an earlier round timeout
      console.log(`\n=== Round ${round + 1}/${rounds} ===\n`);
      const prompt = round === 0
        ? buildFirstRoundPrompt(task, baseline)
        : buildNextRoundPrompt(task, round, best, bestIter, results.at(-1), lastOutput);

      const timedOut = !(await withTimeout(session.prompt(prompt), roundTimeoutMs));
      if (timedOut) {
        const errorMessage = `agent timed out after ${Math.round(roundTimeoutMs / 1000)}s`;
        pushResult(session, { round, success: false, decision: "unknown", errorMessage });
        if (ownsSession) {
          session.dispose(); // kill the agent so in-flight work stops
          session = undefined;
        }
        // TODO: retry the round once with a fresh session (Iter 5.1 "kill and retry")
        break;
      }

      const agentError = agentErrorOf(session);
      if (agentError) {
        pushResult(session, { round, success: false, decision: "unknown", errorMessage: agentError });
        break;
      }

      // Deterministic compile gate (PRD dual-layer robustness): the orchestrator
      // verifies the round's kernel actually builds — otherwise the benchmark
      // could measure a stale binary. A broken kernel fails the round but the
      // tuning continues from the best-known version.
      const build = await runCommand(task.buildCmd, task.cwd, false);
      if (!build.ok) {
        pushResult(session, { round, success: false, decision: "unknown", errorMessage: `build failed: ${build.error}` });
        try {
          await restoreIter(task, bestIter); // kernel on disk is broken — start next round from best
        } catch { /* iter000 baseline always exists */ }
        continue;
      }

      try {
        await saveIter(task, round + 1); // iteration artifact (iter00N)
      } catch (err) {
        pushResult(session, { round, success: false, decision: "unknown", errorMessage: `failed to save iteration artifact: ${errMsg(err)}` });
        break;
      }

      const measured = await runMeasure(task.profileCmd, task.cwd);
      const tflops = measured.tflops;
      const decision = decide(tflops, best);
      const prev = results.at(-1);
      let errorMessage = measured.error;

      // Settlement: the round's kernel is the new reference iff it became the
      // best (kept AND >= best); otherwise restore the best-known version to
      // disk so the next round always starts from the best kernel.
      if (becomesReference(decision, tflops, best)) {
        best = tflops;
        bestIter = round + 1;
      } else {
        try {
          await restoreIter(task, bestIter);
          if (decision === "reject") {
            errorMessage = `regressed ${fmtPct(tflops, best)} vs best — reverted to iter${iterTag(bestIter)}`;
          } else {
            console.log(`  Kernel unmeasurable — restored iter${iterTag(bestIter)} (best-known)`);
          }
        } catch (err) {
          errorMessage = `restore failed: ${errMsg(err)}`;
        }
      }

      console.log(`  Measured: ${fmtTflops(tflops)}, ${fmtPct(tflops, baseline.tflops)} vs baseline, ${fmtPct(tflops, prev?.tflops)} vs prev → ${decision.toUpperCase()}`);
      if (decision === "reject") console.log(`  Rejected — kernel restored to iter${iterTag(bestIter)}`);
      if (measured.ok) lastOutput = tailCap(measured.output, PROMPT_OUTPUT_CAP);

      if (config.store) {
        const stored = storeRound({
          task, model: config.modelId ?? process.env.CROQTILE_MODEL ?? "auto",
          round, tflops: tflops ?? 0, decision, idea: agentIdea(session),
        });
        if (stored) console.log(`  Stored: ${iterTag(round + 1)} (${decision})`);
      }

      pushResult(session, { round, success: decision !== "reject", tflops, decision, errorMessage });
    }
  } finally {
    if (ownsSession && session) session.dispose();
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
- Make sure the kernel compiles — run the build yourself whenever you need compile feedback.
- The tuner re-builds and benchmarks after your round — you only need to make the kernel fast and correct.
- If unsure what to change, run the profile command for analysis, but don't rely on it as the official measurement.
${dslKnowledge !== undefined ? `
--- DSL CONTRACT (croq-dsl-${task.dsl}) ---
${dslKnowledge}
--- END DSL CONTRACT ---
` : ""}`;
}

function buildFirstRoundPrompt(task: TuneTask, baseline: { tflops?: number }): string {
  return `Start optimizing ${task.kernelPath}.

1. Read the current kernel source
2. Make ONE targeted optimization (tiling, pipeline, memory layout, launch config, ...)
3. Make sure it compiles (run the build yourself for feedback)

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

Best so far: ${best ?? "unknown"} TFLOPS (iter${iterTag(bestIter)}).
${prevLine}
The current kernel is the best-known version — start from it.

1. Make ONE targeted change to beat ${best ?? "the current kernel"}
2. Make sure it compiles (run the build yourself for feedback)

${lastOutput ? `Last benchmark output:\n${lastOutput}\n` : ""}`;
}

function agentErrorOf(session: AgentSession): string | undefined {
  const last = session.messages.at(-1);
  if (last?.role !== "assistant") return undefined;
  return last.stopReason === "error" ? (last.errorMessage ?? "agent stopped with error") : undefined;
}

// Last assistant text — used as the "idea" summary when storing a round.
// Reverse scan from the tail: the round's final message is usually it.
function agentIdea(session: AgentSession): string {
  for (let i = session.messages.length - 1; i >= 0; i--) {
    const message = session.messages[i];
    if (message?.role !== "assistant" || !Array.isArray(message.content)) continue;
    for (let j = message.content.length - 1; j >= 0; j--) {
      const block = message.content[j] as { type?: string; text?: string } | undefined;
      if (block?.type === "text" && block.text) return block.text;
    }
  }
  return "no idea";
}

function fmtTflops(tflops: number | undefined): string {
  return tflops !== undefined ? `${tflops} TFLOPS` : "no measurement";
}

function fmtPct(a: number | undefined, b: number | undefined): string {
  if (a === undefined || b === undefined) return "n/a";
  const delta = ((a - b) / b) * 100;
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%`;
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
