import { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  api,
  type TaskData,
  type IterationLogData,
  type AgentLogData,
  type SessionHistoryData,
} from "../api";
import { parseDtype, dtypeLabel } from "../dtype";
import { StatusBadge } from "./StatusBadge";
import { TflopsChart, isBaselineEntry } from "./TflopsChart";
import { LiveLog } from "./LiveLog";
import { SessionHistory } from "./SessionHistory";
import type { SSEEvent } from "../hooks/useSSE";

interface Props {
  sseEvent: SSEEvent | null;
}

function providerGroup(models: string[]): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const m of models) {
    const slash = m.indexOf("/");
    const provider = slash > 0 ? m.slice(0, slash) : "other";
    (groups[provider] ??= []).push(m);
  }
  return groups;
}

export function TaskDetail({ sseEvent }: Props) {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [task, setTask] = useState<TaskData | null>(null);
  const [iterLogs, setIterLogs] = useState<IterationLogData[]>([]);
  const [agentLogs, setAgentLogs] = useState<AgentLogData[]>([]);
  const [sessionHistory, setSessionHistory] = useState<SessionHistoryData | null>(null);
  const [error, setError] = useState("");
  const [showResume, setShowResume] = useState(false);
  const [resumeIter, setResumeIter] = useState("");
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [availableVariants, setAvailableVariants] = useState<string[]>([]);
  const [editModel, setEditModel] = useState("");
  const [editVariant, setEditVariant] = useState("");
  const [savingModel, setSavingModel] = useState(false);

  const taskId = id ? parseInt(id) : NaN;

  const loadSessionHistory = useCallback(async () => {
    if (isNaN(taskId)) return;
    try {
      setSessionHistory(await api.getSessionHistory(taskId, 400));
    } catch {
      // allow the detail view to keep rendering even if session history is unavailable
    }
  }, [taskId]);

  const loadData = useCallback(async () => {
    if (isNaN(taskId)) return;
    try {
      const [t, il, al, sh, settings] = await Promise.all([
        api.getTask(taskId),
        api.getIterationLogs(taskId),
        api.getAgentLogs(taskId, 200),
        api.getSessionHistory(taskId, 400),
        api.getModelSettings(),
      ]);
      setTask(t);
      setIterLogs(il);
      setAgentLogs(al.reverse());
      setSessionHistory(sh);
      setAvailableModels(settings.available_models ?? []);
      setAvailableVariants(settings.available_variants ?? [""]);
      setEditModel(t.model ?? settings.available_models?.[0] ?? "");
      setEditVariant(t.variant ?? settings.available_variants?.[0] ?? "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load task");
    }
  }, [taskId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (!task) return;
    // Poll more frequently for running tasks, less often for stopped/pending tasks
    const interval = task.status === "running" ? 5000 : 30000;
    const timer = window.setInterval(() => {
      void loadData();
    }, interval);
    return () => window.clearInterval(timer);
  }, [loadData, task?.status]);

  useEffect(() => {
    if (!sseEvent || isNaN(taskId)) return;
    const d = sseEvent.data as Record<string, unknown>;
    if ((d.task_id ?? d.id) !== taskId) return;

    if (sseEvent.type === "task_update") {
      setTask(d as unknown as TaskData);
      void loadSessionHistory();
    } else if (sseEvent.type === "iteration") {
      setIterLogs((prev) => {
        const exists = prev.some((l) => l.iteration === (d.iteration as number));
        if (exists) return prev;
        return [
          ...prev,
          {
            id: Date.now(),
            task_id: taskId,
            iteration: d.iteration as number,
            tflops: d.tflops as number,
            decision: d.decision as string,
            kernel_path: null,
            bottleneck: null,
            idea_summary: null,
            request_number: (d.request_number as number | null) ?? null,
            logged_at: new Date().toISOString(),
          },
        ];
      });
    } else if (sseEvent.type === "agent_log") {
      setAgentLogs((prev) => [
        ...prev.slice(-499),
        {
          id: Date.now(),
          task_id: taskId,
          level: d.level as string,
          message: d.message as string,
          timestamp: new Date().toISOString(),
        },
      ]);
      void loadSessionHistory();
    }
  }, [loadSessionHistory, sseEvent, taskId]);

  const handleCancel = async () => {
    if (!task) return;
    try {
      await api.cancelTask(task.id);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel");
    }
  };

  const handlePromote = async () => {
    if (!task) return;
    try {
      await api.promoteTask(task.id);
      loadData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to promote");
    }
  };

  const handleDelete = async () => {
    if (!task) return;
    if (!window.confirm(`Delete task "${task.shape_key}"? This cannot be undone.`)) return;
    try {
      await api.deleteTask(task.id);
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete task");
    }
  };

  const handleResume = async () => {
    if (!task) return;
    const iter = parseInt(resumeIter);
    if (isNaN(iter) || iter < 0) {
      setError("Invalid iteration number");
      return;
    }
    try {
      const resumed = await api.resumeTask(task.id, iter);
      navigate(`/tasks/${resumed.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to resume");
    }
  };

  const handleSaveModel = async () => {
    if (!task) return;
    if (!editModel.trim()) {
      setError("Model is required");
      return;
    }
    setSavingModel(true);
    setError("");
    try {
      const updated = await api.updateTask(task.id, { model: editModel, variant: editVariant });
      setTask(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update task model");
    } finally {
      setSavingModel(false);
    }
  };

  if (error) {
    return <div className="p-6 text-red-400">{error}</div>;
  }
  if (!task) {
    return <div className="p-6 text-gray-500">Loading...</div>;
  }

  const pct = task.max_iterations > 0
    ? Math.round((task.current_iteration / task.max_iterations) * 100)
    : 0;
  const elapsed =
    task.started_at
      ? formatDuration(
          new Date(task.completed_at ?? Date.now()).getTime() -
            new Date(task.started_at).getTime(),
        )
      : "--";

  const variantSuffix = task.variant ? ` (${task.variant})` : "";
  const platformLabel = task.mode === "cursor_cli" ? "Cursor CLI" : task.mode === "opencode" ? "OpenCode" : task.mode;
  const taskModelLabel = task.model ? modelLabel(task.model) + variantSuffix : "system default";
  const modelHasBuiltinVariant = editModel.startsWith("cursor/");
  const effectiveVariants = modelHasBuiltinVariant ? [""] : availableVariants;
  const modelEditable = task.status !== "running" && task.status !== "completed";
  const modelChanged = editModel !== (task.model ?? "") || editVariant !== (task.variant ?? "");
  const platformModels = availableModels.filter((m) => {
    if (task.mode === "cursor_cli") return m.startsWith("cursor/");
    if (task.mode === "opencode") return m.startsWith("opencode/") || m.startsWith("github-copilot/");
    return true;
  });
  const effectiveModels = task.model && !platformModels.includes(task.model)
    ? [task.model, ...platformModels]
    : platformModels;
  const groups = providerGroup(effectiveModels);
  const providerOrder = task.mode === "cursor_cli"
    ? ["cursor", "other"]
    : ["github-copilot", "opencode", "nvidia"];
  const sortedProviders = [
    ...providerOrder.filter((p) => groups[p]),
    ...Object.keys(groups).filter((p) => !providerOrder.includes(p)),
  ];

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate("/")}
          className="text-gray-400 hover:text-gray-200 transition"
        >
          &larr; Back
        </button>
        <div>
          <h2 className="text-xl font-bold font-mono text-gray-100">{task.shape_key}</h2>
          {task.dtype && (() => {
            const p = parseDtype(task.dtype);
            return (
              <div className="flex items-center gap-1.5 mt-0.5 font-mono text-xs">
                <span className="text-gray-500">dtype:</span>
                {p.symmetric ? (
                  <span className="text-gray-300">{dtypeLabel(p.in)}</span>
                ) : (
                  <>
                    <span className="text-cyan-400">{dtypeLabel(p.in)}</span>
                    <span className="text-gray-600">→</span>
                    <span className="text-amber-400">{dtypeLabel(p.out)}</span>
                  </>
                )}
                <span className="text-gray-600">·</span>
                <span className="text-gray-400">{task.m}×{task.n}×{task.k}</span>
              </div>
            );
          })()}
        </div>
        <StatusBadge status={task.status} />
        <span className="text-sm text-gray-500">
          {({ croqtile: "Croqtile", cuda: "CUDA", triton: "Triton", cute: "CuTe", cutile: "CuTile", helion: "Helion", tilelang: "TileLang" } as Record<string, string>)[task.dsl ?? ""] ?? task.dsl ?? task.mode}
        </span>
        <div className="ml-auto flex items-center gap-2">
          {task.status === "waiting" && (
            <button
              type="button"
              onClick={handlePromote}
              className="px-3 py-1 text-xs rounded bg-emerald-900/50 hover:bg-emerald-800 text-emerald-200 border border-emerald-700 transition"
            >
              Promote to queue
            </button>
          )}
          {(task.status === "pending" || task.status === "completed" || task.status === "cancelled" || task.status === "waiting") && (
            <button
              type="button"
              onClick={() => { setShowResume(!showResume); setResumeIter(String(task.current_iteration)); }}
              className="px-3 py-1 text-xs rounded bg-amber-900/50 hover:bg-amber-800 text-amber-200 border border-amber-700 transition"
            >
              Resume
            </button>
          )}
          {(task.status === "running" || task.status === "pending" || task.status === "waiting") && (
            <button
              type="button"
              onClick={handleCancel}
              className="px-3 py-1 text-xs rounded bg-orange-900/50 hover:bg-orange-800 text-orange-300 border border-orange-800 transition"
            >
              Pause
            </button>
          )}
          {(task.status === "cancelled" || task.status === "completed" || task.status === "pending" || task.status === "waiting") && (
            <button
              type="button"
              onClick={handleDelete}
              className="px-3 py-1 text-xs rounded bg-red-900/50 hover:bg-red-800 text-red-300 border border-red-800 transition"
            >
              Delete
            </button>
          )}
        </div>
      </div>

      {showResume && (
        <div className="flex items-center gap-3 bg-amber-950/30 border border-amber-800 rounded-lg px-4 py-2">
          <span className="text-sm text-amber-200">Resume from iteration:</span>
          <input
            type="number"
            value={resumeIter}
            onChange={(e) => setResumeIter(e.target.value)}
            min={0}
            max={task.current_iteration}
            className="w-24 bg-gray-800 rounded px-2 py-1 text-sm text-gray-100 border border-gray-600 focus:border-amber-500 focus:outline-none"
          />
          <button
            type="button"
            onClick={handleResume}
            className="px-3 py-1 text-xs rounded bg-amber-600 hover:bg-amber-500 text-white font-medium transition"
          >
            Resume
          </button>
          <button
            type="button"
            onClick={() => setShowResume(false)}
            className="px-2 py-1 text-xs text-gray-400 hover:text-gray-200"
          >
            Cancel
          </button>
        </div>
      )}

      {task.status === "waiting" && (
        <p className="text-sm text-slate-400 bg-slate-900/50 border border-slate-700 rounded-lg px-3 py-2">
          This shape is in the <strong className="text-slate-200">backlog</strong> (imported from{" "}
          <code className="text-slate-300">tuning/state.json</code>). The scheduler only runs{" "}
          <strong className="text-slate-200">pending</strong> tasks. Use <em>Promote to queue</em> when you
          want this job eligible for the next run.
        </p>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: "Progress", value: `${task.current_iteration} / ${task.max_iterations} (${pct}%)` },
          { label: "Baseline", value: task.baseline_tflops != null ? `${task.baseline_tflops.toFixed(1)} TFLOPS` : "--" },
          { label: "Best", value: task.best_tflops != null ? `${task.best_tflops.toFixed(1)} TFLOPS` : "--" },
          { label: "Elapsed", value: elapsed },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-800 rounded-lg p-3 border border-gray-700">
            <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
            <div className="text-lg font-semibold text-gray-200 mt-1">{value}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Model</h3>
          <div className="text-base font-semibold text-gray-100">{taskModelLabel}</div>
          <div className="mt-1 flex items-center gap-2">
            <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${task.mode === "cursor_cli" ? "bg-violet-500/20 text-violet-300" : "bg-cyan-500/20 text-cyan-300"}`}>
              {platformLabel}
            </span>
            <span className="font-mono text-xs text-gray-500">
              {task.model ?? "inherited at dispatch"}{task.variant ? ` --variant ${task.variant}` : ""}
            </span>
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_auto]">
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Model</label>
                <select
                  value={editModel}
                  onChange={(e) => setEditModel(e.target.value)}
                  disabled={!modelEditable || savingModel || sortedProviders.length === 0}
                  className="w-full rounded bg-gray-900 border border-gray-600 px-2 py-1.5 text-sm text-gray-100 disabled:opacity-60"
                >
                  {sortedProviders.length === 0 && <option value="">No models available</option>}
                  {sortedProviders.map((provider) => (
                    <optgroup key={provider} label={provider}>
                      {groups[provider].map((item) => (
                        <option key={item} value={item}>{modelLabel(item)}</option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">Variant</label>
                <select
                  value={editVariant}
                  onChange={(e) => setEditVariant(e.target.value)}
                  disabled={!modelEditable || savingModel || modelHasBuiltinVariant}
                  className="w-full rounded bg-gray-900 border border-gray-600 px-2 py-1.5 text-sm text-gray-100 disabled:opacity-60"
                >
                  {effectiveVariants.map((v) => (
                    <option key={v} value={v}>{v || "(none)"}</option>
                  ))}
                </select>
              </div>
            </div>
            <button
              type="button"
              onClick={handleSaveModel}
              disabled={!modelEditable || savingModel || !modelChanged}
              className="self-end px-3 py-2 rounded bg-cyan-700 hover:bg-cyan-600 disabled:opacity-50 text-white text-sm"
            >
              {savingModel ? "Saving..." : "Save model"}
            </button>
          </div>
          {!modelEditable && (
            <p className="mt-2 text-xs text-amber-400">Model is locked while running or after completion.</p>
          )}
        </div>

        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-semibold text-gray-400 mb-2">Agent Session</h3>
          <div className="font-mono text-sm text-cyan-300 break-all">{task.opencode_session_id ?? "Not captured yet"}</div>
          <div className="mt-1 text-xs text-gray-500">Session ID is captured from the live agent logs for this task.</div>
        </div>
      </div>

      {task.error_message && (
        <div className="rounded-lg border border-red-900 bg-red-950/30 p-4 text-sm text-red-200">
          <div className="font-semibold">Last error</div>
          <div className="mt-2 whitespace-pre-wrap font-mono text-xs text-red-100">{task.error_message}</div>
        </div>
      )}

      <div className="w-full bg-gray-700 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-semibold text-gray-400 mb-3">TFLOPS Over Iterations</h3>
        <TflopsChart logs={iterLogs} baseline={task.baseline_tflops ?? null} />
      </div>

      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Iteration Log</h3>

        {/* Request-iteration summary */}
        {(() => {
          const groups: Record<number, { min: number; max: number }> = {};
          for (const log of iterLogs) {
            const rn = log.request_number ?? 0;
            if (!groups[rn]) groups[rn] = { min: log.iteration, max: log.iteration };
            else {
              groups[rn].min = Math.min(groups[rn].min, log.iteration);
              groups[rn].max = Math.max(groups[rn].max, log.iteration);
            }
          }
          const entries = Object.entries(groups)
            .map(([k, v]) => ({ rn: Number(k), ...v }))
            .filter(e => e.rn > 0)
            .sort((a, b) => a.rn - b.rn);
          if (entries.length === 0) return null;
          return (
            <div className="mb-3 flex flex-wrap gap-2">
              {entries.map(e => (
                <span key={e.rn} className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300 font-mono">
                  Request #{e.rn}: iter{String(e.min).padStart(2, "0")}
                  {e.max > e.min ? `–${String(e.max).padStart(2, "0")}` : ""}
                </span>
              ))}
            </div>
          );
        })()}

        <div className="overflow-x-auto max-h-64 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-gray-800">
              <tr className="text-left text-gray-500 border-b border-gray-700">
                <th className="py-2 px-3">#</th>
                <th className="py-2 px-3">Req</th>
                <th className="py-2 px-3">TFLOPS</th>
                <th className="py-2 px-3">Decision</th>
                <th className="py-2 px-3">Bottleneck</th>
                <th className="py-2 px-3">Idea</th>
              </tr>
            </thead>
            <tbody>
              {[...iterLogs].sort((a, b) => b.iteration - a.iteration).map((log) => {
                const isBL = isBaselineEntry(log);
                return (
                <tr key={log.id} className={`border-b border-gray-700/50 ${isBL ? "bg-amber-950/20" : ""}`}>
                  <td className="py-1.5 px-3 text-gray-400">
                    {isBL ? <span className="text-amber-400 font-semibold">base</span> : log.iteration}
                  </td>
                  <td className="py-1.5 px-3 text-gray-500 font-mono">
                    {log.request_number ? `#${log.request_number}` : "--"}
                  </td>
                  <td className={`py-1.5 px-3 font-mono ${isBL ? "text-amber-300" : "text-gray-200"}`}>
                    {log.tflops?.toFixed(1) ?? "--"}
                    {isBL && " ★"}
                  </td>
                  <td className="py-1.5 px-3">
                    {isBL ? (
                      <span className="text-amber-400 text-[10px] uppercase tracking-wider">cuBLAS rooftop</span>
                    ) : (
                      <span className={log.decision === "KEEP" ? "text-emerald-400" : "text-red-400"}>
                        {log.decision ?? "--"}
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 px-3 text-gray-400">{log.bottleneck ?? "--"}</td>
                  <td className="py-1.5 px-3 text-gray-400 max-w-xs truncate">{log.idea_summary ?? "--"}</td>
                </tr>
                );
              })}
              {iterLogs.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-4 text-center text-gray-600">
                    No iterations recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Agent Session History</h3>
        <SessionHistory history={sessionHistory} />
      </div>

      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Runtime Event Log</h3>
        <LiveLog logs={agentLogs} />
      </div>
    </div>
  );
}

function modelLabel(id: string): string {
  const slash = id.indexOf("/");
  return slash > 0 ? id.slice(slash + 1) : id;
}

function formatDuration(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}
