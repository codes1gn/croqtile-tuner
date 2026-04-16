import { useEffect, useState, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  api,
  type TaskData,
  type IterationLogData,
  type AgentLogData,
  type SessionHistoryData,
} from "../api";
import { StatusBadge } from "./StatusBadge";
import { TflopsChart } from "./TflopsChart";
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
    if (!task || task.status !== "running") return;
    const timer = window.setInterval(() => {
      void loadData();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [loadData, task]);

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

  const handleRetry = async () => {
    if (!task) return;
    try {
      const retried = await api.retryTask(task.id);
      navigate(`/tasks/${retried.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to retry");
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
  const taskModelLabel = task.model ? modelLabel(task.model) + variantSuffix : "system default";
  const modelEditable = task.status !== "running" && task.status !== "completed";
  const modelChanged = editModel !== (task.model ?? "") || editVariant !== (task.variant ?? "");
  const groups = providerGroup(availableModels);
  const providerOrder = ["github-copilot", "opencode", "nvidia"];
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
        <h2 className="text-xl font-bold font-mono text-gray-100">{task.shape_key}</h2>
        <StatusBadge status={task.status} />
        <span className="text-sm text-gray-500">
          {task.mode === "from_current_best" ? "from-best" : "from-scratch"}
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
          {(task.status === "failed" || task.status === "completed" || task.status === "cancelled" || task.status === "stopped") && (
            <>
              <button
                type="button"
                onClick={handleRetry}
                className="px-3 py-1 text-xs rounded bg-cyan-900/50 hover:bg-cyan-800 text-cyan-200 border border-cyan-700 transition"
              >
                Retry (fresh)
              </button>
              <button
                type="button"
                onClick={() => { setShowResume(!showResume); setResumeIter(String(task.current_iteration)); }}
                className="px-3 py-1 text-xs rounded bg-amber-900/50 hover:bg-amber-800 text-amber-200 border border-amber-700 transition"
              >
                Resume from iter
              </button>
            </>
          )}
          {(task.status === "running" || task.status === "pending" || task.status === "waiting" || task.status === "stopped") && (
            <button
              type="button"
              onClick={handleCancel}
              className="px-3 py-1 text-xs rounded bg-red-900/50 hover:bg-red-800 text-red-300 border border-red-800 transition"
            >
              Cancel
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
          <div className="mt-1 font-mono text-xs text-gray-500">
            {task.model ?? "inherited at dispatch"}{task.variant ? ` --variant ${task.variant}` : ""}
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
                  disabled={!modelEditable || savingModel}
                  className="w-full rounded bg-gray-900 border border-gray-600 px-2 py-1.5 text-sm text-gray-100 disabled:opacity-60"
                >
                  {availableVariants.map((v) => (
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
          <h3 className="text-sm font-semibold text-gray-400 mb-2">OpenCode Session</h3>
          <div className="font-mono text-sm text-cyan-300 break-all">{task.opencode_session_id ?? "Not captured yet"}</div>
          <div className="mt-1 text-xs text-gray-500">Session ID is captured from the live opencode logs for this task.</div>
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
        <TflopsChart logs={iterLogs} baseline={task.baseline_tflops ?? iterLogs.find((l) => l.iteration === 0)?.tflops ?? null} />
      </div>

      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-semibold text-gray-400 mb-3">Iteration Log</h3>
        <div className="overflow-x-auto max-h-64 overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-gray-800">
              <tr className="text-left text-gray-500 border-b border-gray-700">
                <th className="py-2 px-3">#</th>
                <th className="py-2 px-3">TFLOPS</th>
                <th className="py-2 px-3">Decision</th>
                <th className="py-2 px-3">Bottleneck</th>
                <th className="py-2 px-3">Idea</th>
              </tr>
            </thead>
            <tbody>
              {[...iterLogs].sort((a, b) => b.iteration - a.iteration).map((log) => {
                const isBaseline = log.iteration === 0;
                return (
                <tr key={log.id} className={`border-b border-gray-700/50 ${isBaseline ? "bg-amber-950/20" : ""}`}>
                  <td className="py-1.5 px-3 text-gray-400">
                    {isBaseline ? <span className="text-amber-400 font-semibold">base</span> : log.iteration}
                  </td>
                  <td className={`py-1.5 px-3 font-mono ${isBaseline ? "text-amber-300" : "text-gray-200"}`}>
                    {log.tflops?.toFixed(1) ?? "--"}
                    {isBaseline && " ★"}
                  </td>
                  <td className="py-1.5 px-3">
                    {isBaseline ? (
                      <span className="text-amber-400 text-[10px] uppercase tracking-wider">rooftop</span>
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
                  <td colSpan={5} className="py-4 text-center text-gray-600">
                    No iterations recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
        <h3 className="text-sm font-semibold text-gray-400 mb-3">OpenCode Session History</h3>
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
