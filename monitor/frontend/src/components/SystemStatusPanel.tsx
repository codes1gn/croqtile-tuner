import { useState, useCallback, useEffect } from "react";
import { api, type HealthData } from "../api";

interface Props {
  health: HealthData | null;
  onRefresh: () => Promise<void>;
}

function AutoWakeToggle({ enabled, onToggle, disabled }: { enabled: boolean; onToggle: () => void; disabled: boolean }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      className={`
        relative inline-flex h-6 w-11 items-center rounded-full transition-colors
        ${enabled ? "bg-emerald-600" : "bg-gray-700"}
        ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer hover:opacity-90"}
      `}
    >
      <span
        className={`
          inline-block h-4 w-4 transform rounded-full bg-white transition-transform
          ${enabled ? "translate-x-6" : "translate-x-1"}
        `}
      />
    </button>
  );
}

function GpuInfoDisplay({ raw }: { raw: string }) {
  const parts = raw.split(",").map((s) => s.trim());
  if (parts.length < 4) {
    return <pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-gray-300">{raw}</pre>;
  }
  const [, name, util, memUsed, memTotal, temp] = parts;
  const memPct = memTotal ? Math.round((parseFloat(memUsed) / parseFloat(memTotal)) * 100) : null;
  return (
    <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
      <div className="col-span-2 text-sm font-semibold text-gray-100 mb-1">{name}</div>
      <div className="text-gray-500">Utilization</div>
      <div className="text-gray-200 font-mono">{util}</div>
      <div className="text-gray-500">Memory</div>
      <div className="text-gray-200 font-mono">{memUsed} / {memTotal}{memPct != null ? ` (${memPct}%)` : ""}</div>
      {temp && <><div className="text-gray-500">Temperature</div><div className="text-gray-200 font-mono">{temp} C</div></>}
    </div>
  );
}

export function SystemStatusPanel({ health, onRefresh }: Props) {
  const [error, setError] = useState("");
  const [togglingAutoWake, setTogglingAutoWake] = useState(false);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedVariant, setSelectedVariant] = useState("");
  const [savingModel, setSavingModel] = useState(false);

  useEffect(() => {
    if (health && !selectedModel) {
      setSelectedModel(health.default_model);
      setSelectedVariant(health.default_variant);
    }
  }, [health, selectedModel]);

  const handleModelSave = useCallback(async () => {
    if (!selectedModel) return;
    setSavingModel(true);
    try {
      await api.setDefaultModel(selectedModel, selectedVariant);
      await onRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save model");
    } finally {
      setSavingModel(false);
    }
  }, [selectedModel, selectedVariant, onRefresh]);

  const handleAutoWakeToggle = useCallback(async () => {
    if (!health) return;
    setTogglingAutoWake(true);
    try {
      await api.setAutoWakeEnabled(!health.auto_wake_enabled);
      await onRefresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to toggle auto-wake");
    } finally {
      setTogglingAutoWake(false);
    }
  }, [health, onRefresh]);

  if (!health) {
    return (
      <section className="rounded-2xl border border-gray-800 bg-gray-900/80 p-4 text-sm text-gray-500">
        Loading system status...
      </section>
    );
  }

  const queueItems = [
    { label: "Waiting", value: health.task_counts.waiting ?? 0 },
    { label: "Pending", value: health.task_counts.pending ?? 0 },
    { label: "Running", value: health.task_counts.running ?? 0 },
    { label: "Stopped", value: health.task_counts.stopped ?? 0 },
    { label: "Completed", value: health.task_counts.completed ?? 0 },
    { label: "Failed", value: health.task_counts.failed ?? 0 },
    { label: "Cancelled", value: health.task_counts.cancelled ?? 0 },
  ];

  return (
    <section className="rounded-2xl border border-gray-800 bg-gradient-to-br from-gray-900 via-gray-900 to-slate-950 p-4 shadow-[0_20px_80px_rgba(0,0,0,0.35)]">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.3em] text-cyan-500">System</div>
          <div className="mt-1 flex items-center gap-3">
            <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${health.scheduler_running ? "bg-emerald-500/20 text-emerald-300" : "bg-red-500/20 text-red-300"}`}>
              {health.scheduler_running ? "Scheduler running" : "Scheduler stopped"}
            </span>
            <span className="text-sm text-gray-400">
              Active task: {health.active_task_id ?? "none"}
            </span>
          </div>
          <div className="mt-2 flex items-center gap-3">
            <AutoWakeToggle
              enabled={health.auto_wake_enabled}
              onToggle={handleAutoWakeToggle}
              disabled={togglingAutoWake}
            />
            <span className="text-sm text-gray-400">
              Auto-wake: {health.auto_wake_enabled ? (
                <span className="text-emerald-400">ON (auto-starts opencode)</span>
              ) : (
                <span className="text-amber-400">OFF (monitor only)</span>
              )}
            </span>
          </div>
          <div className="mt-3 text-sm text-gray-300">
            Model assignment is task-scoped. Pick model + variant when creating or editing a task.
          </div>
        </div>

        <div className="grid grid-cols-4 gap-2 sm:grid-cols-7">
          {queueItems.map((item) => (
            <div key={item.label} className="rounded-lg border border-gray-800 bg-black/20 px-3 py-2 text-center">
              <div className="text-[10px] uppercase tracking-wider text-gray-500">{item.label}</div>
              <div className="mt-0.5 text-xl font-semibold text-gray-100">{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="rounded-xl border border-gray-800 bg-black/20 px-4 py-3 text-sm text-gray-300 lg:flex-1">
          <div className="text-[11px] uppercase tracking-[0.25em] text-gray-500">GPU</div>
          {health.gpu_info && health.gpu_info !== "nvidia-smi not available" ? (
            <GpuInfoDisplay raw={health.gpu_info} />
          ) : (
            <p className="mt-2 text-xs text-gray-500">Unavailable</p>
          )}
        </div>

        <div className="lg:w-[32rem]">
          <div className="rounded-xl border border-cyan-800/60 bg-cyan-950/20 px-4 py-3">
            <div className="text-[11px] uppercase tracking-[0.25em] text-cyan-400">Default Model for AutoTune <span className="text-gray-500 normal-case tracking-normal">(OpenCode)</span></div>
            <div className="mt-2 flex items-center gap-2">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="flex-1 bg-gray-800 rounded px-2 py-1.5 text-sm text-gray-100 border border-gray-700 focus:border-cyan-500 focus:outline-none"
              >
                {(health?.available_models ?? []).filter((m) => m.startsWith("opencode/") || m.startsWith("github-copilot/")).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <select
                value={selectedVariant}
                onChange={(e) => setSelectedVariant(e.target.value)}
                className="w-24 bg-gray-800 rounded px-2 py-1.5 text-sm text-gray-100 border border-gray-700 focus:border-cyan-500 focus:outline-none"
              >
                {(health?.available_variants ?? [""]).map((v) => (
                  <option key={v} value={v}>{v || "(none)"}</option>
                ))}
              </select>
              <button
                type="button"
                onClick={handleModelSave}
                disabled={savingModel || (selectedModel === health?.default_model && selectedVariant === health?.default_variant)}
                className="px-3 py-1.5 rounded text-xs font-medium bg-cyan-700 hover:bg-cyan-600 text-white transition disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {savingModel ? "..." : "Save"}
              </button>
            </div>
            <p className="mt-1.5 text-xs text-gray-500">Used when auto-wake creates new tasks. Per-task model can be set in task details.</p>
            {error && <p className="mt-3 rounded-lg bg-red-950/40 px-3 py-2 text-sm text-red-300">{error}</p>}
          </div>
        </div>
      </div>
    </section>
  );
}
