import { useState, useCallback } from "react";
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

function providerGroup(models: string[]): Record<string, string[]> {
  const groups: Record<string, string[]> = {};
  for (const m of models) {
    const slash = m.indexOf("/");
    const provider = slash > 0 ? m.slice(0, slash) : "other";
    (groups[provider] ??= []).push(m);
  }
  return groups;
}

function shortName(model: string): string {
  const slash = model.indexOf("/");
  return slash > 0 ? model.slice(slash + 1) : model;
}

export function SystemStatusPanel({ health, onRefresh }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedVariant, setSelectedVariant] = useState("");
  const [togglingAutoWake, setTogglingAutoWake] = useState(false);

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

  const curModel = selectedModel || health.default_model;
  const curVariant = selectedVariant || health.default_variant;

  const queueItems = [
    { label: "Waiting", value: health.task_counts.waiting ?? 0 },
    { label: "Pending", value: health.task_counts.pending ?? 0 },
    { label: "Running", value: health.task_counts.running ?? 0 },
    { label: "Stopped", value: health.task_counts.stopped ?? 0 },
    { label: "Failed", value: health.task_counts.failed ?? 0 },
  ];

  const handleApply = async () => {
    setSaving(true);
    setError("");
    try {
      await api.setDefaultModel(curModel, curVariant);
      await onRefresh();
      setSelectedModel("");
      setSelectedVariant("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update model");
    } finally {
      setSaving(false);
    }
  };

  const groups = providerGroup(health.available_models);
  const providerOrder = ["github-copilot", "opencode", "nvidia"];
  const sortedProviders = [
    ...providerOrder.filter((p) => groups[p]),
    ...Object.keys(groups).filter((p) => !providerOrder.includes(p)),
  ];

  const displayLabel =
    health.default_variant
      ? `${shortName(health.default_model)} (${health.default_variant})`
      : shortName(health.default_model);

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
            Default model: <span className="font-medium text-white">{displayLabel}</span>
            <span className="ml-2 font-mono text-xs text-gray-500">{health.default_model}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          {queueItems.map((item) => (
            <div key={item.label} className="rounded-xl border border-gray-800 bg-black/20 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.25em] text-gray-500">{item.label}</div>
              <div className="mt-1 text-2xl font-semibold text-gray-100">{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="rounded-xl border border-gray-800 bg-black/20 px-4 py-3 text-sm text-gray-300 lg:flex-1">
          <div className="text-[11px] uppercase tracking-[0.25em] text-gray-500">GPU</div>
          <pre className="mt-2 whitespace-pre-wrap font-mono text-xs text-gray-300">{health.gpu_info ?? "Unavailable"}</pre>
        </div>

        <div className="lg:w-[32rem]">
          <button
            type="button"
            onClick={() => setExpanded((prev) => !prev)}
            className="flex w-full items-center justify-between rounded-xl border border-cyan-800/60 bg-cyan-950/20 px-4 py-3 text-left transition hover:border-cyan-700 hover:bg-cyan-950/30"
          >
            <div>
              <div className="text-[11px] uppercase tracking-[0.25em] text-cyan-400">Model control</div>
              <div className="mt-1 text-sm text-cyan-50">Choose the default model + variant for new tasks</div>
            </div>
            <span className="text-cyan-300">{expanded ? "Hide" : "Show"}</span>
          </button>

          {expanded && (
            <div className="mt-2 space-y-3 rounded-xl border border-gray-800 bg-black/20 p-3">
              <div>
                <label className="block text-[11px] uppercase tracking-[0.2em] text-gray-500 mb-1">Model</label>
                <select
                  value={curModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  disabled={saving}
                  className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm text-gray-100 border border-gray-700 focus:border-cyan-500 focus:outline-none"
                >
                  {sortedProviders.map((provider) => (
                    <optgroup key={provider} label={provider}>
                      {groups[provider].map((m) => (
                        <option key={m} value={m}>{shortName(m)}</option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-[11px] uppercase tracking-[0.2em] text-gray-500 mb-1">Variant (reasoning effort)</label>
                <select
                  value={curVariant}
                  onChange={(e) => setSelectedVariant(e.target.value)}
                  disabled={saving}
                  className="w-full bg-gray-800 rounded-lg px-3 py-2 text-sm text-gray-100 border border-gray-700 focus:border-cyan-500 focus:outline-none"
                >
                  {health.available_variants.map((v) => (
                    <option key={v} value={v}>{v || "(none)"}</option>
                  ))}
                </select>
              </div>

              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-gray-500">
                  {curModel}{curVariant ? ` --variant ${curVariant}` : ""}
                </span>
                <button
                  type="button"
                  onClick={handleApply}
                  disabled={saving || (curModel === health.default_model && curVariant === health.default_variant)}
                  className="px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium transition disabled:opacity-40"
                >
                  {saving ? "Saving..." : "Apply"}
                </button>
              </div>

              {error && <p className="rounded-lg bg-red-950/40 px-3 py-2 text-sm text-red-300">{error}</p>}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
