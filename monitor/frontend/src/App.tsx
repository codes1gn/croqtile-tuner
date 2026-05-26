import { useState, useEffect, useCallback } from "react";
import { Routes, Route } from "react-router-dom";
import { api, type HealthData, type TaskData } from "./api";
import { useSSE, type SSEEvent } from "./hooks/useSSE";
import { TaskList } from "./components/TaskList";
import { TaskDetail } from "./components/TaskDetail";
import { AddTaskForm } from "./components/AddTaskForm";
import { SystemStatusPanel } from "./components/SystemStatusPanel";
import { AgentMonitorPanel } from "./components/AgentMonitorPanel";
import { ThemeSwitcher } from "./components/ThemeSwitcher";

export default function App() {
  const [tasks, setTasks] = useState<TaskData[]>([]);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [tasksError, setTasksError] = useState<string | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [lastEvent, setLastEvent] = useState<SSEEvent | null>(null);

  const loadTasks = useCallback(async () => {
    try {
      setTasks(await api.listTasks());
      setTasksError(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load task list";
      setTasksError(msg);
    }
  }, []);

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await api.getHealth());
      setHealthError(null);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to load system status";
      setHealthError(msg);
    }
  }, []);

  useEffect(() => {
    loadTasks();
    loadHealth();
  }, [loadHealth, loadTasks]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void loadHealth();
      void loadTasks();
    }, 10000);
    return () => window.clearInterval(timer);
  }, [loadHealth, loadTasks]);

  useSSE((event) => {
    setLastEvent(event);
    if (event.type === "task_update") {
      const updated = event.data as unknown as TaskData;
      setTasks((prev) => {
        const idx = prev.findIndex((t) => t.id === updated.id);
        if (idx >= 0) {
          const next = [...prev];
          next[idx] = updated;
          return next;
        }
        return [updated, ...prev];
      });
      void loadHealth();
    } else if (event.type === "task_deleted") {
      const { id } = event.data as { id: number };
      setTasks((prev) => prev.filter((t) => t.id !== id));
      void loadHealth();
    }
  });

  return (
    <div className="min-h-screen" style={{ backgroundColor: "var(--c-bg)" }}>
      <header className="sticky top-0 z-40 backdrop-blur-sm" style={{ backgroundColor: "var(--c-headerBg)", borderBottom: "1px solid var(--c-headerBorder)" }}>
        <div className="max-w-screen-2xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div>
              <h1 className="text-xl font-bold tracking-[0.06em]" style={{ color: "var(--c-text)" }}>CroqTuner</h1>
              <p className="text-xs mt-0.5 tracking-wide" style={{ color: "var(--c-textFaint)" }}>GPU Kernel Tuning Agent</p>
            </div>
            <ThemeSwitcher />
          </div>
          <button
            onClick={() => setShowAdd(true)}
            disabled={health?.read_only_mode}
            className="px-4 py-2 rounded-lg text-white text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40"
            style={{ backgroundColor: "var(--c-accent)" }}
          >
            {health?.read_only_mode ? "Read-only" : "+ Add Task"}
          </button>
        </div>
      </header>

      <main className="max-w-screen-2xl mx-auto px-6 py-6">
        {(tasksError || healthError) && (
          <div className="mb-4 rounded-lg px-4 py-3 text-sm" style={{ border: "1px solid var(--c-warning)", backgroundColor: "color-mix(in srgb, var(--c-warning) 10%, transparent)", color: "var(--c-warning)" }}>
            <div className="font-semibold">Monitor data is partially unavailable</div>
            {tasksError && <div className="mt-1">Tasks: {tasksError}</div>}
            {healthError && <div className="mt-1">Health: {healthError}</div>}
          </div>
        )}
        <div className="mb-6">
          <SystemStatusPanel health={health} onRefresh={loadHealth} />
        </div>
        <div className="mb-6">
          <AgentMonitorPanel />
        </div>
        <Routes>
          <Route path="/" element={<TaskList tasks={tasks} activeTaskIds={health?.active_task_ids ?? []} />} />
          <Route path="/tasks/:id" element={<TaskDetail sseEvent={lastEvent} />} />
        </Routes>
      </main>

      {showAdd && !health?.read_only_mode && (
        <AddTaskForm
          availableModels={health?.available_models ?? []}
          availableVariants={health?.available_variants ?? [""]}
          defaultModel={health?.default_model ?? ""}
          defaultVariant={health?.default_variant ?? ""}
          useProxy={health?.use_proxy ?? false}
          onCreated={() => {
            setShowAdd(false);
            loadTasks();
            loadHealth();
          }}
          onCancel={() => setShowAdd(false)}
          onRefreshModels={async () => {
            const res = await api.refreshModels();
            setHealth((h) => h ? { ...h, available_models: res.available_models } : h);
            return res.available_models;
          }}
          onToggleProxy={async (enabled) => {
            await api.setUseProxy(enabled);
            setHealth((h) => h ? { ...h, use_proxy: enabled } : h);
          }}
        />
      )}
    </div>
  );
}
