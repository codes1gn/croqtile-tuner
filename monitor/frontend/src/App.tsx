import { useState, useEffect, useCallback } from "react";
import { Routes, Route } from "react-router-dom";
import { api, type HealthData, type TaskData } from "./api";
import { useSSE, type SSEEvent } from "./hooks/useSSE";
import { TaskList } from "./components/TaskList";
import { TaskDetail } from "./components/TaskDetail";
import { AddTaskForm } from "./components/AddTaskForm";
import { SystemStatusPanel } from "./components/SystemStatusPanel";
import { AgentMonitorPanel } from "./components/AgentMonitorPanel";

export default function App() {
  const [tasks, setTasks] = useState<TaskData[]>([]);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [showAdd, setShowAdd] = useState(false);
  const [lastEvent, setLastEvent] = useState<SSEEvent | null>(null);

  const loadTasks = useCallback(async () => {
    try {
      setTasks(await api.listTasks());
    } catch {
      // silently retry on next interval
    }
  }, []);

  const loadHealth = useCallback(async () => {
    try {
      setHealth(await api.getHealth());
    } catch {
      // silently retry on next interval
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
    <div className="min-h-screen bg-gray-950">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-40">
        <div className="max-w-screen-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-100 tracking-tight">CroqTuner</h1>
            <p className="text-xs text-gray-500">GPU Kernel Tuning Agent</p>
          </div>
          <button
            onClick={() => setShowAdd(true)}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition"
          >
            + Add Task
          </button>
        </div>
      </header>

      <main className="max-w-screen-2xl mx-auto px-6 py-6">
        <div className="mb-6">
          <SystemStatusPanel health={health} onRefresh={loadHealth} />
        </div>
        <div className="mb-6">
          <AgentMonitorPanel />
        </div>
        <Routes>
          <Route path="/" element={<TaskList tasks={tasks} activeTaskId={health?.active_task_id ?? null} />} />
          <Route path="/tasks/:id" element={<TaskDetail sseEvent={lastEvent} />} />
        </Routes>
      </main>

      {showAdd && (
        <AddTaskForm
          availableModels={health?.available_models ?? []}
          availableVariants={health?.available_variants ?? [""]}
          defaultModel={health?.default_model ?? ""}
          defaultVariant={health?.default_variant ?? ""}
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
        />
      )}
    </div>
  );
}
