const BASE = "/api";

export interface TaskData {
  id: number;
  shape_key: string;
  op_type: string | null;
  dtype: string;
  m: number;
  n: number;
  k: number;
  mode: string;
  max_iterations: number;
  status: string;
  current_iteration: number;
  best_tflops: number | null;
  baseline_tflops: number | null;
  best_kernel: string | null;
  model: string | null;
  variant: string | null;
  agent_type: string | null;
  opencode_session_id: string | null;
  error_message: string | null;
  created_at: string | null;
  updated_at: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface IterationLogData {
  id: number;
  task_id: number;
  iteration: number;
  kernel_path: string | null;
  tflops: number | null;
  decision: string | null;
  bottleneck: string | null;
  idea_summary: string | null;
  logged_at: string | null;
}

export interface AgentLogData {
  id: number;
  task_id: number;
  level: string;
  message: string;
  timestamp: string | null;
}

export interface SessionHistoryEntryData {
  id: string;
  message_id: string;
  role: string;
  kind: string;
  text: string;
  tool: string | null;
  status: string | null;
  timestamp: string | null;
}

export interface SessionHistoryData {
  session_id: string | null;
  session_title: string | null;
  session_directory: string | null;
  entries: SessionHistoryEntryData[];
}

export interface HealthData {
  status: string;
  scheduler_running: boolean;
  active_task_id: number | null;
  auto_wake_enabled: boolean;
  gpu_info: string | null;
  default_model: string;
  default_variant: string;
  available_models: string[];
  available_variants: string[];
  task_counts: Record<string, number>;
}

export interface AutoWakeSettingsData {
  auto_wake_enabled: boolean;
}

export interface ModelSettingsData {
  default_model: string;
  default_variant: string;
  available_models: string[];
  available_variants: string[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json();
}

export const api = {
  listTasks: (status?: string) =>
    request<TaskData[]>(`/tasks${status ? `?status=${status}` : ""}`),

  createTask: (data: { op_type: string; dtype: string; m: number; n: number; k: number; mode: string; model?: string; variant?: string }) =>
    request<TaskData>("/tasks", { method: "POST", body: JSON.stringify(data) }),

  getTask: (id: number) => request<TaskData>(`/tasks/${id}`),

  cancelTask: (id: number) =>
    request<TaskData>(`/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status: "cancelled" }),
    }),

  promoteTask: (id: number) =>
    request<TaskData>(`/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ status: "pending" }),
    }),

  retryTask: (id: number) =>
    request<TaskData>(`/tasks/${id}/retry`, {
      method: "POST",
    }),

  resumeTask: (id: number, fromIteration: number) =>
    request<TaskData>(`/tasks/${id}/resume`, {
      method: "POST",
      body: JSON.stringify({ from_iteration: fromIteration }),
    }),

  deleteTask: (id: number) =>
    request<void>(`/tasks/${id}`, { method: "DELETE" }),

  getIterationLogs: (id: number) =>
    request<IterationLogData[]>(`/tasks/${id}/logs`),

  getAgentLogs: (id: number, limit = 100) =>
    request<AgentLogData[]>(`/tasks/${id}/agent-logs?limit=${limit}`),

  getSessionHistory: (id: number, limit = 200) =>
    request<SessionHistoryData>(`/tasks/${id}/session-history?limit=${limit}`),

  getHealth: () => request<HealthData>("/health"),

  getModelSettings: () => request<ModelSettingsData>("/settings/model"),

  setDefaultModel: (default_model: string, default_variant: string = "") =>
    request<ModelSettingsData>("/settings/model", {
      method: "PATCH",
      body: JSON.stringify({ default_model, default_variant }),
    }),

  getAutoWakeSettings: () => request<AutoWakeSettingsData>("/settings/auto-wake"),

  setAutoWakeEnabled: (auto_wake_enabled: boolean) =>
    request<AutoWakeSettingsData>("/settings/auto-wake", {
      method: "PATCH",
      body: JSON.stringify({ auto_wake_enabled }),
    }),
};
