const BASE = "/api";

export interface TaskData {
  id: number;
  task_uid: string;
  shape_key: string;
  op_type: string | null;
  dtype: string;
  m: number;
  n: number;
  k: number;
  mode: string;
  dsl: string | null;
  max_iterations: number;
  status: string;
  current_iteration: number;
  best_tflops: number | null;
  baseline_tflops: number | null;
  best_kernel: string | null;
  model: string | null;
  variant: string | null;
  request_budget: number;
  request_number: number;
  agent_type: string | null;
  device: string | null;
  opencode_session_id: string | null;
  session_ids: string[];
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
  request_number: number | null;
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

export interface TaskSessionData {
  id: number;
  task_id: number;
  session_id: string;
  agent_type: string | null;
  model: string | null;
  request_number: number | null;
  started_at: string | null;
  ended_at: string | null;
}

export interface ActivityLogEntry {
  ts: string;
  tool: string;
  msg: string;
  level: string;
  [key: string]: unknown;
}

export interface HealthData {
  status: string;
  scheduler_running: boolean;
  active_task_id: number | null;
  active_task_ids: number[];
  auto_wake_enabled: boolean;
  use_proxy: boolean;
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

export interface ProxySettingsData {
  use_proxy: boolean;
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

  createTask: (data: { op_type: string; dtype: string; m: number; n: number; k: number; dsl: string; mode: string; model: string; variant?: string; request_budget?: number }) =>
    request<TaskData>("/tasks", { method: "POST", body: JSON.stringify(data) }),

  updateTask: (id: number, data: { status?: string; model?: string; variant?: string }) =>
    request<TaskData>(`/tasks/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

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

  getSessionHistory: (id: number, limit = 200, sessionId?: string) =>
    request<SessionHistoryData>(
      `/tasks/${id}/session-history?limit=${limit}${sessionId ? `&session_id=${encodeURIComponent(sessionId)}` : ""}`
    ),

  getTaskSessions: (id: number) =>
    request<TaskSessionData[]>(`/tasks/${id}/sessions`),

  getActivityLog: (id: number, limit = 200) =>
    request<ActivityLogEntry[]>(`/tasks/${id}/activity-log?limit=${limit}`),

  getHealth: () => request<HealthData>("/health"),

  getModelSettings: () => request<ModelSettingsData>("/settings/model"),

  setDefaultModel: (default_model: string, default_variant: string = "") =>
    request<ModelSettingsData>("/settings/model", {
      method: "PATCH",
      body: JSON.stringify({ default_model, default_variant }),
    }),

  refreshModels: () =>
    request<ModelSettingsData>("/settings/model/refresh", { method: "POST" }),

  getAutoWakeSettings: () => request<AutoWakeSettingsData>("/settings/auto-wake"),

  setAutoWakeEnabled: (auto_wake_enabled: boolean) =>
    request<AutoWakeSettingsData>("/settings/auto-wake", {
      method: "PATCH",
      body: JSON.stringify({ auto_wake_enabled }),
    }),

  getProxySettings: () => request<ProxySettingsData>("/settings/proxy"),

  setUseProxy: (use_proxy: boolean) =>
    request<ProxySettingsData>("/settings/proxy", {
      method: "PATCH",
      body: JSON.stringify({ use_proxy }),
    }),
};
