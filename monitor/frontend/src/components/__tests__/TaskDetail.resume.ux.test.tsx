import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { TaskDetail } from "../TaskDetail";
import type { TaskData } from "../../api";
import { api } from "../../api";

vi.mock("../../api", () => ({
  api: {
    getTask: vi.fn(),
    getIterationLogs: vi.fn(),
    getAgentLogs: vi.fn(),
    getSessionHistory: vi.fn(),
    getTaskSessions: vi.fn(),
    getActivityLog: vi.fn(),
    getModelSettings: vi.fn(),
    resumeTask: vi.fn(),
    cancelTask: vi.fn(),
    promoteTask: vi.fn(),
    deleteTask: vi.fn(),
    updateTask: vi.fn(),
  },
}));

const baseTask: TaskData = {
  id: 9,
  task_uid: "7988070b0e82",
  shape_key: "sm90_NVIDIA_H800_PCIe/croqtile/matmul_fp16fp32_512x16384x16384/claude-4-5-opus-high",
  op_type: "matmul",
  dtype: "fp16fp32",
  m: 512,
  n: 16384,
  k: 16384,
  mode: "cursor_cli",
  dsl: "croqtile",
  max_iterations: 30,
  status: "completed",
  current_iteration: 30,
  best_tflops: 349.73,
  baseline_tflops: 423.0,
  best_kernel: null,
  model: "claude-4-5-opus-high",
  variant: "",
  request_budget: 1,
  request_number: 0,
  agent_type: "cursor_cli",
  device: "NVIDIA H800 PCIe",
  opencode_session_id: null,
  session_ids: [],
  error_message: null,
  created_at: "2026-04-17T21:00:00Z",
  updated_at: "2026-04-17T21:10:00Z",
  started_at: null,
  completed_at: "2026-04-17T21:10:00Z",
};

describe("TaskDetail resume UX", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  function setupCommonSuccessfulMocks() {
    vi.mocked(api.getTask).mockResolvedValue(baseTask);
    vi.mocked(api.getIterationLogs).mockResolvedValue([]);
    vi.mocked(api.getAgentLogs).mockResolvedValue([]);
    vi.mocked(api.getSessionHistory).mockResolvedValue({
      session_id: null,
      session_title: null,
      session_directory: null,
      entries: [],
    });
    vi.mocked(api.getTaskSessions).mockResolvedValue([]);
    vi.mocked(api.getActivityLog).mockResolvedValue([]);
  }

  it("updates local task state and closes resume panel after resume", async () => {
    setupCommonSuccessfulMocks();
    vi.mocked(api.getModelSettings).mockResolvedValue({
      default_model: "claude-4-5-opus-high",
      default_variant: "",
      available_models: ["claude-4-5-opus-high"],
      available_variants: [""],
    });

    const resumedTask: TaskData = {
      ...baseTask,
      status: "pending",
      completed_at: null,
    };
    vi.mocked(api.resumeTask).mockResolvedValue(resumedTask);

    render(
      <MemoryRouter initialEntries={["/tasks/9"]}>
        <Routes>
          <Route path="/tasks/:id" element={<TaskDetail sseEvent={null} />} />
        </Routes>
      </MemoryRouter>,
    );

    await screen.findByText(baseTask.shape_key);
    expect(screen.getByText("completed")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: "Resume" })[0]);

    const input = screen.getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "12" } });
    fireEvent.click(screen.getAllByRole("button", { name: "Resume" })[1]);

    await waitFor(() => {
      expect(api.resumeTask).toHaveBeenCalledWith(baseTask.id, 12);
    });
    await waitFor(() => {
      expect(screen.queryByText("Resume from iteration:")).not.toBeInTheDocument();
    });

    expect(screen.getByText("pending")).toBeInTheDocument();
  });

  it("still renders task detail when model settings endpoint fails", async () => {
    setupCommonSuccessfulMocks();
    vi.mocked(api.getModelSettings).mockRejectedValue(new Error("500: Internal Server Error"));

    render(
      <MemoryRouter initialEntries={["/tasks/9"]}>
        <Routes>
          <Route path="/tasks/:id" element={<TaskDetail sseEvent={null} />} />
        </Routes>
      </MemoryRouter>,
    );

    // Critical path should still succeed even if /api/settings/model fails.
    expect(await screen.findByText(baseTask.shape_key)).toBeInTheDocument();
    expect(screen.getByText("completed")).toBeInTheDocument();
    expect(screen.queryByText("500: Internal Server Error")).not.toBeInTheDocument();
  });
});
