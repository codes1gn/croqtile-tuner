import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { TaskList } from "../TaskList";
import type { TaskData } from "../../api";

const mockTask: TaskData = {
  id: 1,
  shape_key: "gemm_sp_f16_768x768x768",
  op_type: "gemm_sp",
  dtype: "f16",
  m: 768,
  n: 768,
  k: 768,
  mode: "opencode",
  dsl: "croqtile",
  max_iterations: 30,
  status: "running",
  current_iteration: 10,
  best_tflops: 55.3,
  baseline_tflops: 45.0,
  best_kernel: null,
  model: "opencode/qwen3.6-plus-free",
  variant: "",
  agent_type: "opencode",
  device: "NVIDIA GeForce RTX 3070",
  opencode_session_id: "ses_abc123",
  error_message: null,
  created_at: "2026-04-03T10:00:00",
  updated_at: "2026-04-03T10:05:00",
  started_at: "2026-04-03T10:00:00",
  completed_at: null,
  request_budget: 0,
  request_number: 0,
};

describe("TaskList", () => {
  it("shows empty state when no tasks", () => {
    render(
      <MemoryRouter>
        <TaskList tasks={[]} activeTaskId={null} />
      </MemoryRouter>,
    );
    expect(screen.getByText("No tuning tasks")).toBeInTheDocument();
  });

  it("renders task rows with correct columns", () => {
    render(
      <MemoryRouter>
        <TaskList tasks={[mockTask]} activeTaskId={1} />
      </MemoryRouter>,
    );
    expect(screen.getByText("gemm_sp")).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
    expect(screen.getByText("Croqtile")).toBeInTheDocument();
    expect(screen.getByText("55.3")).toBeInTheDocument();
  });

  it("renders multiple task rows", () => {
    const tasks: TaskData[] = [
      mockTask,
      { ...mockTask, id: 2, shape_key: "gemm_sp_e4m3_512x8192x8192_fs", op_type: "gemm_sp", status: "pending", current_iteration: 0, best_tflops: null },
    ];
    render(
      <MemoryRouter>
        <TaskList tasks={tasks} activeTaskId={null} />
      </MemoryRouter>,
    );
    const rows = screen.getAllByRole("row");
    expect(rows.length).toBe(3); // header + 2 data rows
  });
});
