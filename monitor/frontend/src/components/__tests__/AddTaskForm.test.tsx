import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { AddTaskForm } from "../AddTaskForm";

const defaultProps = {
  availableModels: ["opencode/qwen3.6-plus-free", "opencode/minimax-m2.5-free"],
  availableVariants: ["", "minimal", "low", "medium", "high", "xhigh", "max"],
  onCreated: vi.fn(),
  onCancel: vi.fn(),
};

describe("AddTaskForm", () => {
  it("renders all form fields", () => {
    render(<AddTaskForm {...defaultProps} />);
    expect(screen.getByText("Add Kernel Tuning Task")).toBeInTheDocument();
    expect(screen.getByText("Input Type")).toBeInTheDocument();
    expect(screen.getByText("Output Type")).toBeInTheDocument();
    expect(screen.getByText("Mode")).toBeInTheDocument();
    expect(screen.getByText("Model")).toBeInTheDocument();
    const spinbuttons = screen.getAllByRole("spinbutton");
    expect(spinbuttons).toHaveLength(3);
  });

  it("shows cancel and submit buttons", () => {
    render(<AddTaskForm {...defaultProps} />);
    expect(screen.getByText("Cancel")).toBeInTheDocument();
    expect(screen.getByText("Add Task")).toBeInTheDocument();
  });

  it("calls onCancel when cancel is clicked", () => {
    const onCancel = vi.fn();
    render(<AddTaskForm {...defaultProps} onCancel={onCancel} />);
    fireEvent.click(screen.getByText("Cancel"));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("shows max iterations for from_current_best", () => {
    render(<AddTaskForm {...defaultProps} />);
    expect(screen.getByText("Max iterations: 30")).toBeInTheDocument();
  });

  it("updates max iterations when mode changes to from_scratch", () => {
    render(<AddTaskForm {...defaultProps} />);
    const modeSelect = screen.getByDisplayValue("From Current Best (30 iter)");
    fireEvent.change(modeSelect, { target: { value: "from_scratch" } });
    expect(screen.getByText("Max iterations: 150")).toBeInTheDocument();
  });

  it("shows validation error for M < 128", async () => {
    render(<AddTaskForm {...defaultProps} />);
    const inputs = screen.getAllByRole("spinbutton");
    fireEvent.change(inputs[0], { target: { value: "64" } });
    fireEvent.change(inputs[1], { target: { value: "512" } });
    fireEvent.change(inputs[2], { target: { value: "512" } });
    const form = screen.getByText("Add Task").closest("form")!;
    fireEvent.submit(form);
    await waitFor(() => {
      expect(screen.getByText(/M must be/)).toBeInTheDocument();
    });
  });

  it("shows validation error for N < 256", async () => {
    render(<AddTaskForm {...defaultProps} />);
    const inputs = screen.getAllByRole("spinbutton");
    fireEvent.change(inputs[0], { target: { value: "256" } });
    fireEvent.change(inputs[1], { target: { value: "128" } });
    fireEvent.change(inputs[2], { target: { value: "512" } });
    fireEvent.submit(screen.getByText("Add Task").closest("form")!);
    await waitFor(() => {
      expect(screen.getByText(/N must be/)).toBeInTheDocument();
    });
  });
});
