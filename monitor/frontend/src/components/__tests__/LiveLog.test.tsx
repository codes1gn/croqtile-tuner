import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { LiveLog } from "../LiveLog";
import type { AgentLogData } from "../../api";

describe("LiveLog", () => {
  it("shows empty state", () => {
    render(<LiveLog logs={[]} />);
    expect(screen.getByText("Waiting for agent output...")).toBeInTheDocument();
  });

  it("renders log messages", () => {
    const logs: AgentLogData[] = [
      { id: 1, task_id: 1, level: "info", message: "Compiling iter001...", timestamp: "2026-04-03T10:00:00" },
      { id: 2, task_id: 1, level: "error", message: "nvcc warning", timestamp: "2026-04-03T10:00:01" },
    ];
    render(<LiveLog logs={logs} />);
    expect(screen.getByText(/Compiling iter001/)).toBeInTheDocument();
    expect(screen.getByText(/nvcc warning/)).toBeInTheDocument();
  });
});
