import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { StatusBadge } from "../StatusBadge";

describe("StatusBadge", () => {
  it("renders the status text", () => {
    render(<StatusBadge status="running" />);
    expect(screen.getByText("running")).toBeInTheDocument();
  });

  it("applies running style class", () => {
    render(<StatusBadge status="running" />);
    const badge = screen.getByText("running");
    expect(badge.className).toContain("bg-blue-600");
    expect(badge.className).toContain("animate-pulse");
  });

  it("applies completed style class", () => {
    render(<StatusBadge status="completed" />);
    const badge = screen.getByText("completed");
    expect(badge.className).toContain("bg-emerald-600");
  });

  it("applies failed style class", () => {
    render(<StatusBadge status="failed" />);
    const badge = screen.getByText("failed");
    expect(badge.className).toContain("bg-red-600");
  });

  it("applies waiting style class", () => {
    render(<StatusBadge status="waiting" />);
    const badge = screen.getByText("waiting");
    expect(badge.className).toContain("bg-slate-600");
  });

  it("handles unknown status gracefully", () => {
    render(<StatusBadge status="unknown" />);
    expect(screen.getByText("unknown")).toBeInTheDocument();
  });
});
