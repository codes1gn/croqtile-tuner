import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SessionHistory } from "../SessionHistory";
import type { SessionHistoryData } from "../../api";

describe("SessionHistory", () => {
  it("shows empty state when no session id exists", () => {
    render(
      <SessionHistory
        history={{
          session_id: null,
          session_title: null,
          session_directory: null,
          entries: [],
        }}
      />,
    );

    expect(screen.getByText(/OpenCode session has not been captured/)).toBeInTheDocument();
  });

  it("renders stored session entries", () => {
    const history: SessionHistoryData = {
      session_id: "ses_123",
      session_title: "Demo session",
      session_directory: "/tmp/project",
      entries: [
        {
          id: "entry_1",
          message_id: "msg_1",
          role: "user",
          kind: "text",
          text: "Tune this shape.",
          tool: null,
          status: null,
          timestamp: "2026-04-04T10:00:00Z",
        },
        {
          id: "entry_2",
          message_id: "msg_2",
          role: "assistant",
          kind: "tool",
          text: "Check FSM state\ntool=bash\nstatus=completed",
          tool: "bash",
          status: "completed",
          timestamp: "2026-04-04T10:00:01Z",
        },
      ],
    };

    render(<SessionHistory history={history} />);

    expect(screen.getByText("Demo session")).toBeInTheDocument();
    expect(screen.getByText("Tune this shape.")).toBeInTheDocument();
    expect(screen.getAllByText(/tool=bash/)).toHaveLength(2);
  });
});