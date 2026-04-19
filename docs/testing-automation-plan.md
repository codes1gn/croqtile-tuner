# Automated Testing Plan (Safe / Mocked)

This project now has a test path that is explicitly isolated from real tuning runs and paid agent calls.

## Principles

- Never execute real `opencode` or `cursor-agent` during tests.
- Never read/write your live `tuning/` tree during tests.
- Keep backend tests deterministic with in-memory DB sessions and mocked process/agent detection.
- Cover frontend UX flows with component-level automation (Testing Library + Vitest).
- Add fuzz-style checks around historically fragile logic (baseline classification, dtype normalization).

## What Is Covered

### Backend

- Baseline row classification logic in `artifact_scanner`.
- Randomized normalization checks for dtype parser behavior.
- Artifact scan behavior for `cancelled` tasks (ensures they are still indexed in DB).
- Agent-state synchronization FSM rules:
  - only `pending` can auto-transition to `running`
  - `waiting/completed/cancelled` are not auto-woken
  - stale `running` transitions to `waiting`
- Command dispatch safety:
  - verifies `proxychains4` prefix behavior in `run_task` with mocked subprocess launch.

### Frontend

- Existing component tests (`TaskList`, `AddTaskForm`, `StatusBadge`, `LiveLog`, `SessionHistory`).
- New resume UX regression test for `TaskDetail`:
  - resume dialog opens
  - calls resume API with selected iteration
  - local task status updates
  - resume panel closes.
- Fuzz-style chart tests for baseline filtering to prevent staircase regressions.

## Safe Execution

Run this command from repo root:

`monitor/scripts/test-safe.sh`

It automatically:

- creates a temporary isolated tuning directory
- uses a temporary isolated monitor DB path
- forces mock mode
- runs backend pytest + frontend vitest
- cleans up temp artifacts on exit

For flakiness observation, run repeated safe cycles:

`monitor/scripts/test-burn.sh 5`

## Notes

- In this environment, `npm` itself is currently broken due missing global `semver` module, so frontend tests are executed via local `./node_modules/.bin/vitest` in scripts/tests.
- The safe test suite is designed to avoid interference with active tuning sessions.
