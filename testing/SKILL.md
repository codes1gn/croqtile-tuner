---
name: croq-test
description: Run the croq-tune harness test suite. Load when asked to test, verify, or check that harness scripts work correctly. Covers next_iter.sh, store_round.sh, and session marker checks.
---

# Croq-Test — Harness Test Runner

## Quick Start (subagent / agent)

Run all tests from the repo root:

```bash
bash testing/run_tests.sh
```

Expected output:
```
=== test_next_iter ===
1..10
ok 1 - next iter after 009 is iter010_mytest
...
  PASSED: test_next_iter

=== test_store_round ===
1..18
ok 1 - store exit code 0
...
  PASSED: test_store_round

=== test_mock_tools ===
1..30
ok 1 - mock_ncu --version prints mock_ncu
...
  PASSED: test_mock_tools

===============================================
 Results: 6 passed, 0 failed
===============================================
All tests passed.
```

Exit code 0 = all pass. Exit code 1 = one or more failed.

## Quick Start — Mock Environment (no GPU, no nvcc)

To run a tuning session or e2e test **without a real GPU**, inject mock tools into PATH first:

```bash
# From repo root — add mock tools to PATH
export PATH="$(git rev-parse --show-toplevel)/testing/mocks:$PATH"

# Choose the bottleneck scenario for the mock ncu profiler
export MOCK_NCU_SCENARIO=memory_bound  # memory_bound | compute_bound | launch_bound | latency_bound | mixed

# Optional: control mock kernel output
export MOCK_KERNEL_TFLOPS=28.5
export MOCK_KERNEL_TIME_MS=120.3

# Verify mock_ncu is active (should print "mock_ncu 1.0.0")
ncu --version
```

With this in place, `ncu`, `mock_websearch`, and `mock_kernel` are all available.
The `profile_extract.sh` and `store_round.sh` harnesses will work correctly against mock output.

### Running the e2e mock-only test (no agent, no GPU):

```bash
bash testing/e2e/test_mini_session.sh --mock-only
```

This simulates one complete tuning round (PROFILE→IDEA→IMPLEMENT→VERIFY→MEASURE→DECIDE→STORE)
on the filesystem, without launching an agent. All 14 checks should pass.

### Running a real e2e session validation (requires a completed agent run):

```bash
# 1. Start a tuning session with mock tools in PATH
export PATH="$(git rev-parse --show-toplevel)/testing/mocks:$PATH"
export MOCK_NCU_SCENARIO=memory_bound
/croq-tune cuda bf16fp32 matmul_bf16fp32_512x16384x16384

# 2. After the session completes ≥1 round, validate the transcript
bash testing/e2e/test_mini_session.sh \
    --session ~/.cursor/projects/.../agent-transcripts/<uuid>.jsonl \
    --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384
```

This runs 20 transcript marker checks: ncu called, WebSearch called, checkpoint written,
VERIFY: PASS present, TFLOPS measured, store_round.sh executed, no bare filenames, etc.

## Individual Test Scripts

```bash
# Harness unit tests
bash testing/harness/test_next_iter.sh       # 11 tests for next_iter.sh
bash testing/harness/test_store_round.sh     # 19 tests for store_round.sh
bash testing/harness/test_profile_extract.sh # 14 tests for profile_extract.sh
bash testing/harness/test_mock_tools.sh      # 30 tests for mock_ncu + mock_websearch

# E2E tests (mock-only mode — no agent, no GPU, validates filesystem artifacts)
bash testing/e2e/test_mini_session.sh --mock-only  # 14 tests for one simulated round

# E2E Mode 1: session transcript markers (requires a session JSONL file)
bash testing/e2e/grep_session_markers.sh \
    --session ~/.cursor/projects/.../agent-transcripts/<uuid>/<uuid>.jsonl

# E2E Mode 2: filesystem artifact checks (authentic criteria)
bash testing/e2e/grep_session_markers.sh \
    --fs --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384

# E2E Both modes together
bash testing/e2e/grep_session_markers.sh \
    --session <uuid>.jsonl \
    --fs --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384
```

## Mock Tools for Subagent Tests (No GPU Required)

The `testing/mocks/` directory contains drop-in shims for `ncu` and web search.
See `testing/mocks/README.md` for full documentation.

**Quick setup for a mock session:**
```bash
export PATH="$(git rev-parse --show-toplevel)/testing/mocks:$PATH"
export MOCK_NCU_SCENARIO=memory_bound   # or: compute_bound / launch_bound / latency_bound / mixed
```

| Tool | What it replaces | Scenarios |
|---|---|---|
| `mock_ncu` | real `ncu` binary | 5 bottleneck scenarios → compatible CSV for `profile_extract.sh` |
| `mock_websearch` | agent `WebSearch` tool | 7 topics → realistic JSON results for IDEA step testing |

Both tools are tested by `testing/harness/test_mock_tools.sh` (30 tests).

---

## What the Tests Cover

### `test_next_iter.sh` (11 tests)

| Test | What it checks |
|---|---|
| next iter after 009 | Reads real artifact files, returns iter010_tag |
| empty workspace | Returns iter001_tag when no artifacts exist |
| attempt mode | Returns attempt0001_tag |
| uppercase tag | Rejected with ERROR |
| 1-char tag | Rejected with ERROR |
| tag with space | Rejected with ERROR |
| missing --tag | Rejected with ERROR |
| missing --dsl | Rejected with ERROR |
| clean output | No ERROR in stdout |
| output format | Matches `iter[0-9]{3}_tag` pattern |
| malformed JSON | Script does not crash; skips bad lines |

### `test_store_round.sh` (19 tests)

| Test | What it checks |
|---|---|
| exit code 0 | Successful store exits 0 |
| success message | `[store_round] STORE complete` printed |
| rounds.raw.jsonl created | File exists after store |
| rounds.raw has iter | Correct iter in file |
| rounds.md created | File exists after store |
| rounds.md has iter | Markdown section written |
| idea-log.jsonl created | File exists after store |
| idea-log has round N | Correct round number |
| results.tsv created | File exists after store |
| results.tsv header | Header line present |
| results.tsv iter row | Correct iter row |
| 2 stores = 2 lines | Appends, not overwrites |
| header + 2 data rows | Correct TSV line count |
| missing kernel | Rejected with ERROR |
| bare-number kernel | `iter008` without `_tag` rejected |
| invalid decision | Non-standard DECISION value rejected |
| 2-digit iter | `iter09` rejected (must be 3-digit) |
| rounds.raw is valid JSON | All lines parse as JSON |
| idea-log is valid JSON | All lines parse as JSON |

### `test_mock_tools.sh` (30 tests)

Tests for mock_ncu and mock_websearch (see `testing/mocks/`):

| Test | What it checks |
|---|---|
| mock_ncu --version | Prints version string |
| Profile mode | Writes .ncu-rep sentinel file |
| CSV mode | Emits correct column format |
| 5 scenarios | memory/compute/launch/latency/mixed → correct bottleneck via profile_extract |
| Error handling | Missing .ncu-rep, unknown scenario |
| mock_websearch query | Returns valid JSON array |
| Keyword detection | compute/memory/bank/swizzle queries → correct topic |
| Explicit --topic | All 7 topics return ≥1 result |

### E2E: `test_mini_session.sh --mock-only` (14 tests)

Simulates one complete tuning round using mock tools without a real agent or GPU:

| Test | What it checks |
|---|---|
| iter001_draft.cu | Source stub present in srcs/ |
| iter000_baseline.cu | Baseline stub present |
| .ncu-rep sentinel | mock_ncu profile mode creates file |
| ncu CSV | mock_ncu csv mode creates readable CSV |
| dram/sm metrics | CSV has required columns for profile_extract |
| profile_extract | Correctly classifies memory_bound from mock CSV |
| Timing file | Has TFLOPS: and VERIFY: PASS |
| rounds.raw.jsonl | ≥2 entries after one round |
| results.tsv | ≥2 data rows |
| Tag naming | No bare iter*.cu files |
| Checkpoint | Status is VERIFIED or PROFILE |

**For real e2e session validation** (with a real agent run):
```bash
bash testing/e2e/test_mini_session.sh \
    --session ~/.cursor/projects/.../agent-transcripts/<uuid>.jsonl \
    --dsl cuda --shape-key matmul_bf16fp32_512x16384x16384
```
This asserts the transcript has all 20 protocol markers (ncu, WebSearch, checkpoint, VERIFY: PASS, TFLOPS, store_round, etc.).

---

## Adding a New Test

Create `testing/harness/test_<feature>.sh` following the TAP pattern:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source testing/harness/tap_helpers.sh

plan <N>

# ... test assertions using: ok, is, like, unlike ...

done_testing
```

The `run_all.sh` script auto-discovers all `test_*.sh` files in `testing/harness/`.
