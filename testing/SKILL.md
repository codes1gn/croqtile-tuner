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

===============================================
 Results: 2 passed, 0 failed
===============================================
All tests passed.
```

Exit code 0 = all pass. Exit code 1 = one or more failed.

## Individual Test Scripts

```bash
# Harness unit tests
bash testing/harness/test_next_iter.sh       # 11 tests for next_iter.sh
bash testing/harness/test_store_round.sh     # 19 tests for store_round.sh
bash testing/harness/test_profile_extract.sh # 14 tests for profile_extract.sh

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
