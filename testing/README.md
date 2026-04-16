# Croq-Tune Testing Framework

## Structure

```
testing/
├── harness/          — Unit tests for each shell harness script
│   ├── test_next_iter.sh
│   ├── test_store_round.sh
│   └── run_all.sh
├── e2e/              — End-to-end session behaviour tests
│   ├── test_mini_session.sh
│   └── grep_session_markers.sh
├── fixtures/         — Fake tuning artifacts for tests (no GPU needed)
│   ├── cuda/matmul_bf16fp32_8x16x16/
│   └── ...
└── README.md
```

## Running Tests

```bash
# All harness unit tests
bash testing/harness/run_all.sh

# Individual harness test
bash testing/harness/test_next_iter.sh
bash testing/harness/test_store_round.sh

# E2E session marker grep (after running a session)
bash testing/e2e/grep_session_markers.sh <session_jsonl_path>
```

## Philosophy

Tests operate against a **fixture workspace** under `testing/fixtures/`.
No real GPU, no real nvcc. The harness scripts are pure bash and work
against any directory structure that mimics the tuning artifact layout.

The e2e grep tests check that key behavioural markers appeared in an
agent session transcript:
- `[store_round] STORE complete` — storage harness was called and succeeded
- `next_iter.sh` — naming harness was called
- `VERIFY: PASS` — correctness check passed before measurement
- `TFLOPS:` — benchmark produced output
- `ncu` — profiling was run

## Adding a New Test

For a harness unit test, add a new `test_<script>.sh` in `testing/harness/`
following the TAP (Test Anything Protocol) format:

```bash
#!/usr/bin/env bash
# test_my_harness.sh
set -euo pipefail
source "$(dirname "$0")/tap_helpers.sh"

plan 3

result=$(bash .claude/skills/croq-tune/tools/my_harness.sh --arg value 2>&1)
ok "exit 0" $?
like "output contains expected" "$result" "expected_string"
unlike "output does not contain error" "$result" "ERROR"
```
