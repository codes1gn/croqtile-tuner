# Testing Mocks

Drop-in shims for `ncu` and web search that let subagent tests exercise the full
`croq-tune` PROFILE → IDEA loop without a real GPU or network.

## Tools

### `mock_ncu` — ncu shim

Replaces `ncu` in PATH. Produces synthetic profile data for any of 5 bottleneck scenarios.

**Install for a test session:**
```bash
export PATH="$(git rev-parse --show-toplevel)/testing/mocks:$PATH"
export MOCK_NCU_SCENARIO=memory_bound   # or compute_bound / launch_bound / latency_bound / mixed
```

**Profile mode** (write .ncu-rep stub):
```bash
ncu --set full \
    --export tuning/.../ncu_iter001.ncu-rep \
    --force-overwrite \
    ./my_kernel
```

**CSV export mode** (read stub, emit synthetic CSV):
```bash
ncu --import tuning/.../ncu_iter001.ncu-rep \
    --csv --page raw \
    > tuning/.../ncu_iter001.csv
```

**Scenarios:**

| `MOCK_NCU_SCENARIO` | DRAM% | SM% | Occ% | Stall% | → bottleneck |
|---|---|---|---|---|---|
| `memory_bound`  | 92 | 32 | 65 | 15 | `memory_bound`  |
| `compute_bound` | 28 | 88 | 72 | 10 | `compute_bound` |
| `launch_bound`  | 18 | 14 |  8 |  5 | `launch_bound`  |
| `latency_bound` | 45 | 35 | 50 | 58 | `latency_bound` |
| `mixed`         | 55 | 52 | 48 | 22 | `memory_bound` (DRAM wins) |

The CSV output is processed correctly by `.claude/skills/croq-tune/tools/profile_extract.sh`.

---

### `mock_websearch` — web search shim

Returns canned GPU optimization references without hitting the real internet.

**Usage:**
```bash
# Auto-detect topic from query keywords
python3 testing/mocks/mock_websearch "CUDA shared memory bank conflict matmul bf16"

# Explicit topic
python3 testing/mocks/mock_websearch --topic memory_bound
```

**Output:** JSON array of `{title, url, snippet}` objects.

**Topics:**

| Topic | Triggered by keywords | Content |
|---|---|---|
| `memory_bound`  | memory, dram, bandwidth, coalesce | async copy, vectorized loads, TMA |
| `compute_bound` | compute, tensor core, mma, register | MMA shapes, occupancy, ILP |
| `launch_bound`  | launch, occupancy, wave, persistent | thread block sizing, stream-K |
| `latency_bound` | async, pipeline, prefetch, stall, tma | double buffering, mbarrier |
| `bank_conflict` | bank, conflict, pad | smem padding, swizzled layouts |
| `swizzle`       | swizzle, rasteriz, l2, locality | threadblock swizzle, tile rasterization |
| `generic`       | (fallback) | general GEMM tuning, ncu metric reference |

---

## Running the Mock Tests

```bash
bash testing/harness/test_mock_tools.sh
```

Or via the full test suite:
```bash
bash testing/run_tests.sh
```

Expected: 33 tests, all `ok`.

---

## Integration in Subagent Tests

For an end-to-end subagent test that exercises PROFILE→IDEA without a GPU:

```bash
#!/usr/bin/env bash
# Prepend mocks to PATH so ncu resolves to mock_ncu
REPO=$(git rev-parse --show-toplevel)
export PATH="$REPO/testing/mocks:$PATH"
export MOCK_NCU_SCENARIO=memory_bound

# Now run the real profile_extract harness — it will call mock_ncu transparently
# via the profile step scripts
bash .claude/skills/croq-tune/tools/profile_extract.sh \
    --csv tuning/<gpu>/cuda/perf/mykey/ncu_iter001.csv \
    --iter iter001_test
```

For web search, call `mock_websearch` as a CLI in test scaffolding or set
`MOCK_WEBSEARCH=1` in environment — the SKILL.md IDEA step instructs the agent
to use the `WebSearch` tool, which is a native agent tool and cannot be shimmed
at the shell level. For subagent test validation, assert that the agent's
IDEA transcript contains a `WebSearch` tool call by grepping session markers.
