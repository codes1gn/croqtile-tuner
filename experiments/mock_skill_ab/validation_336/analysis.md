# Mock CroqTuner Skill A/B

A = strict migrated contract.
B = control legacy/partial-load model.

## Configuration

- DSLs: croqtile, cuda, cute, triton, tilelang, helion, cutile
- Runs per DSL and variant: 24
- Round target: 336
- Shape length: 30
- Seed: 20260409

## Strict Variant

| DSL | Reached target | Runs | Pass rate | Avg rounds | P95 rounds | Store miss runs | Cross-DSL runs | Attempt-only runs | Research runs | Resume-state fails | Resume-validation fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| croqtile | 24 | 24 | 100.00% | 336.0 | 336.0 | 0 | 0 | 0 | 7 | 0 | 0 |
| cuda | 24 | 24 | 100.00% | 336.0 | 336.0 | 0 | 0 | 3 | 21 | 0 | 0 |
| cute | 24 | 24 | 100.00% | 336.0 | 336.0 | 0 | 0 | 2 | 21 | 0 | 0 |
| cutile | 24 | 24 | 100.00% | 336.0 | 336.0 | 0 | 0 | 2 | 20 | 0 | 0 |
| helion | 24 | 24 | 100.00% | 336.0 | 336.0 | 0 | 0 | 1 | 23 | 0 | 0 |
| tilelang | 24 | 24 | 100.00% | 336.0 | 336.0 | 0 | 0 | 1 | 22 | 0 | 0 |
| triton | 24 | 24 | 100.00% | 336.0 | 336.0 | 0 | 0 | 1 | 22 | 0 | 0 |

## Control Variant

| DSL | Reached target | Runs | Pass rate | Avg rounds | P95 rounds | Store miss runs | Cross-DSL runs | Attempt-only runs | Research runs | Resume-state fails | Resume-validation fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| croqtile | 0 | 24 | 0.00% | 15.58 | 59.35 | 15 | 15 | 0 | 0 | 0 | 0 |
| cuda | 0 | 24 | 0.00% | 27.0 | 64.0 | 19 | 8 | 0 | 0 | 0 | 1 |
| cute | 0 | 24 | 0.00% | 22.21 | 47.2 | 15 | 11 | 0 | 0 | 0 | 0 |
| cutile | 0 | 24 | 0.00% | 23.46 | 60.0 | 18 | 13 | 0 | 0 | 0 | 0 |
| helion | 0 | 24 | 0.00% | 29.46 | 64.0 | 16 | 12 | 0 | 0 | 0 | 0 |
| tilelang | 0 | 24 | 0.00% | 26.54 | 64.0 | 17 | 11 | 0 | 0 | 1 | 1 |
| triton | 0 | 24 | 0.00% | 26.25 | 82.1 | 18 | 8 | 0 | 0 | 0 | 0 |

## CroqTile Path Audit

| Variant | pure_co | co___cpp__ | generated_cu | cross_dsl_violation |
| --- | --- | --- | --- | --- |
| strict | 5500 | 1933 | 655 | 0 |
| control | 104 | 81 | 188 | 25 |

## Example Failing Runs

| Variant | DSL | Run | Stop reason | Rounds completed |
| --- | --- | --- | --- | --- |
| control | croqtile | run_001 | cross_dsl_contamination | 6 |
| control | croqtile | run_002 | missing_store_step | 69 |
| control | croqtile | run_003 | cross_dsl_contamination | 5 |
| control | croqtile | run_004 | missing_store_step | 4 |
| control | croqtile | run_005 | cross_dsl_contamination | 8 |
| control | croqtile | run_006 | missing_store_step | 29 |
| control | croqtile | run_007 | missing_store_step | 4 |
| control | croqtile | run_008 | missing_store_step | 8 |
| control | croqtile | run_009 | cross_dsl_contamination | 7 |
| control | croqtile | run_010 | missing_store_step | 33 |
| control | croqtile | run_011 | missing_store_step | 4 |
| control | croqtile | run_012 | missing_store_step | 14 |

## Notes

- All raw traces are preserved as `trace.jsonl` per run.
- Every run also writes `summary.json` beside the trace.
- Strict traces include an explicit `iter000` baseline record plus attempt-only compile-fail records that do not consume public iteration ids.
- Strict ideation records when discard streaks escalate to research-backed exploration, and full profiling now reappears after wins or stalls.
- Resume checkpoints now model raw-history, markdown-history, compaction-summary, and active-state availability so 300+ iteration continuity can be stress-tested explicitly.
- The control variant intentionally models stale-skill failure modes such as shape-boundary stops, missing STORE, and cross-DSL contamination.
