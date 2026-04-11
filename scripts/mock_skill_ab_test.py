#!/usr/bin/env python3
"""Run a deterministic mock A/B stress test for CroqTuner skill loops.

The harness compares:
- strict: the migrated per-DSL FSM contract with mandatory skill loading
- control: a legacy/partial-load model that can early-stop or skip persistence

Each run simulates long tuning sessions with mock profile/compile/benchmark data and
stores every run's trace plus aggregated markdown analysis for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


ALL_DSLS = ["croqtile", "cuda", "cute", "triton", "tilelang", "helion", "cutile"]
BOTTLENECKS = [
    "smem_throughput",
    "l2_throughput",
    "dram_throughput",
    "compute_bound",
    "latency_bound",
    "occupancy_limited",
]
IDEA_CATEGORIES = ["macro", "structural", "choreo", "ncu_micro"]

BASELINE_RANGES = {
    "croqtile": (700.0, 790.0),
    "cuda": (650.0, 760.0),
    "cute": (620.0, 735.0),
    "triton": (590.0, 700.0),
    "tilelang": (575.0, 690.0),
    "helion": (565.0, 680.0),
    "cutile": (600.0, 710.0),
}


@dataclass
class IterationRecord:
    round_id: int
    public_iter_id: int | None
    attempt_id: int | None
    record_kind: str
    shape_index: int
    bottleneck: str
    profile_source: str
    idea_category: str
    implementation_path: str
    fallback_reason: str | None
    compile_attempts: int
    compile_succeeded: bool
    measured_tflops: float | None
    previous_best_tflops: float
    best_tflops_after_round: float
    decision: str
    store_expected: bool
    store_completed: bool
    round_memory_saved: bool
    round_memory_raw_saved: bool
    round_memory_md_saved: bool
    compaction_summary_updated: bool
    resume_state_present: bool
    resume_source_validated: bool
    cross_dsl_violation: bool
    suspicious_measurement: bool
    research_basis: bool


@dataclass
class RunSummary:
    variant: str
    dsl: str
    run_id: str
    seed: int
    rounds_target: int
    rounds_completed: int
    shape_switches: int
    baseline_tflops: float
    best_tflops: float
    keep_count: int
    discard_count: int
    compile_fail_count: int
    store_miss_count: int
    round_memory_miss_count: int
    round_memory_raw_miss_count: int
    round_memory_md_miss_count: int
    compaction_summary_miss_count: int
    cross_dsl_violation_count: int
    attempt_only_count: int
    research_escalation_count: int
    resume_state_fail_count: int
    resume_validation_fail_count: int
    croqtile_paths: Dict[str, int]
    reached_target: bool
    stop_reason: str


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mock A/B stress tests for CroqTuner skills.")
    parser.add_argument("--dsls", default=",".join(ALL_DSLS), help="Comma-separated DSL list.")
    parser.add_argument("--agents-per-dsl", type=int, default=32, help="Mock runs per DSL and variant.")
    parser.add_argument("--round-target", type=int, default=220, help="Target rounds per run.")
    parser.add_argument("--shape-length", type=int, default=30, help="Rounds per shape before NEXT_SHAPE.")
    parser.add_argument("--seed", type=int, default=20260409, help="Master RNG seed.")
    parser.add_argument(
        "--output-root",
        default="experiments/mock_skill_ab",
        help="Directory under which run artifacts are written.",
    )
    parser.add_argument("--run-name", default=None, help="Optional fixed run directory name.")
    return parser.parse_args()


def choose_idea_category(
    variant: str,
    round_id: int,
    discard_streak: int,
    last_categories: List[str],
    rng: random.Random,
) -> tuple[str, bool]:
    research_basis = False
    if variant == "strict":
        research_basis = discard_streak >= 5 or (round_id > 40 and discard_streak >= 3)
        if discard_streak >= 5:
            pool = ["structural", "choreo", "ncu_micro"]
        elif discard_streak >= 3:
            pool = ["structural", "ncu_micro", "choreo"]
        elif round_id <= 10:
            pool = ["macro", "structural"]
        elif round_id <= 20:
            pool = ["structural", "choreo", "macro"]
        else:
            pool = ["ncu_micro", "choreo", "structural"]
        if len(last_categories) >= 2 and last_categories[-1] == last_categories[-2]:
            pool = [category for category in pool if category != last_categories[-1]] or pool
        return rng.choice(pool), research_basis

    if discard_streak >= 4:
        return rng.choice(IDEA_CATEGORIES), research_basis
    if last_categories and rng.random() < 0.45:
        return last_categories[-1], research_basis
    return rng.choice(IDEA_CATEGORIES), research_basis


def choose_profile_source(
    variant: str,
    public_iter_id: int,
    previous_public_decision: str,
    discard_streak: int,
    rng: random.Random,
) -> str:
    if variant == "strict":
        if public_iter_id == 1:
            return "lightweight_trace"
        if previous_public_decision == "KEEP" or discard_streak >= 3:
            return "ncu_full"
        return "lightweight_trace"
    return "ncu" if rng.random() < 0.35 else "weak_signal"


def choose_croqtile_path(variant: str, rng: random.Random) -> tuple[str, str | None]:
    if variant == "strict":
        if rng.random() < 0.68:
            return "pure_co", None
        if rng.random() < 0.75:
            return "co___cpp__", None
        return "generated_cu", "co_or_inline_cpp_insufficient"

    roll = rng.random()
    if roll < 0.45:
        return "generated_cu", "jumped_directly_to_cuda"
    if roll < 0.70:
        return "pure_co", None
    if roll < 0.90:
        return "co___cpp__", None
    return "cross_dsl_violation", "used_non_croqtile_language"


def choose_implementation_path(dsl: str, variant: str, rng: random.Random) -> tuple[str, str | None, bool]:
    if dsl == "croqtile":
        path_name, fallback_reason = choose_croqtile_path(variant, rng)
        return path_name, fallback_reason, path_name == "cross_dsl_violation"

    if variant == "control" and rng.random() < 0.025:
        other = rng.choice([candidate for candidate in ALL_DSLS if candidate != dsl])
        return other, "used_wrong_dsl", True
    return dsl, None, False


def compile_success(variant: str, rng: random.Random) -> tuple[int, bool]:
    max_attempts = 3 if variant == "strict" else 2
    success_prob = 0.94 if variant == "strict" else 0.88
    for attempt in range(1, max_attempts + 1):
        if rng.random() < success_prob:
            return attempt, True
    return max_attempts, False


def improvement_delta(
    variant: str,
    dsl: str,
    category: str,
    bottleneck: str,
    implementation_path: str,
    profile_source: str,
    research_basis: bool,
    discard_streak: int,
    cross_dsl_violation: bool,
    rng: random.Random,
) -> float:
    base_shift = rng.gauss(5.5 if variant == "strict" else 1.0, 9.0 if variant == "strict" else 12.0)
    category_bonus = {
        "macro": 0.8,
        "structural": 2.5,
        "choreo": 3.0 if dsl == "croqtile" else 0.4,
        "ncu_micro": 1.8,
    }[category]
    bottleneck_bonus = {
        "smem_throughput": 2.0 if category in {"structural", "ncu_micro"} else -1.5,
        "l2_throughput": 1.8 if category in {"macro", "structural"} else -1.0,
        "dram_throughput": 1.3 if category in {"structural", "choreo"} else -0.7,
        "compute_bound": 1.4 if category in {"choreo", "ncu_micro"} else -0.8,
        "latency_bound": 1.6 if category in {"ncu_micro", "structural"} else -1.0,
        "occupancy_limited": 1.7 if category in {"macro", "structural"} else -0.9,
    }[bottleneck]

    impl_bonus = 0.0
    if dsl == "croqtile":
        impl_bonus = {
            "pure_co": 2.0 if variant == "strict" else 0.5,
            "co___cpp__": 1.0,
            "generated_cu": -0.2 if variant == "strict" else -2.0,
            "cross_dsl_violation": -6.0,
        }[implementation_path]
    elif cross_dsl_violation:
        impl_bonus = -7.0

    discard_penalty = -min(discard_streak * (0.3 if variant == "strict" else 0.6), 4.0)
    profile_bonus = 1.8 if profile_source == "ncu_full" else 0.4 if profile_source == "lightweight_trace" else 0.0
    research_bonus = 1.2 if research_basis else 0.0
    return base_shift + category_bonus + bottleneck_bonus + impl_bonus + discard_penalty + profile_bonus + research_bonus


def run_session(
    variant: str,
    dsl: str,
    run_index: int,
    round_target: int,
    shape_length: int,
    run_seed: int,
    output_dir: Path,
) -> RunSummary:
    rng = random.Random(run_seed)
    baseline_low, baseline_high = BASELINE_RANGES[dsl]
    baseline = round(rng.uniform(baseline_low, baseline_high), 2)
    current_best = baseline
    last_categories: List[str] = []
    croqtile_paths = Counter()
    records: List[IterationRecord] = []
    discard_streak = 0
    keep_count = 0
    discard_count = 0
    compile_fail_count = 0
    attempt_only_count = 0
    research_escalation_count = 0
    store_miss_count = 0
    round_memory_miss_count = 0
    round_memory_raw_miss_count = 0
    round_memory_md_miss_count = 0
    compaction_summary_miss_count = 0
    cross_dsl_violations = 0
    stop_reason = "target_reached"
    rounds_completed = 0
    resume_state_fail_count = 0
    resume_validation_fail_count = 0

    baseline_path = "pure_co" if dsl == "croqtile" else dsl
    if dsl == "croqtile":
        croqtile_paths[baseline_path] += 1
    records.append(
        IterationRecord(
            round_id=0,
            public_iter_id=0,
            attempt_id=None,
            record_kind="baseline",
            shape_index=0,
            bottleneck="baseline_setup",
            profile_source="lightweight_trace",
            idea_category="iter000_scalar",
            implementation_path=baseline_path,
            fallback_reason=None,
            compile_attempts=1,
            compile_succeeded=True,
            measured_tflops=round(baseline, 2),
            previous_best_tflops=round(baseline, 2),
            best_tflops_after_round=round(baseline, 2),
            decision="BASELINE",
            store_expected=True,
            store_completed=True,
            round_memory_saved=True,
            round_memory_raw_saved=True,
            round_memory_md_saved=True,
            compaction_summary_updated=True,
            resume_state_present=True,
            resume_source_validated=True,
            cross_dsl_violation=False,
            suspicious_measurement=False,
            research_basis=False,
        )
    )

    event_id = 0
    public_iter_id = 1
    attempt_id = 0
    previous_public_decision = "BASELINE"

    while public_iter_id <= round_target:
        shape_index = ((public_iter_id - 1) // shape_length) + 1
        if variant == "control" and public_iter_id > 1 and (public_iter_id - 1) % shape_length == 0 and rng.random() < 0.18:
            stop_reason = "shape_boundary_stop"
            break

        previous_best = current_best
        bottleneck = rng.choice(BOTTLENECKS)
        profile_source = choose_profile_source(variant, public_iter_id, previous_public_decision, discard_streak, rng)
        category, research_basis = choose_idea_category(variant, public_iter_id, discard_streak, last_categories, rng)
        implementation_path, fallback_reason, cross_dsl_violation = choose_implementation_path(dsl, variant, rng)
        compile_attempts, compile_ok = compile_success(variant, rng)

        if research_basis:
            research_escalation_count += 1

        if variant == "strict" and not compile_ok:
            event_id += 1
            attempt_id += 1
            compile_fail_count += 1
            attempt_only_count += 1
            discard_count += 1
            discard_streak += 1
            if cross_dsl_violation:
                cross_dsl_violations += 1
            if dsl == "croqtile":
                croqtile_paths[implementation_path] += 1
            records.append(
                IterationRecord(
                    round_id=event_id,
                    public_iter_id=None,
                    attempt_id=attempt_id,
                    record_kind="attempt",
                    shape_index=shape_index,
                    bottleneck=bottleneck,
                    profile_source=profile_source,
                    idea_category=category,
                    implementation_path=implementation_path,
                    fallback_reason=fallback_reason,
                    compile_attempts=compile_attempts,
                    compile_succeeded=False,
                    measured_tflops=None,
                    previous_best_tflops=round(previous_best, 2),
                    best_tflops_after_round=round(current_best, 2),
                    decision="DISCARD_COMPILE_FAIL",
                    store_expected=True,
                    store_completed=True,
                    round_memory_saved=True,
                    round_memory_raw_saved=True,
                    round_memory_md_saved=True,
                    compaction_summary_updated=True,
                    resume_state_present=True,
                    resume_source_validated=True,
                    cross_dsl_violation=cross_dsl_violation,
                    suspicious_measurement=False,
                    research_basis=research_basis,
                )
            )
            last_categories.append(category)
            last_categories = last_categories[-3:]
            continue

        measured = None
        suspicious = False
        if compile_ok:
            delta = improvement_delta(
                variant,
                dsl,
                category,
                bottleneck,
                implementation_path,
                profile_source,
                research_basis,
                discard_streak,
                cross_dsl_violation,
                rng,
            )
            measured = round(max(current_best + delta, baseline * 0.55), 2)
            suspicious = measured > current_best * 1.5 or measured < current_best * 0.5
            if suspicious and variant == "strict":
                measured = round((measured + current_best + rng.uniform(-3.0, 3.0)) / 2.0, 2)
                suspicious = False

            if measured > current_best + 0.75:
                decision = "KEEP"
                current_best = measured
                keep_count += 1
                discard_streak = 0
            else:
                decision = "DISCARD"
                discard_count += 1
                discard_streak += 1
        else:
            decision = "DISCARD_COMPILE_FAIL"
            compile_fail_count += 1
            discard_count += 1
            discard_streak += 1

        store_completed = True
        round_memory_saved = True
        round_memory_raw_saved = True
        round_memory_md_saved = True
        compaction_summary_updated = True
        resume_state_present = True
        resume_source_validated = True
        if variant == "control":
            if decision != "KEEP" and rng.random() < 0.06:
                store_completed = False
                store_miss_count += 1
            if rng.random() < 0.08:
                round_memory_raw_saved = False
                round_memory_saved = False
                round_memory_miss_count += 1
                round_memory_raw_miss_count += 1
            if rng.random() < 0.08:
                round_memory_md_saved = False
                round_memory_saved = False
                round_memory_miss_count += 1
                round_memory_md_miss_count += 1
            if rng.random() < 0.04:
                compaction_summary_updated = False
                compaction_summary_miss_count += 1

        if cross_dsl_violation:
            cross_dsl_violations += 1

        if dsl == "croqtile":
            croqtile_paths[implementation_path] += 1

        event_id += 1
        record = IterationRecord(
            round_id=event_id,
            public_iter_id=public_iter_id,
            attempt_id=None,
            record_kind="public_iteration",
            shape_index=shape_index,
            bottleneck=bottleneck,
            profile_source=profile_source,
            idea_category=category,
            implementation_path=implementation_path,
            fallback_reason=fallback_reason,
            compile_attempts=compile_attempts,
            compile_succeeded=compile_ok,
            measured_tflops=measured,
            previous_best_tflops=round(previous_best, 2),
            best_tflops_after_round=round(current_best, 2),
            decision=decision,
            store_expected=True,
            store_completed=store_completed,
            round_memory_saved=round_memory_saved,
            round_memory_raw_saved=round_memory_raw_saved,
            round_memory_md_saved=round_memory_md_saved,
            compaction_summary_updated=compaction_summary_updated,
            resume_state_present=resume_state_present,
            resume_source_validated=resume_source_validated,
            cross_dsl_violation=cross_dsl_violation,
            suspicious_measurement=suspicious,
            research_basis=research_basis,
        )
        records.append(record)
        last_categories.append(category)
        last_categories = last_categories[-3:]
        rounds_completed = public_iter_id
        previous_public_decision = decision

        if variant == "control" and not store_completed:
            stop_reason = "missing_store_step"
            break

        if public_iter_id > 1 and public_iter_id % 64 == 0:
            if variant == "control":
                if rng.random() < 0.07:
                    resume_state_present = False
                    resume_state_fail_count += 1
                if rng.random() < 0.07:
                    resume_source_validated = False
                    resume_validation_fail_count += 1

                records[-1].resume_state_present = resume_state_present
                records[-1].resume_source_validated = resume_source_validated

                if not round_memory_raw_saved:
                    stop_reason = "missing_round_memory_raw"
                    break
                if not round_memory_md_saved:
                    stop_reason = "missing_round_memory_md"
                    break
                if not compaction_summary_updated:
                    stop_reason = "missing_compaction_summary"
                    break
                if not resume_state_present:
                    stop_reason = "missing_active_loop_state"
                    break
                if not resume_source_validated:
                    stop_reason = "invalid_resume_source"
                    break

        if variant == "control" and cross_dsl_violation and rng.random() < 0.3:
            stop_reason = "cross_dsl_contamination"
            break

        public_iter_id += 1

    reached_target = rounds_completed >= round_target and stop_reason == "target_reached"
    if rounds_completed < round_target and stop_reason == "target_reached":
        stop_reason = "unknown_abort"

    run_id = f"run_{run_index:03d}"
    run_dir = output_dir / variant / dsl / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    summary = RunSummary(
        variant=variant,
        dsl=dsl,
        run_id=run_id,
        seed=run_seed,
        rounds_target=round_target,
        rounds_completed=rounds_completed,
        shape_switches=max(rounds_completed - 1, 0) // shape_length,
        baseline_tflops=round(baseline, 2),
        best_tflops=round(current_best, 2),
        keep_count=keep_count,
        discard_count=discard_count,
        compile_fail_count=compile_fail_count,
        store_miss_count=store_miss_count,
        round_memory_miss_count=round_memory_miss_count,
        round_memory_raw_miss_count=round_memory_raw_miss_count,
        round_memory_md_miss_count=round_memory_md_miss_count,
        compaction_summary_miss_count=compaction_summary_miss_count,
        cross_dsl_violation_count=cross_dsl_violations,
        attempt_only_count=attempt_only_count,
        research_escalation_count=research_escalation_count,
        resume_state_fail_count=resume_state_fail_count,
        resume_validation_fail_count=resume_validation_fail_count,
        croqtile_paths=dict(croqtile_paths),
        reached_target=reached_target,
        stop_reason=stop_reason,
    )

    (run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def render_markdown_table(headers: List[str], rows: Iterable[Iterable[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def aggregate(summaries: List[RunSummary]) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    by_key: Dict[tuple[str, str], List[RunSummary]] = defaultdict(list)
    for summary in summaries:
        by_key[(summary.variant, summary.dsl)].append(summary)

    for (variant, dsl), runs in by_key.items():
        rounds = [run.rounds_completed for run in runs]
        bests = [run.best_tflops for run in runs]
        stop_reasons = Counter(run.stop_reason for run in runs)
        croqtile_path_counts = Counter()
        for run in runs:
            croqtile_path_counts.update(run.croqtile_paths)

        grouped[f"{variant}:{dsl}"] = {
            "variant": variant,
            "dsl": dsl,
            "runs": len(runs),
            "reached_target": sum(1 for run in runs if run.reached_target),
            "pass_rate": round(sum(1 for run in runs if run.reached_target) / len(runs), 4),
            "avg_rounds": round(sum(rounds) / len(runs), 2),
            "p50_rounds": round(percentile(rounds, 0.50), 2),
            "p95_rounds": round(percentile(rounds, 0.95), 2),
            "avg_best_tflops": round(sum(bests) / len(runs), 2),
            "store_miss_runs": sum(1 for run in runs if run.store_miss_count > 0),
            "round_memory_miss_runs": sum(1 for run in runs if run.round_memory_miss_count > 0),
            "round_memory_raw_miss_runs": sum(1 for run in runs if run.round_memory_raw_miss_count > 0),
            "round_memory_md_miss_runs": sum(1 for run in runs if run.round_memory_md_miss_count > 0),
            "compaction_summary_miss_runs": sum(1 for run in runs if run.compaction_summary_miss_count > 0),
            "cross_dsl_violation_runs": sum(1 for run in runs if run.cross_dsl_violation_count > 0),
            "attempt_only_runs": sum(1 for run in runs if run.attempt_only_count > 0),
            "research_escalation_runs": sum(1 for run in runs if run.research_escalation_count > 0),
            "resume_state_fail_runs": sum(1 for run in runs if run.resume_state_fail_count > 0),
            "resume_validation_fail_runs": sum(1 for run in runs if run.resume_validation_fail_count > 0),
            "stop_reasons": dict(stop_reasons),
            "croqtile_paths": dict(croqtile_path_counts),
        }

    return grouped


def write_summary_files(output_dir: Path, args: argparse.Namespace, summaries: List[RunSummary]) -> None:
    aggregate_data = aggregate(summaries)
    config = {
        "dsls": [dsl.strip() for dsl in args.dsls.split(",") if dsl.strip()],
        "agents_per_dsl": args.agents_per_dsl,
        "round_target": args.round_target,
        "shape_length": args.shape_length,
        "seed": args.seed,
        "variants": ["strict", "control"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(aggregate_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    headers = [
        "variant",
        "dsl",
        "runs",
        "reached_target",
        "pass_rate",
        "avg_rounds",
        "p50_rounds",
        "p95_rounds",
        "avg_best_tflops",
        "store_miss_runs",
        "round_memory_miss_runs",
        "round_memory_raw_miss_runs",
        "round_memory_md_miss_runs",
        "compaction_summary_miss_runs",
        "cross_dsl_violation_runs",
        "attempt_only_runs",
        "research_escalation_runs",
        "resume_state_fail_runs",
        "resume_validation_fail_runs",
    ]
    rows = []
    for key in sorted(aggregate_data):
        item = aggregate_data[key]
        rows.append([
            item["variant"],
            item["dsl"],
            item["runs"],
            item["reached_target"],
            item["pass_rate"],
            item["avg_rounds"],
            item["p50_rounds"],
            item["p95_rounds"],
            item["avg_best_tflops"],
            item["store_miss_runs"],
            item["round_memory_miss_runs"],
            item["round_memory_raw_miss_runs"],
            item["round_memory_md_miss_runs"],
            item["compaction_summary_miss_runs"],
            item["cross_dsl_violation_runs"],
            item["attempt_only_runs"],
            item["research_escalation_runs"],
            item["resume_state_fail_runs"],
            item["resume_validation_fail_runs"],
        ])

    summary_tsv = ["\t".join(headers)]
    for row in rows:
        summary_tsv.append("\t".join(str(cell) for cell in row))
    (output_dir / "summary.tsv").write_text("\n".join(summary_tsv) + "\n", encoding="utf-8")

    strict_rows = []
    control_rows = []
    for key in sorted(aggregate_data):
        item = aggregate_data[key]
        row = [
            item["dsl"],
            item["reached_target"],
            item["runs"],
            f"{item['pass_rate']:.2%}",
            item["avg_rounds"],
            item["p95_rounds"],
            item["store_miss_runs"],
            item["cross_dsl_violation_runs"],
            item["attempt_only_runs"],
            item["research_escalation_runs"],
            item["resume_state_fail_runs"],
            item["resume_validation_fail_runs"],
        ]
        if item["variant"] == "strict":
            strict_rows.append(row)
        else:
            control_rows.append(row)

    failure_examples = []
    for summary in summaries:
        if not summary.reached_target:
            failure_examples.append((summary.variant, summary.dsl, summary.run_id, summary.stop_reason, summary.rounds_completed))
    failure_examples = sorted(failure_examples)[:12]

    croqtile_strict = aggregate_data.get("strict:croqtile", {}).get("croqtile_paths", {})
    croqtile_control = aggregate_data.get("control:croqtile", {}).get("croqtile_paths", {})

    analysis_lines = [
        "# Mock CroqTuner Skill A/B",
        "",
        "A = strict migrated contract.",
        "B = control legacy/partial-load model.",
        "",
        "## Configuration",
        "",
        f"- DSLs: {', '.join(config['dsls'])}",
        f"- Runs per DSL and variant: {config['agents_per_dsl']}",
        f"- Round target: {config['round_target']}",
        f"- Shape length: {config['shape_length']}",
        f"- Seed: {config['seed']}",
        "",
        "## Strict Variant",
        "",
        render_markdown_table(
            ["DSL", "Reached target", "Runs", "Pass rate", "Avg rounds", "P95 rounds", "Store miss runs", "Cross-DSL runs", "Attempt-only runs", "Research runs", "Resume-state fails", "Resume-validation fails"],
            strict_rows,
        ),
        "",
        "## Control Variant",
        "",
        render_markdown_table(
            ["DSL", "Reached target", "Runs", "Pass rate", "Avg rounds", "P95 rounds", "Store miss runs", "Cross-DSL runs", "Attempt-only runs", "Research runs", "Resume-state fails", "Resume-validation fails"],
            control_rows,
        ),
        "",
        "## CroqTile Path Audit",
        "",
        render_markdown_table(
            ["Variant", "pure_co", "co___cpp__", "generated_cu", "cross_dsl_violation"],
            [
                [
                    "strict",
                    croqtile_strict.get("pure_co", 0),
                    croqtile_strict.get("co___cpp__", 0),
                    croqtile_strict.get("generated_cu", 0),
                    croqtile_strict.get("cross_dsl_violation", 0),
                ],
                [
                    "control",
                    croqtile_control.get("pure_co", 0),
                    croqtile_control.get("co___cpp__", 0),
                    croqtile_control.get("generated_cu", 0),
                    croqtile_control.get("cross_dsl_violation", 0),
                ],
            ],
        ),
        "",
        "## Example Failing Runs",
        "",
    ]

    if failure_examples:
        analysis_lines.append(
            render_markdown_table(
                ["Variant", "DSL", "Run", "Stop reason", "Rounds completed"],
                failure_examples,
            )
        )
    else:
        analysis_lines.append("No failing runs were observed.")

    analysis_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- All raw traces are preserved as `trace.jsonl` per run.",
            "- Every run also writes `summary.json` beside the trace.",
            "- Strict traces include an explicit `iter000` baseline record plus attempt-only compile-fail records that do not consume public iteration ids.",
            "- Strict ideation records when discard streaks escalate to research-backed exploration, and full profiling now reappears after wins or stalls.",
            "- Resume checkpoints now model raw-history, markdown-history, compaction-summary, and active-state availability so 300+ iteration continuity can be stress-tested explicitly.",
            "- The control variant intentionally models stale-skill failure modes such as shape-boundary stops, missing STORE, and cross-DSL contamination.",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dsls = [dsl.strip() for dsl in args.dsls.split(",") if dsl.strip()]
    invalid = [dsl for dsl in dsls if dsl not in ALL_DSLS]
    if invalid:
        raise SystemExit(f"Unsupported DSLs: {', '.join(invalid)}")

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    master_rng = random.Random(args.seed)
    summaries: List[RunSummary] = []
    for variant in ["strict", "control"]:
        for dsl in dsls:
            for run_index in range(1, args.agents_per_dsl + 1):
                summaries.append(
                    run_session(
                        variant=variant,
                        dsl=dsl,
                        run_index=run_index,
                        round_target=args.round_target,
                        shape_length=args.shape_length,
                        run_seed=master_rng.randrange(1, 10**9),
                        output_dir=output_dir,
                    )
                )

    write_summary_files(output_dir, args, summaries)

    aggregate_data = aggregate(summaries)
    print(f"Wrote mock A/B artifacts to {output_dir}")
    for key in sorted(aggregate_data):
        item = aggregate_data[key]
        print(
            f"{item['variant']:>7} {item['dsl']:<9} "
            f"pass_rate={item['pass_rate']:.2%} avg_rounds={item['avg_rounds']:.1f} "
            f"store_miss_runs={item['store_miss_runs']} cross_dsl_runs={item['cross_dsl_violation_runs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())