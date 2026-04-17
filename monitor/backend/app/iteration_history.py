from __future__ import annotations

from pathlib import Path

from .config import settings


def _find_results_tsv(shape_key: str) -> Path | None:
    """Find results.tsv file in the croqtile-tuner tuning directory structure.

    Accepts compound keys ({gpu}/{dsl}/{bare_shape_key}/{model}),
    legacy 3-part keys ({gpu}/{dsl}/{bare_shape_key}), and bare shape keys.
    Structure: tuning/<gpu>/<dsl>/logs/<shape_key>/<model>/results.tsv
    """
    tuning_dir = settings.tuning_dir
    if not tuning_dir.exists():
        return None

    parts = shape_key.split("/")
    if len(parts) == 4:
        # Compound key: {gpu}/{dsl}/{bare_shape_key}/{model}
        candidate = tuning_dir / parts[0] / parts[1] / "logs" / parts[2] / parts[3] / "results.tsv"
        if candidate.exists():
            return candidate
        return None

    if len(parts) == 3:
        # Legacy 3-part key: {gpu}/{dsl}/{bare_shape_key} — search model dirs
        logs_dir = tuning_dir / parts[0] / parts[1] / "logs" / parts[2]
        if logs_dir.is_dir():
            for model_dir in logs_dir.iterdir():
                if model_dir.is_dir():
                    candidate = model_dir / "results.tsv"
                    if candidate.exists():
                        return candidate
        return None

    # Bare shape key — search all gpu/dsl/model combos
    for gpu_dir in tuning_dir.iterdir():
        if not gpu_dir.is_dir():
            continue
        for dsl_dir in gpu_dir.iterdir():
            if not dsl_dir.is_dir():
                continue
            shape_dir = dsl_dir / "logs" / shape_key
            if shape_dir.is_dir():
                for model_dir in shape_dir.iterdir():
                    if model_dir.is_dir():
                        results_path = model_dir / "results.tsv"
                        if results_path.exists():
                            return results_path

    return None


def read_iteration_history(shape_key: str, task_id: int) -> list[dict]:
    """Read iteration history from results.tsv.
    
    TSV format: iter<tab>kernel<tab>tflops<tab>decision<tab>bottleneck<tab>idea
    """
    results_path = _find_results_tsv(shape_key)
    if results_path is None:
        return []

    try:
        lines = results_path.read_text(errors="replace").splitlines()
    except OSError:
        return []

    entries: list[dict] = []
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        
        # Skip header line
        if line.lower().startswith("iter\t"):
            continue

        parts = [part.strip() for part in line.split("\t")]
        if len(parts) < 3:
            continue

        # Parse iteration number (iterXXX or just the number)
        iter_str = parts[0]
        if iter_str.lower().startswith("iter"):
            iter_str = iter_str[4:]
        try:
            iteration = int(iter_str)
        except ValueError:
            continue

        # Parse kernel name
        kernel_path = parts[1] if len(parts) > 1 and parts[1] else None

        # Parse TFLOPS
        try:
            tflops = float(parts[2])
        except (ValueError, IndexError):
            tflops = None

        decision = parts[3] if len(parts) > 3 and parts[3] else None
        bottleneck = parts[4] if len(parts) > 4 and parts[4] else None
        idea_summary = parts[5] if len(parts) > 5 and parts[5] else None

        from .artifact_scanner import _is_baseline_row
        is_baseline = _is_baseline_row(kernel_path, bottleneck)
        if is_baseline:
            decision = "BASELINE"

        entries.append(
            {
                "id": -iteration,
                "task_id": task_id,
                "iteration": iteration,
                "kernel_path": kernel_path,
                "tflops": tflops,
                "decision": decision,
                "bottleneck": bottleneck,
                "idea_summary": idea_summary,
                "logged_at": None,
            }
        )

    entries.sort(key=lambda entry: entry["iteration"], reverse=True)
    return entries