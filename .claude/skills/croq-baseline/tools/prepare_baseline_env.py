#!/usr/bin/env python3
"""Prepare baseline environment for croq-tune sessions."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_LIBS = {"cublas", "torch-cuda", "cusparselt"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_command(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def infer_libs(operator: str, kernel: str) -> list[str]:
    text = f"{operator} {kernel}".lower()
    libs: set[str] = {"torch-cuda"}

    if any(token in text for token in ["spmm", "sparse", "cusparselt"]):
        libs.add("cusparselt")
    if any(token in text for token in ["gemm", "dense", "matmul", "cublas"]):
        libs.add("cublas")

    if not libs:
        libs = {"torch-cuda", "cublas"}

    return sorted(libs)


def parse_libs(libs_arg: str, operator: str, kernel: str) -> list[str]:
    if libs_arg == "auto":
        return infer_libs(operator, kernel)
    libs = sorted({item.strip().lower() for item in libs_arg.split(",") if item.strip()})
    invalid = [item for item in libs if item not in SUPPORTED_LIBS]
    if invalid:
        raise SystemExit(f"Unsupported libs: {', '.join(invalid)}")
    return libs


def check_ldconfig_symbol(symbol: str) -> tuple[bool, str]:
    if shutil.which("ldconfig") is None:
        return False, "ldconfig-not-found"
    code, out, err = run_command(["ldconfig", "-p"])
    if code != 0:
        return False, f"ldconfig-error: {err or 'unknown'}"
    return (symbol in out), ("found-in-ldconfig" if symbol in out else "not-in-ldconfig")


def check_torch_cuda() -> dict:
    result = {
        "name": "torch-cuda",
        "present": False,
        "detail": "",
        "metadata": {},
        "install_hint": "python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124",
    }
    try:
        import torch  # type: ignore

        result["present"] = True
        result["detail"] = "import-ok"
        result["metadata"] = {
            "torch_version": getattr(torch, "__version__", "unknown"),
            "torch_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception as exc:  # pragma: no cover
        result["detail"] = f"import-failed: {exc}"
    return result


def check_cublas() -> dict:
    found, detail = check_ldconfig_symbol("libcublas.so")
    return {
        "name": "cublas",
        "present": found,
        "detail": detail,
        "metadata": {},
        "install_hint": "Install CUDA toolkit/runtime with cuBLAS libraries visible to loader.",
    }


def check_cusparselt() -> dict:
    found, detail = check_ldconfig_symbol("libcusparseLt.so")
    return {
        "name": "cusparselt",
        "present": found,
        "detail": detail,
        "metadata": {},
        "install_hint": "Install cuSPARSELt runtime (or python package nvidia-cusparselt-cu12) and ensure libcusparseLt is discoverable.",
    }


def detect_gpu_env() -> dict:
    info = {"nvidia_smi": False, "gpu_rows": [], "driver": "", "cuda_runtime": ""}
    if shutil.which("nvidia-smi") is None:
        return info

    query_variants = [
        ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"],
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
    ]

    for command in query_variants:
        code, out, _ = run_command(command)
        if code != 0 or not out:
            continue
        rows = [row.strip() for row in out.splitlines() if row.strip()]
        if not rows:
            continue
        info["nvidia_smi"] = True
        info["gpu_rows"] = rows
        parts = [part.strip() for part in rows[0].split(",")]
        if len(parts) >= 2:
            info["driver"] = parts[1]
        if len(parts) >= 3:
            info["cuda_runtime"] = parts[2]
        break
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare baseline environment for croq-baseline.")
    parser.add_argument("--dsl", default="cuda")
    parser.add_argument("--operator", default="spmm")
    parser.add_argument("--kernel", default="sparse_gemm")
    parser.add_argument("--shape-key", default="unknown")
    parser.add_argument("--libs", default="auto", help="auto or comma-separated libs: cublas,torch-cuda,cusparselt")
    parser.add_argument("--install-missing", action="store_true", help="Attempt install for pip-installable missing libs.")
    args = parser.parse_args()

    root = repo_root()
    selected_libs = parse_libs(args.libs, args.operator, args.kernel)

    checks: list[dict] = []
    for lib in selected_libs:
        if lib == "torch-cuda":
            checks.append(check_torch_cuda())
        elif lib == "cublas":
            checks.append(check_cublas())
        elif lib == "cusparselt":
            checks.append(check_cusparselt())

    missing = [item["name"] for item in checks if not item["present"]]
    report = {
        "schema": "baseline-env-prepare-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dsl": args.dsl,
        "operator": args.operator,
        "kernel": args.kernel,
        "shape_key": args.shape_key,
        "selected_libs": selected_libs,
        "gpu_env": detect_gpu_env(),
        "checks": checks,
        "missing_libs": missing,
        "install_attempted": bool(args.install_missing),
    }

    report_dir = root / "baseline-workspace" / "_env"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp_key = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = report_dir / f"prep_{timestamp_key}.json"
    latest_path = report_dir / "latest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[prepare-baseline-env] report: {report_path.relative_to(root)}")
    if missing:
        print(f"[prepare-baseline-env] missing libs: {', '.join(missing)}")
        for check in checks:
            if not check["present"]:
                print(f"[prepare-baseline-env] hint {check['name']}: {check['install_hint']}")
    else:
        print("[prepare-baseline-env] all selected libs are present")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
