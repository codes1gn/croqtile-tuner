from __future__ import annotations

import random

from app.artifact_scanner import _CANONICAL_DTYPE, _is_baseline_row, _normalize_dtype


def test_is_baseline_row_iter000_markers_only():
    assert _is_baseline_row("iter000_cublas", None) is True
    assert _is_baseline_row("iter000_torch", None) is True
    assert _is_baseline_row("iter000_baseline_ref", None) is True
    assert _is_baseline_row("framework/torch_mm", None) is True
    assert _is_baseline_row("iter001_baseline_1p1c", None) is False
    assert _is_baseline_row("iter123_custom", "baseline") is True


def test_normalize_dtype_random_compositions():
    rng = random.Random(0)
    tokens = list(_CANONICAL_DTYPE.keys())

    for _ in range(500):
        picked = [rng.choice(tokens) for _ in range(rng.randint(1, 5))]
        raw = "".join(picked)
        expected = "".join(_CANONICAL_DTYPE[t] for t in picked)
        assert _normalize_dtype(raw) == expected


def test_normalize_dtype_idempotent_on_random_noise():
    rng = random.Random(1)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    for _ in range(500):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 20)))
        normalized = _normalize_dtype(raw)
        assert _normalize_dtype(normalized) == normalized
