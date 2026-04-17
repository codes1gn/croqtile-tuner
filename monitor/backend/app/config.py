import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger("croqtuner.config")

AVAILABLE_VARIANTS = ("", "minimal", "low", "medium", "high", "xhigh")

DEFAULT_OPENCODE_MODEL = "github-copilot/gpt-5-mini"
DEFAULT_OPENCODE_VARIANT = "high"  # max is 3x cost — not worth it

_MODEL_CACHE_TTL = 300  # 5 minutes

_cached_opencode_models: list[str] | None = None
_cached_opencode_ts: float = 0.0
_cached_cursor_models: list[str] | None = None
_cached_cursor_ts: float = 0.0


OPENCODE_FREE_MODELS = {
    "opencode/big-pickle",
    "opencode/nemotron-3-super-free",
    "opencode/minimax-m2.5-free",
}


def fetch_opencode_models() -> list[str]:
    """Fetch opencode models, keeping only free opencode/zen + all github-copilot models.

    Results are cached for _MODEL_CACHE_TTL seconds. A failed fetch that only
    produces the fallback default is cached for just 30 seconds so a retry
    happens quickly once opencode becomes available.
    """
    global _cached_opencode_models, _cached_opencode_ts
    if _cached_opencode_models is not None and (time.monotonic() - _cached_opencode_ts) < _MODEL_CACHE_TTL:
        return _cached_opencode_models
    opencode_bin = _resolve_opencode_bin()
    try:
        result = subprocess.run(
            [opencode_bin, "models"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            all_models = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            filtered = [
                m for m in all_models
                if m in OPENCODE_FREE_MODELS or m.startswith("github-copilot/")
            ]
            if filtered:
                _cached_opencode_models = filtered
                _cached_opencode_ts = time.monotonic()
                return filtered
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    _cached_opencode_models = [DEFAULT_OPENCODE_MODEL]
    _cached_opencode_ts = time.monotonic() - _MODEL_CACHE_TTL + 30  # retry after 30s
    return _cached_opencode_models


CURSOR_ALLOWED_MODELS = {
    "cursor/claude-4.6-opus-max",
    "cursor/claude-4.6-opus-high",
    "cursor/claude-4.5-opus-high",
    "cursor/claude-4.6-sonnet-medium",
    "cursor/gpt-5.3-codex-high",
}


def fetch_cursor_cli_models() -> list[str]:
    """Fetch models from cursor-agent CLI, filtered to allowed set.

    No thinking models. No max-mode variants.
    """
    global _cached_cursor_models, _cached_cursor_ts
    if _cached_cursor_models is not None and (time.monotonic() - _cached_cursor_ts) < _MODEL_CACHE_TTL:
        return _cached_cursor_models

    all_models: list[str] = []
    try:
        result = subprocess.run(
            ["script", "-qc", "cursor-agent models", "/dev/null"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[\?25[hl]")
            for line in result.stdout.splitlines():
                clean = ansi_re.sub("", line).strip()
                if " - " in clean and not clean.lower().startswith(("available", "tip:")):
                    model_id = clean.split(" - ", 1)[0].strip()
                    if model_id and not model_id.startswith("-"):
                        full_id = f"cursor/{model_id}"
                        if full_id in CURSOR_ALLOWED_MODELS:
                            all_models.append(full_id)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("cursor-agent not available for model listing")

    if not all_models:
        all_models = [m for m in CURSOR_ALLOWED_MODELS]
        all_models.sort()
    _cached_cursor_models = all_models
    _cached_cursor_ts = time.monotonic()
    return _cached_cursor_models


def fetch_all_models() -> list[str]:
    """Return combined model list from all supported platforms."""
    all_models: list[str] = []
    all_models.extend(fetch_opencode_models())
    all_models.extend(fetch_cursor_cli_models())
    return all_models


def invalidate_model_cache() -> None:
    global _cached_opencode_models, _cached_opencode_ts, _cached_cursor_models, _cached_cursor_ts
    _cached_opencode_models = None
    _cached_opencode_ts = 0.0
    _cached_cursor_models = None
    _cached_cursor_ts = 0.0


def is_supported_opencode_model(model: str) -> bool:
    return model in fetch_opencode_models()


def is_valid_variant(variant: str) -> bool:
    return variant in AVAILABLE_VARIANTS


# monitor/ lives inside croqtile-tuner/, so project root is two levels up
_MONITOR_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _MONITOR_ROOT.parent  # croqtile-tuner/


def _resolve_opencode_bin() -> str:
    """Resolve the opencode binary to a full path."""
    found = shutil.which("opencode")
    if found:
        return found
    home_bin = Path.home() / ".opencode" / "bin" / "opencode"
    if home_bin.exists():
        return str(home_bin)
    return "opencode"


class Settings(BaseSettings):
    # croqtile-tuner paths (one level up from monitor/)
    tuning_dir: Path = _PROJECT_ROOT / "tuning"
    skills_dir: Path = _PROJECT_ROOT / ".claude" / "skills"
    project_dir: Path = _PROJECT_ROOT
    
    # Monitor-specific paths
    monitor_dir: Path = _MONITOR_ROOT
    db_path: str = str(_MONITOR_ROOT / "data" / "monitor.db")
    
    # Heartbeat and auto-wake settings
    heartbeat_sec: int = 30
    auto_wake_enabled: bool = False  # Toggle for auto-waking opencode
    
    # opencode settings
    opencode_bin: str = _resolve_opencode_bin()
    opencode_model: str = DEFAULT_OPENCODE_MODEL
    opencode_variant: str = DEFAULT_OPENCODE_VARIANT
    opencode_db_path: Path = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
    
    # GPU and execution settings
    cuda_visible_devices: str = "0"
    mock_mode: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8642
    
    # Choreo compiler settings
    choreo_home: Path = Path("/home/albert/workspace/croqtile")
    cute_home: Path = Path("/home/albert/workspace/croqtile/extern/cutlass")

    model_config = {"env_prefix": "CROQTUNER_"}


settings = Settings()
