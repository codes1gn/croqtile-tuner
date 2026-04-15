import subprocess
from pathlib import Path

from pydantic_settings import BaseSettings

AVAILABLE_VARIANTS = ("", "minimal", "low", "medium", "high", "xhigh", "max")

DEFAULT_OPENCODE_MODEL = "opencode/big-pickle"
DEFAULT_OPENCODE_VARIANT = "high"  # max is 3x cost — not worth it

_cached_models: list[str] | None = None


def fetch_opencode_models() -> list[str]:
    global _cached_models
    if _cached_models is not None:
        return _cached_models
    try:
        result = subprocess.run(
            ["opencode", "models"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if result.returncode == 0:
            models = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if models:
                _cached_models = models
                return models
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    _cached_models = [DEFAULT_OPENCODE_MODEL]
    return _cached_models


def invalidate_model_cache() -> None:
    global _cached_models
    _cached_models = None


def is_supported_opencode_model(model: str) -> bool:
    return model in fetch_opencode_models()


def is_valid_variant(variant: str) -> bool:
    return variant in AVAILABLE_VARIANTS


# monitor/ lives inside croqtile-tuner/, so project root is two levels up
_MONITOR_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _MONITOR_ROOT.parent  # croqtile-tuner/


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
    opencode_bin: str = "opencode"
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
