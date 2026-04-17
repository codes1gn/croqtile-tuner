from pydantic import BaseModel, field_validator, model_validator

from .config import is_valid_variant


VALID_INPUT_DTYPES = ("f16", "e4m3", "e5m2", "bf16", "f32")
VALID_OUTPUT_DTYPES = ("f16", "f32", "bf16")
VALID_OP_TYPES = (
    "gemm_sp",
    "gemm",
    "matmul",
    "matmul_bf16",
    "blockscale_gemm",
    "bmm",
    "gemv",
    "fused_moe",
)


VALID_DSLS = ("croqtile", "cuda", "triton", "cute", "cutile", "helion", "tilelang")
VALID_PLATFORMS = ("opencode", "cursor_cli")


class TaskCreate(BaseModel):
    op_type: str = "gemm_sp"
    dtype: str
    m: int
    n: int
    k: int
    dsl: str = "croqtile"
    mode: str = "opencode"
    model: str
    variant: str = ""
    request_budget: int = 1

    @field_validator("op_type")
    @classmethod
    def validate_op_type(cls, v: str) -> str:
        return v

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        if v in VALID_INPUT_DTYPES:
            return v
        for inp in VALID_INPUT_DTYPES:
            if v.startswith(inp):
                out = v[len(inp):]
                if out in VALID_OUTPUT_DTYPES:
                    return v
        raise ValueError(
            f"dtype must be a valid input type {VALID_INPUT_DTYPES} "
            f"or input+output combo (e.g. bf16fp32, f16f32)"
        )

    @field_validator("m")
    @classmethod
    def validate_m(cls, v: int) -> int:
        if v < 128:
            raise ValueError("M must be >= 128")
        return v

    @field_validator("n")
    @classmethod
    def validate_n(cls, v: int) -> int:
        if v < 256:
            raise ValueError("N must be >= 256")
        return v

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        if v < 128:
            raise ValueError("K must be >= 128")
        return v

    @field_validator("dsl")
    @classmethod
    def validate_dsl(cls, v: str) -> str:
        if v not in VALID_DSLS:
            raise ValueError(f"dsl must be one of {VALID_DSLS}")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in VALID_PLATFORMS:
            raise ValueError(f"mode (agent platform) must be one of {VALID_PLATFORMS}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model is required")
        return v.strip()

    @field_validator("variant")
    @classmethod
    def validate_variant(cls, v: str) -> str:
        if not is_valid_variant(v):
            raise ValueError("unsupported variant")
        return v

    @field_validator("request_budget")
    @classmethod
    def validate_request_budget(cls, v: int) -> int:
        if v < 1:
            raise ValueError("request_budget must be >= 1")
        return v

    @model_validator(mode="after")
    def validate_platform_model_compatibility(self) -> "TaskCreate":
        mode = self.mode
        model = self.model
        if mode == "opencode" and model and model.startswith("cursor/"):
            raise ValueError(
                f"Model '{model}' is a Cursor IDE model and cannot be used with the opencode platform. "
                "Use an opencode/ or github-copilot/ model instead."
            )
        if mode == "cursor_cli" and model and (
            model.startswith("opencode/") or model.startswith("github-copilot/")
        ):
            raise ValueError(
                f"Model '{model}' is an OpenCode/GitHub-Copilot model and cannot be used with the cursor_cli platform. "
                "Use a cursor/ model instead."
            )
        return self


class TaskUpdate(BaseModel):
    status: str | None = None
    model: str | None = None
    variant: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str | None) -> str | None:
        if v is not None and v not in ("pending", "cancelled", "waiting"):
            raise ValueError("status must be 'pending', 'cancelled', or 'waiting'")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("model cannot be empty")
        return v.strip() if v is not None else None

    @field_validator("variant")
    @classmethod
    def validate_variant(cls, v: str | None) -> str | None:
        if v is not None and not is_valid_variant(v):
            raise ValueError("unsupported variant")
        return v


VALID_TASK_STATUSES = (
    "pending",
    "waiting",
    "running",
    "completed",
    "cancelled",
)


class TaskResponse(BaseModel):
    id: int
    shape_key: str
    op_type: str | None
    dtype: str
    m: int
    n: int
    k: int
    mode: str
    dsl: str | None
    max_iterations: int
    status: str
    current_iteration: int
    best_tflops: float | None
    baseline_tflops: float | None
    best_kernel: str | None
    model: str | None
    variant: str | None
    request_budget: int
    request_number: int
    agent_type: str | None
    device: str | None
    opencode_session_id: str | None
    session_ids: list[str] = []
    error_message: str | None
    created_at: str | None
    updated_at: str | None
    started_at: str | None
    completed_at: str | None


class TaskSessionResponse(BaseModel):
    id: int
    task_id: int
    session_id: str
    agent_type: str | None
    model: str | None
    request_number: int | None
    started_at: str | None
    ended_at: str | None


class IterationLogResponse(BaseModel):
    id: int
    task_id: int
    iteration: int
    request_number: int | None
    kernel_path: str | None
    tflops: float | None
    decision: str | None
    bottleneck: str | None
    idea_summary: str | None
    logged_at: str | None


class AgentLogResponse(BaseModel):
    id: int
    task_id: int
    level: str
    message: str
    timestamp: str | None


class SessionHistoryEntryResponse(BaseModel):
    id: str
    message_id: str
    role: str
    kind: str
    text: str
    tool: str | None
    status: str | None
    timestamp: str | None


class SessionHistoryResponse(BaseModel):
    session_id: str | None
    session_title: str | None
    session_directory: str | None
    entries: list[SessionHistoryEntryResponse]


class HealthResponse(BaseModel):
    status: str
    scheduler_running: bool
    active_task_id: int | None
    auto_wake_enabled: bool
    gpu_info: str | None
    default_model: str
    default_variant: str
    available_models: list[str]
    available_variants: list[str]
    task_counts: dict[str, int]


class ResumeRequest(BaseModel):
    from_iteration: int = 0

    @field_validator("from_iteration")
    @classmethod
    def validate_from_iteration(cls, v: int) -> int:
        if v < 0:
            raise ValueError("from_iteration must be >= 0")
        return v


class ModelSettingsUpdate(BaseModel):
    default_model: str
    default_variant: str = ""

    @field_validator("default_variant")
    @classmethod
    def validate_default_variant(cls, v: str) -> str:
        if not is_valid_variant(v):
            raise ValueError("unsupported variant")
        return v


class ModelSettingsResponse(BaseModel):
    default_model: str
    default_variant: str
    available_models: list[str]
    available_variants: list[str]


class AutoWakeSettingsResponse(BaseModel):
    auto_wake_enabled: bool


class AutoWakeSettingsUpdate(BaseModel):
    auto_wake_enabled: bool
