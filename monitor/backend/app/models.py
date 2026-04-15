from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    shape_key = Column(String(128), nullable=False, index=True)
    op_type = Column(String(64), nullable=True, default="gemm_sp")  # operator type
    dtype = Column(String(16), nullable=False)
    m = Column(Integer, nullable=False)
    n = Column(Integer, nullable=False)
    k = Column(Integer, nullable=False)
    mode = Column(String(32), nullable=False)
    max_iterations = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False, default="pending", index=True)
    current_iteration = Column(Integer, nullable=False, default=0)
    best_tflops = Column(Float, nullable=True)
    baseline_tflops = Column(Float, nullable=True)
    best_kernel = Column(String(512), nullable=True)
    model = Column(String(128), nullable=True)
    variant = Column(String(32), nullable=True, default="")
    agent_type = Column(String(32), nullable=True)  # cursor_ide, cursor_cli, opencode, copilot_ide
    opencode_session_id = Column(String(128), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    iteration_logs = relationship("IterationLog", back_populates="task", cascade="all, delete-orphan")
    agent_logs = relationship("AgentLog", back_populates="task", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "shape_key": self.shape_key,
            "op_type": self.op_type,
            "dtype": self.dtype,
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "mode": self.mode,
            "max_iterations": self.max_iterations,
            "status": self.status,
            "current_iteration": self.current_iteration,
            "best_tflops": self.best_tflops,
            "baseline_tflops": self.baseline_tflops,
            "best_kernel": self.best_kernel,
            "model": self.model,
            "variant": self.variant or "",
            "agent_type": self.agent_type,
            "opencode_session_id": self.opencode_session_id,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class IterationLog(Base):
    __tablename__ = "iteration_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    iteration = Column(Integer, nullable=False)
    kernel_path = Column(String(512), nullable=True)
    tflops = Column(Float, nullable=True)
    decision = Column(String(16), nullable=True)
    bottleneck = Column(String(128), nullable=True)
    idea_summary = Column(Text, nullable=True)
    logged_at = Column(DateTime, nullable=False, default=_utcnow)

    task = relationship("Task", back_populates="iteration_logs")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "iteration": self.iteration,
            "kernel_path": self.kernel_path,
            "tflops": self.tflops,
            "decision": self.decision,
            "bottleneck": self.bottleneck,
            "idea_summary": self.idea_summary,
            "logged_at": self.logged_at.isoformat() if self.logged_at else None,
        }


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False, index=True)
    level = Column(String(16), nullable=False, default="info")
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=_utcnow)

    task = relationship("Task", back_populates="agent_logs")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key = Column(String(64), primary_key=True)
    value = Column(String(256), nullable=False)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)
