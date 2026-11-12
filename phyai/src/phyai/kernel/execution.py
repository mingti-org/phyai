"""Host execution phases used by kernel capability selection."""

from contextlib import contextmanager
from contextvars import ContextVar

_phase: ContextVar[str] = ContextVar("phyai_kernel_execution", default="inference")


def current_execution() -> str:
    return _phase.get()


@contextmanager
def execution_phase(phase: str):
    """Scope phase facts during eager execution or graph construction."""
    if phase not in {"inference", "rollout"}:
        raise ValueError(f"Unknown kernel execution phase {phase!r}")
    token = _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(token)
