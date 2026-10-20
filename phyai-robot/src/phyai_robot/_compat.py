"""Exception groups on the robot's Python 3.10 interpreter."""

try:
    from builtins import ExceptionGroup, BaseExceptionGroup
except ImportError:
    from exceptiongroup import ExceptionGroup, BaseExceptionGroup

__all__ = ["BaseExceptionGroup", "ExceptionGroup"]
