"""Explicit opt-in ZMQ backend; pyzmq is checked when connecting."""

from .backend import ZmqBackend

__all__ = ["ZmqBackend"]
