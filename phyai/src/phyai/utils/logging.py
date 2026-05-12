"""Rank-aware logging helpers for phyai.

Two entry points:

* :func:`this_rank_log` emits the message only on a specific distributed
  rank (default rank 0). In non-distributed runs it behaves like a plain
  :meth:`logging.Logger.log` call.
* :func:`all_ranks_log` emits the message on every rank. Each line is
  prefixed with the emitting rank so interleaved multi-rank output stays
  readable.

Both helpers prepend a ``[rank R/W]`` prefix (or ``[rank -/-]`` outside
a distributed context) to the message before handing it to the logger,
so stdout/stderr across ranks remains easy to grep.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.distributed as dist


def _rank_prefix() -> str:
    if dist.is_available() and dist.is_initialized():
        return f"[rank {dist.get_rank()}/{dist.get_world_size()}]"
    return "[rank -/-]"


def this_rank_log(
    logger: logging.Logger,
    level: int,
    msg: Any,
    *args: Any,
    rank: int = 0,
    **kwargs: Any,
) -> None:
    """Log ``msg`` on a single distributed rank.

    Args:
        logger: The logger to write the log.
        level: The log level (e.g., ``logging.INFO``).
        msg: The message or printf-style format string.
        args: Positional args interpolated into ``msg``.
        rank: The rank on which to emit the message. Defaults to 0.
        kwargs: Additional keyword args forwarded to ``Logger.log``.
    """
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != rank:
            return
    logger.log(level, f"{_rank_prefix()} {msg}", *args, **kwargs)


def all_ranks_log(
    logger: logging.Logger,
    level: int,
    msg: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Log ``msg`` on every distributed rank, prefixed with the rank id.

    Args:
        logger: The logger to write the log.
        level: The log level (e.g., ``logging.INFO``).
        msg: The message or printf-style format string.
        args: Positional args interpolated into ``msg``.
        kwargs: Additional keyword args forwarded to ``Logger.log``.
    """
    logger.log(level, f"{_rank_prefix()} {msg}", *args, **kwargs)
