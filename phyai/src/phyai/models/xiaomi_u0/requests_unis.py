"""Tokenized requests and results for single-sequence U0 generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class UNISGenerationRequest:
    """One tokenized image request; replay is for regression only.

    Scene Gen samples the output dimensions and stops at the first image end.
    """

    input_ids: torch.Tensor
    unconditional_ids: torch.Tensor
    max_new_tokens: int = 4169
    seed: int = 42
    replay_ids: torch.Tensor | None = None
    capture_steps: tuple[int, ...] = ()
    scene_gen: bool = False


@dataclass(frozen=True)
class UNISGenerationResult:
    """Generated-only CPU token IDs, termination and optional debug traces."""

    generated_ids: torch.Tensor
    stop_reason: str
    replay: bool
    conditional_cached_tokens: int
    unconditional_cached_tokens: int
    traces: tuple[dict[str, Any], ...] = field(default_factory=tuple)


__all__ = ["UNISGenerationRequest", "UNISGenerationResult"]
