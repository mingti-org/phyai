"""Thin GR00T-N1.7 adapter over the shared Qwen3-VL implementation."""

from __future__ import annotations

from typing import Any

import torch

from phyai.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from phyai.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


class GR00TN17Qwen3VLBackbone(Qwen3VLForConditionalGeneration):
    """Shared Qwen3-VL model with GR00T checkpoint/load-time defaults.

    GR00T consumes the pre-final-norm language hidden state as its backbone
    feature. The generic Qwen module exposes that via
    ``return_pre_norm_hidden_state=True``; this adapter only sets checkpoint
    prefixes and trims unused top LLM layers.
    """

    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        select_layer: int,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        attention_backend: str = "sdpa",
        vision_attention_backend: str = "flashinfer",
    ) -> None:
        super().__init__(
            config,
            params_dtype=params_dtype,
            device=device,
            attn_backend=attention_backend,
            vision_attn_backend=vision_attention_backend,
            prefix="backbone.model",
        )
        self.select_layer = int(select_layer)
        self._truncate_language_layers()

    def _truncate_language_layers(self) -> None:
        if self.select_layer < 0:
            return
        layers = self.model.language_model.layers
        while len(layers) > self.select_layer:
            layers.pop(-1)

    def forward(self, **kwargs: Any):
        kwargs.setdefault("return_pre_norm_hidden_state", True)
        return super().forward(**kwargs)


__all__ = ["GR00TN17Qwen3VLBackbone"]
