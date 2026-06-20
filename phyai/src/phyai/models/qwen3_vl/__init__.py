"""phyai.models.qwen3_vl — Qwen3-VL reference implementation.

A faithful port of HuggingFace Qwen3-VL onto phyai layer primitives:

* :mod:`configuration_qwen3_vl` — frozen dataclass configs
  (:class:`Qwen3VLVisionConfig`, :class:`Qwen3VLTextConfig`,
  :class:`Qwen3VLConfig`).
* :mod:`modeling_qwen3_vl` — every ``nn.Module``: the native ViT vision tower
  (:class:`Qwen3VLVisionModel`), the Qwen3 text decoder with interleaved 3-D
  M-RoPE and DeepStack injection (:class:`Qwen3VLTextModel`), the multimodal
  fusion (:class:`Qwen3VLModel`), and the LM-head wrapper
  (:class:`Qwen3VLForConditionalGeneration`).

This package is a reference *implementation* — the module graph and parameter
names mirror the HF checkpoint, but engine registration, a CUDA-graph runner,
a scheduler, and real-checkpoint validation are not in scope yet (see the
:mod:`modeling_qwen3_vl` docstring).
"""

from __future__ import annotations

from phyai.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from phyai.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)


__all__ = [
    "Qwen3VLConfig",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLTextModel",
    "Qwen3VLVisionModel",
]
