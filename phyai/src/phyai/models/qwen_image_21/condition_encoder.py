"""Qwen3-VL condition features consumed by the Qwen-Image 2.1 transformer."""

from __future__ import annotations

import torch

from phyai.layers.backbones.qwen3_vl import Qwen3VLConfig, Qwen3VLModel


def qwen_image_21_condition_weight_remap(name: str) -> str | None:
    return None if name == "lm_head.weight" else name


class QwenImage21ConditionEncoder(Qwen3VLModel):
    """Return the last decoder layer before final RMS normalization."""

    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        vision_attn_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "model",
    ) -> None:
        super().__init__(
            config,
            params_dtype=params_dtype,
            vision_params_dtype=vision_params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            vision_attn_backend=vision_attn_backend,
            norm_cast_before_affine=True,
            mlp_cast_before_multiply=True,
            device=device,
            prefix=prefix,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_pre_norm_hidden_state=True,
        )


__all__ = [
    "QwenImage21ConditionEncoder",
    "qwen_image_21_condition_weight_remap",
]
