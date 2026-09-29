"""DreamZero text encoder runner."""

from __future__ import annotations

import torch

from phyai.models.dreamzero.text_encoder_wan import DreamZeroWanTextEncoder
from phyai.runtime.model_runner import ModelRunner


class DreamZeroTextEncoderRunner(ModelRunner):
    """Wraps the DreamZero Wan2.1 UMT5 text encoder."""

    def __init__(
        self,
        text_encoder: DreamZeroWanTextEncoder,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.text_encoder = text_encoder
        self.device = torch.device(device)
        self.dtype = dtype

    def setup(self) -> None:
        return None

    def _encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} must match "
                f"attention_mask shape {tuple(attention_mask.shape)}."
            )
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        attention_mask = attention_mask.to(device=self.device)
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=self.dtype)
        return prompt_emb.masked_fill(
            attention_mask.to(torch.bool).unsqueeze(-1) == 0, 0
        )

    @torch.no_grad()
    def encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_prompt(input_ids, attention_mask)

    @torch.no_grad()
    def encode_positive_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_prompt(input_ids, attention_mask)

    @torch.no_grad()
    def encode_negative_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_prompt(input_ids, attention_mask)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_prompt(input_ids, attention_mask)


__all__ = ["DreamZeroTextEncoderRunner"]
