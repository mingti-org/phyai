"""Condition-encoder execution and prompt feature preparation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai_utils_tools.models.qwen_image_21 import (
    QwenImage21ProcessedInputs,
    QwenImage21TokenizedPrompt,
)

from phyai.models.qwen_image_21.condition_encoder import QwenImage21ConditionEncoder
from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class QwenImage21Condition:
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    image_pad_mask: torch.Tensor


class QwenImage21ConditionRunner(ModelRunner):
    """Encode a prompt once before the scheduler enters its denoising loop."""

    def __init__(self, model: QwenImage21ConditionEncoder) -> None:
        self.model = model

    def setup(self) -> None:
        self.model.eval()

    @torch.inference_mode()
    def forward(
        self,
        batch: QwenImage21ProcessedInputs | QwenImage21TokenizedPrompt,
        *,
        num_images_per_prompt: int = 1,
    ) -> QwenImage21Condition:
        if num_images_per_prompt <= 0:
            raise ValueError("num_images_per_prompt must be positive.")
        prompt = (
            batch.prompt if isinstance(batch, QwenImage21ProcessedInputs) else batch
        )
        device = self.model.language_model.embed_tokens.weight.device
        input_ids = prompt.input_ids.to(device)
        mask = prompt.attention_mask.to(device)
        hidden = self.model(
            input_ids,
            attention_mask=mask,
            pixel_values=(
                prompt.pixel_values.to(device)
                if prompt.pixel_values is not None
                else None
            ),
            image_grid_thw=(
                prompt.image_grid_thw.to(device)
                if prompt.image_grid_thw is not None
                else None
            ),
        )
        valid = mask.bool()
        lengths = valid.sum(-1).tolist()
        if any(length <= prompt.drop_idx for length in lengths):
            raise ValueError("The prompt must contain tokens after the system message.")
        parts = hidden[valid].split(lengths)
        token_parts = input_ids[valid].split(lengths)
        parts = [part[prompt.drop_idx :] for part in parts]
        image_parts = [
            part[prompt.drop_idx :] == prompt.image_token_id for part in token_parts
        ]
        max_length = max(part.shape[0] for part in parts)
        embeds = hidden.new_zeros(len(parts), max_length, hidden.shape[-1])
        embeds_mask = mask.new_zeros(len(parts), max_length)
        image_mask = torch.zeros_like(embeds_mask, dtype=torch.bool)
        for index, (part, image_part) in enumerate(zip(parts, image_parts)):
            length = part.shape[0]
            embeds[index, :length] = part
            embeds_mask[index, :length] = 1
            image_mask[index, :length] = image_part
        return QwenImage21Condition(
            embeds.repeat_interleave(num_images_per_prompt, dim=0),
            None
            if bool(embeds_mask.all())
            else embeds_mask.repeat_interleave(num_images_per_prompt, dim=0),
            image_mask.repeat_interleave(num_images_per_prompt, dim=0),
        )


__all__ = ["QwenImage21Condition", "QwenImage21ConditionRunner"]
