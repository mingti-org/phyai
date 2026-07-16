"""Runtime tensor layout and stage execution for MiniCPM-GR00T."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from phyai.layers.rotary_embedding import (
    compute_qwen3vl_mrope_cos_sin_from_inv_freq,
)
from phyai.models.minicpm_gr00t.modeling_minicpm_gr00t import MiniCPMGR00TModel
from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class MiniCPMGR00TVisionLayout:
    """Runner-owned indexing for the two MiniCPM-V spatial merge stages."""

    position_ids: torch.Tensor
    source_cu_seqlens: torch.Tensor
    window_indices: torch.Tensor
    inverse_window_indices: torch.Tensor
    window_cu_seqlens: torch.Tensor
    merged_cu_seqlens: torch.Tensor
    source_sizes: tuple[tuple[int, int], ...]
    merged_sizes: tuple[tuple[int, int], ...]


def build_vision_position_ids(
    target_sizes: torch.Tensor,
    *,
    position_grid_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Match MiniCPM-V's dynamic SigLIP position bucketization."""

    boundaries = torch.arange(
        1 / position_grid_size,
        1.0,
        1 / position_grid_size,
        dtype=torch.float32,
        device=device,
    )
    parts: list[torch.Tensor] = []
    for raw_h, raw_w in target_sizes.detach().cpu().tolist():
        h, w = int(raw_h), int(raw_w)
        coords_h = torch.arange(h, dtype=torch.float32, device=device) / h
        coords_w = torch.arange(w, dtype=torch.float32, device=device) / w
        bucket_h = torch.bucketize(coords_h, boundaries, right=True)
        bucket_w = torch.bucketize(coords_w, boundaries, right=True)
        parts.append((bucket_h[:, None] * position_grid_size + bucket_w).flatten())
    return torch.cat(parts).to(torch.int64)


def build_vision_layout(
    target_sizes: torch.Tensor,
    *,
    position_grid_size: int,
    device: torch.device,
) -> MiniCPMGR00TVisionLayout:
    """Build exact NaViT segment and 2x2-window layouts."""

    sizes = tuple((int(h), int(w)) for h, w in target_sizes.detach().cpu().tolist())
    if not sizes:
        raise ValueError("target_sizes must contain at least one image.")
    if any(h <= 0 or w <= 0 or h % 4 or w % 4 for h, w in sizes):
        raise ValueError(
            f"Every target grid must be positive and divisible by four, got {sizes}."
        )

    lengths = [h * w for h, w in sizes]
    source_cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )

    window_parts: list[torch.Tensor] = []
    token_offset = 0
    for h, w in sizes:
        indices = torch.arange(h * w, dtype=torch.int64, device=device).reshape(h, w)
        indices = indices.reshape(h // 2, 2, w // 2, 2).permute(0, 2, 1, 3).reshape(-1)
        window_parts.append(indices + token_offset)
        token_offset += h * w
    window_indices = torch.cat(window_parts)
    inverse_window_indices = torch.argsort(window_indices)
    window_cu = torch.arange(
        0,
        token_offset + 1,
        4,
        dtype=torch.int32,
        device=device,
    )

    merged_sizes = tuple((h // 2, w // 2) for h, w in sizes)
    merged_lengths = [h * w for h, w in merged_sizes]
    merged_cu = torch.tensor(
        [0, *torch.tensor(merged_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    return MiniCPMGR00TVisionLayout(
        position_ids=build_vision_position_ids(
            target_sizes,
            position_grid_size=position_grid_size,
            device=device,
        ),
        source_cu_seqlens=source_cu,
        window_indices=window_indices,
        inverse_window_indices=inverse_window_indices,
        window_cu_seqlens=window_cu,
        merged_cu_seqlens=merged_cu,
        source_sizes=sizes,
        merged_sizes=merged_sizes,
    )


def group_spatial_2x2(
    hidden_states: torch.Tensor,
    sizes: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group each image into flattened 2x2 cells and mean residuals."""

    if hidden_states.shape[0] != 1:
        raise ValueError("MiniCPM-GR00T vision layout currently requires batch_size=1.")
    grouped_parts: list[torch.Tensor] = []
    pooled_parts: list[torch.Tensor] = []
    offset = 0
    hidden_size = hidden_states.shape[-1]
    for h, w in sizes:
        count = h * w
        image = hidden_states[0, offset : offset + count]
        grouped = (
            image.reshape(h // 2, 2, w // 2, 2, hidden_size)
            .permute(0, 2, 1, 3, 4)
            .reshape((h // 2) * (w // 2), 4, hidden_size)
        )
        grouped_parts.append(grouped.reshape(grouped.shape[0], 4 * hidden_size))
        pooled_parts.append(grouped.mean(dim=1))
        offset += count
    if offset != hidden_states.shape[1]:
        raise ValueError(
            f"target_sizes describe {offset} tokens but hidden_states has "
            f"{hidden_states.shape[1]}."
        )
    return torch.cat(grouped_parts), torch.cat(pooled_parts)


def build_action_time_sinusoid(
    timesteps: torch.Tensor,
    *,
    horizon: int,
    embedding_dim: int,
) -> torch.Tensor:
    """Match ``SinusoidalPositionalEncoding`` in the reference action encoder."""

    half_dim = embedding_dim // 2
    exponent = -torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    ) * (math.log(10_000.0) / half_dim)
    expanded = timesteps.to(torch.float32)[:, None].expand(-1, horizon)
    frequencies = expanded[..., None] * exponent.exp()
    return torch.cat((frequencies.sin(), frequencies.cos()), dim=-1)


def build_dit_time_sinusoid(
    timesteps: torch.Tensor,
    *,
    embedding_dim: int,
) -> torch.Tensor:
    """Match Diffusers ``Timesteps(256, flip_sin_to_cos=True, shift=1)``."""

    half_dim = embedding_dim // 2
    exponent = -math.log(10_000.0) * torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - 1)
    frequencies = timesteps.to(torch.float32)[:, None] * exponent.exp()[None]
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


class MiniCPMGR00TModelRunner(ModelRunner):
    """Eager reference runner; all dynamic tensor construction lives here."""

    def __init__(
        self,
        model: MiniCPMGR00TModel,
        *,
        device: torch.device | str,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.config = model.config

    def setup(self) -> None:
        self.model.eval()

    def _encode_vision_segments(
        self,
        hidden_states: torch.Tensor,
        sizes: tuple[tuple[int, int], ...],
        *,
        suffix: bool,
    ) -> torch.Tensor:
        """Run each image as a padded batch-one sequence for SDPA."""

        parts: list[torch.Tensor] = []
        offset = 0
        for h, w in sizes:
            count = h * w
            segment = hidden_states[:, offset : offset + count]
            if suffix:
                segment = self.model.encode_vision_suffix(segment)
            else:
                segment = self.model.encode_vision_prefix(segment)
            parts.append(segment)
            offset += count
        if offset != hidden_states.shape[1]:
            raise ValueError(
                f"Vision sizes describe {offset} tokens but hidden states has "
                f"{hidden_states.shape[1]}."
            )
        return torch.cat(parts, dim=1)

    @torch.inference_mode()
    def encode_vlm(
        self,
        *,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        target_sizes: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run NaViT, scatter image features, then Qwen3.5 prefill."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must have shape (1, S), got {tuple(input_ids.shape)}."
            )
        target_sizes = target_sizes.to(device=self.device, dtype=torch.int32)
        position_grid_size = (
            self.config.vision.image_size // self.config.vision.patch_size
        )
        layout = build_vision_layout(
            target_sizes,
            position_grid_size=position_grid_size,
            device=self.device,
        )

        pixels = pixel_values.to(device=self.device, dtype=torch.bfloat16)
        vision = self.model.embed_vision(pixels, layout.position_ids)
        vision = self._encode_vision_segments(
            vision,
            layout.source_sizes,
            suffix=False,
        )

        ordered = vision[:, layout.window_indices]
        hidden_size = ordered.shape[-1]
        ordered = ordered.reshape(-1, 4, hidden_size)
        ordered = self.model.attend_vision_windows(
            ordered,
        )
        vision = ordered.reshape(1, -1, hidden_size)
        vision = vision[:, layout.inverse_window_indices]
        grouped, pooled = group_spatial_2x2(vision, layout.source_sizes)
        vision = self.model.merge_vision_windows(grouped, pooled).unsqueeze(0)

        vision = self._encode_vision_segments(
            vision,
            layout.merged_sizes,
            suffix=True,
        )
        vision = self.model.finish_vision(vision)
        grouped, _ = group_spatial_2x2(vision, layout.merged_sizes)
        image_features = self.model.downsample_vision(grouped)

        input_ids = input_ids.to(device=self.device, dtype=torch.int64)
        text = self.model.embed_text(input_ids)
        image_mask = input_ids == self.config.image_token_id
        image_token_count = int(image_mask.sum().item())
        if image_token_count != image_features.shape[0]:
            raise ValueError(
                f"input_ids contain {image_token_count} image tokens but vision "
                f"produced {image_features.shape[0]} features."
            )
        expanded_mask = image_mask.unsqueeze(-1).expand_as(text)
        text = text.masked_scatter(
            expanded_mask,
            image_features.to(text.dtype).reshape(-1),
        )

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(
            seq_len,
            dtype=torch.int64,
            device=self.device,
        ).view(1, 1, seq_len)
        position_ids = position_ids.expand(3, batch_size, seq_len)
        text_model = self.model.vlm.llm.model
        cos, sin = compute_qwen3vl_mrope_cos_sin_from_inv_freq(
            position_ids,
            text_model.rotary_emb.inv_freq,
            self.config.text.mrope_section,
        )
        mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
            if not bool(torch.all(attention_mask == 1)):
                raise NotImplementedError(
                    "MiniCPM-GR00T currently supports unpadded batch-one prefill only."
                )
        return self.model.encode_text(
            text,
            cos=cos.to(text.dtype),
            sin=sin.to(text.dtype),
            attention_mask=mask,
        )

    @torch.inference_mode()
    def predict_clean_action(
        self,
        *,
        vlm_hidden_states: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Run one clean-action DiT prediction at a discrete timestep."""

        cfg = self.config.action
        batch_size = noisy_actions.shape[0]
        if noisy_actions.shape != (batch_size, cfg.action_horizon, cfg.action_dim):
            raise ValueError(
                f"noisy_actions has shape {tuple(noisy_actions.shape)}; expected "
                f"(B, {cfg.action_horizon}, {cfg.action_dim})."
            )
        state = state.to(device=self.device, dtype=torch.float32)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.shape != (batch_size, 1, cfg.proprio_dim):
            raise ValueError(
                f"state has shape {tuple(state.shape)}; expected "
                f"(B, 1, {cfg.proprio_dim})."
            )

        noisy_actions = noisy_actions.to(device=self.device, dtype=torch.float32)
        action_proprio = torch.cat(
            (noisy_actions, state.expand(-1, cfg.action_horizon, -1)),
            dim=-1,
        )
        projected = self.model.project_action_proprio(action_proprio)
        action_time = build_action_time_sinusoid(
            timestep,
            horizon=cfg.action_horizon,
            embedding_dim=cfg.dit.hidden_size,
        ).to(projected.dtype)
        action_features = self.model.encode_action(
            torch.cat((projected, action_time), dim=-1)
        )

        action_position_ids = torch.arange(
            cfg.action_horizon,
            dtype=torch.int64,
            device=self.device,
        )
        action_features = (
            action_features
            + self.model.action_head.embed_action_positions(
                action_position_ids
            ).unsqueeze(0)
        )
        future_ids = torch.arange(
            cfg.num_future_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        future = self.model.action_head.embed_future(future_ids).unsqueeze(0)
        future = future.expand(batch_size, -1, -1)
        dit_input = torch.cat((future, action_features), dim=1)

        dit_time = build_dit_time_sinusoid(
            timestep,
            embedding_dim=cfg.dit.timestep_input_dim,
        )
        dit_output = self.model.run_dit(
            dit_input,
            vlm_hidden_states.to(device=self.device, dtype=torch.float32),
            dit_time,
        )
        decoded = self.model.decode_action(dit_output)
        return decoded[:, -cfg.action_horizon :]

    def forward(self, batch) -> torch.Tensor:
        raise NotImplementedError(
            "Use encode_vlm() and predict_clean_action(); the scheduler owns orchestration."
        )


__all__ = [
    "MiniCPMGR00TModelRunner",
    "MiniCPMGR00TVisionLayout",
    "build_action_time_sinusoid",
    "build_dit_time_sinusoid",
    "build_vision_layout",
    "build_vision_position_ids",
    "group_spatial_2x2",
]
