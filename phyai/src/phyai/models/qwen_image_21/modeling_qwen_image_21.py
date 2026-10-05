# Copyright 2026 Qwen-Image Team, The HuggingFace Team, PHYAI contributors.
# Licensed under the Apache License, Version 2.0.
"""Stateless single-stream Qwen-Image 2.1 denoiser using PHYAI layers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from phyai.engine_config import get_engine_config
from phyai.layers.attention import Attention, AttnCtx, AttnMask
from phyai.layers.axial_rotary_embedding import AxialRotaryEmbedding
from phyai.layers.layer_norm import GemmaRMSNorm, LayerNorm, RMSNorm
from phyai.layers.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from phyai.layers.mlp import DenseMLP

from phyai.models.qwen_image_21.configuration_qwen_image_21 import QwenImage21Config


PrefixKV = tuple[torch.Tensor, torch.Tensor]
PreparedModulation = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class QwenImage21PreparedCondition:
    text: torch.Tensor
    expansion_indices: torch.Tensor
    image_indices: torch.Tensor
    target_mask: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    segments: tuple[tuple[int, int, bool], ...]
    key_valid: torch.Tensor | None
    prefix_length: int
    target_tokens: int


@dataclass(frozen=True)
class QwenImage21ModelOutput:
    sample: torch.Tensor
    prefix_kv: tuple[PrefixKV, ...] | None = None


def select_modulation(
    value: torch.Tensor, target_mask: torch.Tensor | None
) -> torch.Tensor:
    if target_mask is None:
        return value.unsqueeze(1)
    return torch.where(target_mask[None, :, None], value[:-1, None], value[-1:][None])


def prepare_modulation(
    modulation: torch.Tensor,
    target_mask: torch.Tensor | None,
    *,
    cached_target: bool,
) -> PreparedModulation:
    parts = modulation.chunk(4, dim=-1)
    if cached_target:
        scale1, gate1, scale2, gate2 = (part[:-1, None] for part in parts)
    else:
        scale1, gate1, scale2, gate2 = (
            select_modulation(part, target_mask) for part in parts
        )
    return 1 + scale1, gate1.tanh(), 1 + scale2, gate2.tanh()


def apply_gated_residual(
    hidden: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return hidden + gate * update


class QwenImage21TimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int, *, params_dtype, device) -> None:
        super().__init__()
        self.register_buffer(
            "freqs",
            torch.exp(
                -math.log(10000) * torch.arange(128, device=device).float() / 128
            ),
            persistent=False,
        )
        prefix = "time_text_embed.timestep_embedder"
        self.linear_1 = ReplicatedLinear(
            256,
            hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        phases = timestep.float()[:, None] * 1000.0 * self.freqs[None]
        embedding = torch.cat((phases.cos(), phases.sin()), dim=-1).to(timestep.dtype)
        return self.linear_2(F.silu(self.linear_1(embedding)[0]))[0]


class QwenImage21TextProjection(nn.Module):
    def __init__(self, config: QwenImage21Config, *, params_dtype, device) -> None:
        super().__init__()
        self.text_norm = GemmaRMSNorm(
            config.context_in_dim,
            config.eps,
            dtype=params_dtype,
            device=device,
            prefix="txt_in.text_norm",
        )
        self.in_layer = ReplicatedLinear(
            config.context_in_dim,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="txt_in.in_layer",
        )
        self.out_layer = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="txt_in.out_layer",
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.in_layer(self.text_norm(value))[0]
        return self.out_layer(F.gelu(value, approximate="tanh"))[0]


class QwenImage21Attention(nn.Module):
    def __init__(
        self, config: QwenImage21Config, *, prefix, params_dtype, device
    ) -> None:
        super().__init__()
        self.head_dim = config.attention_head_dim
        self.qkv = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            config.num_attention_heads,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.qkv",
            hf_legs={"q": "to_q", "k": "to_k", "v": "to_v"},
        )
        if config.num_attention_heads % self.qkv.tp_size:
            raise ValueError(
                "attention TP size must divide the number of attention heads"
            )
        self.heads = config.num_attention_heads // self.qkv.tp_size
        self.norm_q = RMSNorm(
            self.head_dim,
            config.eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm_q",
            cast_before_affine=True,
        )
        self.norm_k = RMSNorm(
            self.head_dim,
            config.eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm_k",
            cast_before_affine=True,
        )
        self.out = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            group="attention_tp",
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.to_out.0",
        )
        self.full_attention = Attention(self.heads, self.head_dim, causal=False)
        self.causal_attention = Attention(self.heads, self.head_dim, causal=True)

    def forward(
        self,
        value: torch.Tensor,
        condition: QwenImage21PreparedCondition,
        rope: AxialRotaryEmbedding,
        *,
        prefix_kv: PrefixKV | None,
        return_prefix_kv: bool,
        attn_ctx: AttnCtx | None = None,
    ) -> tuple[torch.Tensor, PrefixKV | None]:
        batch, tokens, _ = value.shape
        query, key, val = self.qkv(value)[0].chunk(3, dim=-1)
        query = self.norm_q(query.reshape(batch, tokens, self.heads, self.head_dim))
        key = self.norm_k(key.reshape(batch, tokens, self.heads, self.head_dim))
        val = val.reshape(batch, tokens, self.heads, self.head_dim)
        start = condition.prefix_length if prefix_kv is not None else 0
        query, key = rope.apply(
            query, key, condition.cos[start:], condition.sin[start:]
        )
        extracted = None
        if return_prefix_kv:
            extracted = (
                key[:, : condition.prefix_length].clone(),
                val[:, : condition.prefix_length].clone(),
            )
        if prefix_kv is not None:
            key = torch.cat((prefix_kv[0], key), dim=1)
            val = torch.cat((prefix_kv[1], val), dim=1)
            mask = (
                AttnMask.from_key_mask(condition.key_valid)
                if condition.key_valid is not None
                else None
            )
            result = self.full_attention(
                query, key, val, ctx=attn_ctx, mask=mask if attn_ctx is None else None
            )
        else:
            outputs = []
            for begin, end, causal in condition.segments:
                validity = (
                    condition.key_valid[:, :end]
                    if condition.key_valid is not None
                    else None
                )
                if validity is not None and causal:
                    # Rectangular causal queries plus padded keys need both
                    # structures; the backend selector lowers this mask.
                    segments = ([(begin, True)] if begin else []) + [
                        (1, True) for _ in range(end - begin)
                    ]
                    mask = AttnMask.from_segments(segments, key_mask=validity)
                    attention = self.full_attention
                else:
                    mask = (
                        AttnMask.from_key_mask(validity)
                        if validity is not None
                        else None
                    )
                    attention = self.causal_attention if causal else self.full_attention
                outputs.append(
                    attention(
                        query[:, begin:end], key[:, :end], val[:, :end], mask=mask
                    )
                )
            result = torch.cat(outputs, dim=1)
        return self.out(result.flatten(2))[0], extracted


class QwenImage21Block(nn.Module):
    def __init__(
        self, config: QwenImage21Config, *, prefix, params_dtype, device
    ) -> None:
        super().__init__()
        self.apply_gated_residual = apply_gated_residual
        self.img_norm1 = LayerNorm(
            config.hidden_size,
            config.eps,
            elementwise_affine=False,
            device=device,
        )
        self.img_norm2 = LayerNorm(
            config.hidden_size,
            config.eps,
            elementwise_affine=False,
            device=device,
        )
        self.attn = QwenImage21Attention(
            config,
            prefix=f"{prefix}.attn",
            params_dtype=params_dtype,
            device=device,
        )
        self.img_mlp = DenseMLP(
            config.hidden_size,
            config.hidden_size * config.mlp_ratio,
            gated=True,
            gated_hf_legs=("gate_layer", "proj"),
            cast_before_multiply=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.img_mlp",
            sequence_parallel=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        modulation: PreparedModulation,
        condition: QwenImage21PreparedCondition,
        rope: AxialRotaryEmbedding,
        *,
        prefix_kv: PrefixKV | None = None,
        return_prefix_kv: bool = False,
        attn_ctx: AttnCtx | None = None,
    ) -> tuple[torch.Tensor, PrefixKV | None]:
        scale1, gate1, scale2, gate2 = modulation
        attn, extracted = self.attn(
            self.img_norm1(hidden) * scale1,
            condition,
            rope,
            prefix_kv=prefix_kv,
            return_prefix_kv=return_prefix_kv,
            attn_ctx=attn_ctx,
        )
        hidden = self.apply_gated_residual(hidden, attn, gate1)
        update = self.img_mlp(self.img_norm2(hidden) * scale2)
        hidden = self.apply_gated_residual(hidden, update, gate2)
        if hidden.dtype == torch.float16:
            hidden = hidden.clamp(-65504, 65504)
        return hidden, extracted


class QwenImage21Transformer(nn.Module):
    def __init__(
        self, config: QwenImage21Config, *, params_dtype=torch.bfloat16, device=None
    ) -> None:
        super().__init__()
        self.config = config
        if device is None:
            device = get_engine_config().device.target
        self.pos_embed = AxialRotaryEmbedding(config.axes_dims_rope, device=device)
        self.time_text_embed = QwenImage21TimeEmbedding(
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
        )
        self.txt_in = QwenImage21TextProjection(
            config, params_dtype=params_dtype, device=device
        )
        self.img_in = ReplicatedLinear(
            config.in_channels,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="img_in",
        )
        self.modulation = ReplicatedLinear(
            config.hidden_size,
            4 * config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="modulation.1",
        )
        self.transformer_blocks = nn.ModuleList(
            QwenImage21Block(
                config,
                prefix=f"transformer_blocks.{i}",
                params_dtype=params_dtype,
                device=device,
            )
            for i in range(config.num_layers)
        )
        self.norm_out = LayerNorm(
            config.hidden_size,
            config.eps,
            elementwise_affine=False,
            device=device,
        )
        self.out_modulation = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="norm_out.linear",
        )
        self.proj_out = ReplicatedLinear(
            config.hidden_size,
            config.out_channels,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix="proj_out",
        )

    def enable_compile(self) -> None:
        """Fuse gated residuals while preserving eager low-precision rounding."""
        options = {"emulate_precision_casts": True, "triton.cudagraphs": False}
        residual = torch.compile(
            apply_gated_residual, fullgraph=True, dynamic=False, options=options
        )
        for block in self.transformer_blocks:
            block.apply_gated_residual = residual

    def prepare_condition(
        self,
        encoder_hidden_states: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
    ) -> QwenImage21PreparedCondition:
        batch, text_tokens, _ = encoder_hidden_states.shape
        if len(img_shapes) != batch or any(
            shapes != img_shapes[0] for shapes in img_shapes
        ):
            raise ValueError("all samples must share the same image layout")
        shapes = img_shapes[0]
        if not shapes or any(t != 1 or h <= 0 or w <= 0 for t, h, w in shapes):
            raise ValueError(
                "image shapes must contain (1, positive height, positive width)"
            )
        target_tokens = math.prod(shapes[-1])
        if target_tokens % 4:
            raise ValueError("target latent token count must be divisible by four")
        if img_mask.shape != (batch, text_tokens + target_tokens // 4):
            raise ValueError(
                "img_mask must include prompt slots and target image slots"
            )
        if not torch.equal(img_mask.bool(), img_mask[0:1].bool().expand_as(img_mask)):
            raise ValueError("all samples must share image slot positions")
        slots = img_mask[0].bool().tolist()
        if not all(slots[text_tokens:]):
            raise ValueError("target image slots must occupy the end of img_mask")
        repeats = torch.tensor(
            [4 if value else 1 for value in slots], device=img_mask.device
        )
        expansion = torch.arange(len(slots), device=img_mask.device).repeat_interleave(
            repeats
        )
        image_mask = img_mask[0].bool()[expansion]
        image_indices = image_mask.nonzero(as_tuple=True)[0]
        if image_indices.numel() != sum(math.prod(shape) for shape in shapes):
            raise ValueError("image shapes do not match the expanded image slots")
        image_positions = image_indices.tolist()
        seq_len = expansion.numel()
        coordinates = []
        segments = []
        cursor, position, image_cursor = 0, 0, 0
        for _, height, width in shapes:
            count = height * width
            begin = image_positions[image_cursor]
            if image_positions[image_cursor : image_cursor + count] != list(
                range(begin, begin + count)
            ):
                raise ValueError("each image must occupy contiguous latent positions")
            if begin > cursor:
                segments.append((cursor, begin, True))
                coordinates.extend(
                    (p, p, p) for p in range(position, position + begin - cursor)
                )
                position += begin - cursor
            segments.append((begin, begin + count, False))
            coordinates.extend(
                (position, h, w)
                for h in range(-(height - height // 2), height // 2)
                for w in range(-(width - width // 2), width // 2)
            )
            position += max(height, width)
            cursor = begin + count
            image_cursor += count
        if cursor != seq_len:
            raise ValueError("target image must be the final sequence block")
        coords = torch.tensor(coordinates, device=img_mask.device)
        cos, sin = self.pos_embed.get_cos_sin(coords)
        prefix_length = seq_len - target_tokens
        target_mask = torch.arange(seq_len, device=img_mask.device) >= prefix_length
        key_valid = None
        if encoder_hidden_states_mask is not None:
            if encoder_hidden_states_mask.shape != (batch, text_tokens):
                raise ValueError("encoder mask must match the prompt embedding shape")
            validity = torch.cat(
                (
                    encoder_hidden_states_mask.bool(),
                    torch.ones(
                        batch,
                        target_tokens // 4,
                        device=img_mask.device,
                        dtype=torch.bool,
                    ),
                ),
                dim=1,
            )[:, expansion]
            validity[:, image_mask] = True
            if not bool(validity.all()):
                key_valid = validity
        return QwenImage21PreparedCondition(
            self.txt_in(encoder_hidden_states),
            expansion,
            image_indices,
            target_mask,
            cos,
            sin,
            tuple(segments),
            key_valid,
            prefix_length,
            target_tokens,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        condition: QwenImage21PreparedCondition,
        *,
        prefix_kv: tuple[PrefixKV, ...] | None = None,
        return_prefix_kv: bool = False,
        attn_ctx: AttnCtx | None = None,
    ) -> QwenImage21ModelOutput:
        if (
            prefix_kv is not None or return_prefix_kv
        ) and not self.config.causal_condition:
            raise ValueError("prefix caching requires causal_condition=True")
        if attn_ctx is not None and prefix_kv is None:
            raise ValueError("a prepared attention context requires cached prefixes")
        if prefix_kv is not None and (
            len(prefix_kv) != len(self.transformer_blocks) or return_prefix_kv
        ):
            raise ValueError(
                "cached calls require one KV pair per layer and cannot extract again"
            )
        expected_tokens = (
            condition.target_tokens
            if prefix_kv is not None
            else condition.image_indices.numel()
        )
        if hidden_states.shape[:2] != (condition.text.shape[0], expected_tokens):
            raise ValueError(
                "latent batch or token count does not match the prepared condition"
            )
        hidden = self.img_in(hidden_states)[0]
        batch = hidden.shape[0]
        if prefix_kv is None:
            text = torch.cat(
                (
                    condition.text,
                    condition.text.new_zeros(
                        batch, condition.target_tokens // 4, self.config.hidden_size
                    ),
                ),
                dim=1,
            )
            joint = text[:, condition.expansion_indices].clone()
            joint[:, condition.image_indices] = hidden
            hidden = joint
        timestep = timestep.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch)
        if timestep.numel() != batch:
            raise ValueError("timestep must be a scalar or one value per sample")
        target_mask = None
        if self.config.causal_condition:
            timestep = torch.cat((timestep, timestep.new_zeros(1)))
            target_mask = condition.target_mask
            if prefix_kv is not None:
                target_mask = target_mask[condition.prefix_length :]
        temb = self.time_text_embed(timestep)
        modulation = self.modulation(F.silu(temb))[0]
        prepared_modulation = prepare_modulation(
            modulation, target_mask, cached_target=prefix_kv is not None
        )
        extracted = []
        for index, block in enumerate(self.transformer_blocks):
            hidden, layer_kv = block(
                hidden,
                prepared_modulation,
                condition,
                self.pos_embed,
                prefix_kv=prefix_kv[index] if prefix_kv is not None else None,
                return_prefix_kv=return_prefix_kv,
                attn_ctx=attn_ctx,
            )
            if layer_kv is not None:
                extracted.append(layer_kv)
        scale = self.out_modulation(F.silu(temb))[0]
        scale = (
            scale[:-1, None]
            if prefix_kv is not None
            else select_modulation(scale, target_mask)
        )
        output = self.proj_out(self.norm_out(hidden) * (1 + scale))[0]
        return QwenImage21ModelOutput(
            output, tuple(extracted) if return_prefix_kv else None
        )


def qwen_image_21_weight_remap(name: str) -> str | None:
    return name.replace(".img_mlp.out.", ".img_mlp.down_proj.")
