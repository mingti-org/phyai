# Copyright 2026 The LoongForge Authors.
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge stage-one RL-token encoder and shifted reconstruction decoder."""

from __future__ import annotations

import math

import torch
from torch import nn

from phyai.engine_config import get_engine_config
from phyai.layers.attention import (
    Attention,
    AttnCtx,
    AttnLayout,
    AttnMask,
    AttnMode,
    SdpaAttentionBackend,
    SdpaAttentionPlan,
)
from phyai.layers.layer_norm import LayerNorm
from phyai.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.mlp import DenseMLP
from phyai.weights.shards import replicated


def _packed_loader(parameter, key, shape, legs):
    """Retain native packed checkpoint keys while reusing placement loaders."""
    load_leg = parameter.weight_loader

    def load(param, value, shard_id=None):
        if tuple(value.shape) != shape:
            raise ValueError(f"{key}: expected {shape}, got {tuple(value.shape)}")
        for piece, leg in zip(value.chunk(len(legs), dim=0), legs, strict=True):
            load_leg(param, piece, leg)

    parameter.hf_keys = [(key, None)]
    parameter.weight_loader = load


class PI05RLTBlock(nn.Module):
    """Pre-LayerNorm attention and exact GeGLU, with no dropout or final norm."""

    def __init__(
        self, width, num_heads, mlp_ratio, *, params_dtype, device, prefix, causal
    ):
        super().__init__()
        hidden = int(width * mlp_ratio)
        self.head_dim = width // num_heads
        self.attention_norm = LayerNorm(
            width,
            eps=1e-5,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.attention_norm",
        )
        self.qkv = QKVParallelLinear(
            width,
            self.head_dim,
            num_heads,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.attention.qkv",
        )
        self.local_heads = self.qkv.output_partition_sizes[0] // self.head_dim
        self.attention = Attention(
            self.local_heads,
            self.head_dim,
            causal=causal,
            backend="sdpa",
        )
        # Masked FlashInfer packing is dynamic; this dense backend supports capture.
        self.attention_backend = SdpaAttentionBackend()
        self.attention_plan = SdpaAttentionPlan()
        self.out_proj = RowParallelLinear(
            width,
            width,
            group="attention_tp",
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.attention.out_proj",
        )
        self.mlp_norm = LayerNorm(
            width,
            eps=1e-5,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp_norm",
        )
        self.mlp = DenseMLP(
            width,
            hidden,
            activation="gelu",
            gated=True,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp",
            cast_before_multiply=True,
        )
        for suffix in ("weight", "bias"):
            _packed_loader(
                getattr(self.qkv, suffix),
                f"{prefix}.attention.in_proj_{suffix}",
                (3 * width, width) if suffix == "weight" else (3 * width,),
                ("q", "k", "v"),
            )
            # The source stores value then gate; DenseMLP stores gate then value.
            _packed_loader(
                getattr(self.mlp.gate_up_proj, suffix),
                f"{prefix}.mlp_in.{suffix}",
                (2 * hidden, width) if suffix == "weight" else (2 * hidden,),
                (1, 0),
            )
            getattr(self.mlp.down_proj, suffix).hf_keys = [
                (f"{prefix}.mlp_out.{suffix}", None)
            ]

    def forward(self, x, valid):
        batch, tokens, _ = x.shape
        projected = self.qkv(self.attention_norm(x))[0]
        q, k, v = projected.split(self.qkv.output_partition_sizes, dim=-1)
        shape = (batch, tokens, self.local_heads, self.head_dim)
        ctx = AttnCtx(
            backend=self.attention_backend,
            plan=self.attention_plan,
            mode=AttnMode.PREFILL,
            layout=AttnLayout.PADDED_4D,
            mask=AttnMask.from_key_mask(valid),
        )
        attended = self.attention(
            q.reshape(shape), k.reshape(shape), v.reshape(shape), ctx
        )
        x = x + self.out_proj(attended.reshape(batch, tokens, -1))[0]
        return x + self.mlp(self.mlp_norm(x))


class PI05RLTTokenAutoencoder(nn.Module):
    """Compress final VLM visual states into one token without updating the VLM.

    Parameters and checkpoint names follow LoongForge's ``rlt_token`` sidecar.
    The decoder is loaded with the encoder but is not needed for token inference.
    """

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        embed_dim: int = 2048,
        max_visual_tokens: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        params_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        prefix: str = "rlt_token",
    ):
        super().__init__()
        for name, value in {
            "input_dim": input_dim,
            "embed_dim": embed_dim,
            "max_visual_tokens": max_visual_tokens,
            "num_layers": num_layers,
            "num_heads": num_heads,
        }.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not math.isfinite(mlp_ratio) or int(embed_dim * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce a positive finite hidden width")
        if params_dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("RLT parameters require a supported floating dtype")
        device = get_engine_config().device.target if device is None else device
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.max_visual_tokens = max_visual_tokens
        for name in ("encoder_input", "decoder_input"):
            setattr(
                self,
                name,
                None
                if input_dim == embed_dim
                else ReplicatedLinear(
                    input_dim,
                    embed_dim,
                    bias=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.{name}",
                ),
            )
        for name, length in (
            ("rl_token", 1),
            ("encoder_positions", max_visual_tokens + 1),
            ("decoder_positions", max_visual_tokens),
        ):
            # Learned positions must be loaded, never regenerated at inference.
            parameter = nn.Parameter(
                torch.empty(length, embed_dim, dtype=params_dtype, device=device),
                requires_grad=False,
            )
            parameter.hf_keys = [(f"{prefix}.{name}", None)]
            parameter.weight_loader = replicated()
            setattr(self, name, parameter)
        for name, causal in (("encoder_layers", False), ("decoder_layers", True)):
            setattr(
                self,
                name,
                nn.ModuleList(
                    PI05RLTBlock(
                        embed_dim,
                        num_heads,
                        mlp_ratio,
                        params_dtype=params_dtype,
                        device=device,
                        prefix=f"{prefix}.{name}.{i}",
                        causal=causal,
                    )
                    for i in range(num_layers)
                ),
            )
        self.output_projection = ReplicatedLinear(
            embed_dim,
            input_dim,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.output_projection",
        )

    def _prepare_features(self, features, mask):
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features must have shape [B, M, {self.input_dim}]")
        if (
            features.shape[0] == 0
            or not 0 < features.shape[1] <= self.max_visual_tokens
        ):
            raise ValueError(
                "features must contain a nonempty batch within visual capacity"
            )
        if not features.is_floating_point():
            raise TypeError("features must have a floating-point dtype")
        if features.device != self.rl_token.device:
            raise ValueError("features and sidecar must be on the same device")
        if mask is None:
            mask = torch.ones(
                features.shape[:2], device=features.device, dtype=torch.bool
            )
        elif mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B, M]")
        elif mask.device != features.device:
            raise ValueError("features and mask must be on the same device")
        frozen = features.detach().to(dtype=self.rl_token.dtype)
        return frozen.masked_fill(~mask[..., None], 0), mask

    def _encode(self, frozen, mask):
        batch, length = frozen.shape[:2]
        visual = frozen if self.encoder_input is None else self.encoder_input(frozen)[0]
        visual = visual + self.encoder_positions[:length]
        rl = self.rl_token + self.encoder_positions[-1:]
        x = torch.cat([visual, rl[None].expand(batch, -1, -1)], dim=1)
        valid = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        for layer in self.encoder_layers:
            x = layer(x, valid)
        return x[:, -1:]

    def encode(self, features, mask=None):
        frozen, mask = self._prepare_features(features, mask)
        return self._encode(frozen, mask)

    def decode(self, rl_token, features, mask=None):
        frozen, mask = self._prepare_features(features, mask)
        if rl_token.shape != (frozen.shape[0], 1, self.embed_dim):
            raise ValueError(f"rl_token must have shape [B, 1, {self.embed_dim}]")
        if rl_token.device != frozen.device:
            raise ValueError("rl_token and features must be on the same device")
        shifted = frozen[:, :-1]
        if self.decoder_input is not None:
            shifted = self.decoder_input(shifted)[0]
        x = torch.cat([rl_token.to(dtype=shifted.dtype), shifted], dim=1)
        x = x + self.decoder_positions[: frozen.shape[1]]
        valid = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=1)
        for layer in self.decoder_layers:
            x = layer(x, valid)
        return self.output_projection(x)[0]

    def forward(self, features, mask=None):
        frozen, mask = self._prepare_features(features, mask)
        token = self._encode(frozen, mask)
        prediction = self.decode(token, frozen, mask)
        error = (prediction.float() - frozen.float()).masked_fill(~mask[..., None], 0)
        return error.square().sum() / (mask.sum() * self.input_dim).clamp_min(1)


__all__ = ["PI05RLTTokenAutoencoder"]
