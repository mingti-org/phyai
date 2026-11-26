"""DreamZero Wan2.1 UMT5 text encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from phyai.engine_config import ParallelConfig, get_engine_config
from phyai.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from phyai.layers import RMSNorm, T5GELU
from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.vocab_embedding import VocabParallelEmbedding
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroTextEncoderConfig
from phyai.parallel.layout import build_rank_layout
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import resolve_mesh


def dreamzero_text_encoder_weight_remap(name: str) -> str | None:
    """Map DreamZero checkpoint keys to PHYAI text-encoder parameter keys."""
    if name.startswith("action_head.text_encoder."):
        return name.removeprefix("action_head.text_encoder.")
    if name.startswith("text_encoder."):
        return name.removeprefix("text_encoder.")
    if name.startswith(("token_embedding.", "blocks.", "norm.", "pos_embedding.")):
        return name
    return None


class T5RelativeEmbedding(nn.Module):
    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        max_dist: int = 128,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_dist = max_dist
        self.embedding = VocabParallelEmbedding(
            num_buckets,
            num_heads,
            padding_multiple=1,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.embedding",
        )

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(
            lq, device=device
        ).unsqueeze(1)
        rel_pos = self.relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        return rel_pos_embeds.permute(2, 0, 1).unsqueeze(0).contiguous()

    def relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets // 2
        rel_buckets = (rel_pos > 0).long() * num_buckets
        rel_pos = torch.abs(rel_pos)

        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact)
                / math.log(self.max_dist / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(
            rel_pos_large,
            torch.full_like(rel_pos_large, num_buckets - 1),
        )
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        num_heads: int,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
        prefix: str,
    ) -> None:
        if dim_attn % num_heads != 0:
            raise ValueError(
                f"dim_attn={dim_attn} must be divisible by num_heads={num_heads}."
            )
        super().__init__()
        mesh = resolve_mesh(mesh)
        tp_size = mesh.group_size("dense_tp")
        if num_heads % tp_size:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by tp_size={tp_size}."
            )
        self.num_heads = num_heads // tp_size
        self.head_dim = dim_attn // num_heads
        self.head_start = mesh.group_rank("dense_tp") * self.num_heads
        params_dtype = (
            torch.get_default_dtype() if params_dtype is None else params_dtype
        )
        self.qkv = QKVParallelLinear(
            dim,
            self.head_dim,
            num_heads,
            bias=False,
            group="dense_tp",
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            hf_legs={"q": "q", "k": "k", "v": "v"},
            prefix=f"{prefix}.qkv",
        )
        self.o = RowParallelLinear(
            dim_attn,
            dim,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.o",
        )
        self.attention = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=1.0,
            causal=False,
            kernel_role="dreamzero.text_attention",
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        q, k, v = self.qkv(x)[0].chunk(3, dim=-1)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)

        attn_bias = x.new_zeros(
            batch_size,
            self.num_heads,
            q.size(1),
            k.size(1),
        )
        if pos_bias is not None:
            attn_bias += pos_bias[:, self.head_start : self.head_start + self.num_heads]
        if mask is not None:
            if mask.ndim not in {2, 3}:
                raise ValueError(
                    f"mask must be 2-D or 3-D; got shape {tuple(mask.shape)}."
                )
            mask = (
                mask.view(batch_size, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            )
            attn_bias.masked_fill_(
                mask == 0,
                torch.finfo(x.dtype).min,
            )

        out = self.attention(q, k, v, attn_bias=attn_bias)
        out = out.reshape(batch_size, -1, self.num_heads * self.head_dim)
        out, _ = self.o(out)
        return out


class T5FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_ffn: int,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
        prefix: str,
    ) -> None:
        super().__init__()
        params_dtype = (
            torch.get_default_dtype() if params_dtype is None else params_dtype
        )
        self.gate_up = MergedColumnParallelLinear(
            dim,
            [dim_ffn, dim_ffn],
            bias=False,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            hf_legs=("gate.0", "fc1"),
            prefix=f"{prefix}.gate_up",
        )
        self.activation = T5GELU()
        self.fc2 = RowParallelLinear(
            dim_ffn,
            dim,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, hidden = self.gate_up(x)[0].chunk(2, dim=-1)
        x = hidden * self.activation(gate)
        x, _ = self.fc2(x)
        return x


class T5SelfAttention(nn.Module):
    def __init__(
        self,
        config: DreamZeroTextEncoderConfig,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
        position_mesh: Mesh | str = "model",
        prefix: str,
    ) -> None:
        super().__init__()
        self.shared_pos = config.shared_pos
        # UMT5 can overflow FP16 residuals; BF16 needs no saturation.
        self.residual_clamp = (
            nn.Hardtanh(-64504.0, 64504.0)
            if params_dtype == torch.float16
            else nn.Identity()
        )
        self.norm1 = RMSNorm(
            config.dim,
            eps=1e-6,
            dtype=params_dtype,
            device=device,
            cast_before_affine=True,
            prefix=f"{prefix}.norm1",
        )
        self.attn = T5Attention(
            config.dim,
            config.dim_attn,
            config.num_heads,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = RMSNorm(
            config.dim,
            eps=1e-6,
            dtype=params_dtype,
            device=device,
            cast_before_affine=True,
            prefix=f"{prefix}.norm2",
        )
        self.ffn = T5FeedForward(
            config.dim,
            config.dim_ffn,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.ffn",
        )
        self.pos_embedding = (
            None
            if config.shared_pos
            else T5RelativeEmbedding(
                config.num_buckets,
                config.num_heads,
                params_dtype=params_dtype,
                device=device,
                mesh=position_mesh,
                prefix=f"{prefix}.pos_embedding",
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.shared_pos:
            e = pos_bias
        else:
            if self.pos_embedding is None:
                raise RuntimeError("missing block-local position embedding.")
            e = self.pos_embedding(x.size(1), x.size(1))
        x = self.residual_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        return self.residual_clamp(x + self.ffn(self.norm2(x)))


class DreamZeroWanTextEncoder(nn.Module):
    """Wan2.1 UMT5 encoder used by DreamZero prompts."""

    def __init__(
        self,
        config: DreamZeroTextEncoderConfig | None = None,
        *,
        params_dtype: torch.dtype | None = torch.float32,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
    ) -> None:
        super().__init__()
        self.config = config or DreamZeroTextEncoderConfig()
        params_dtype = params_dtype or get_engine_config().device.params_dtype
        # Small relative-bias tables are replicated; attention slices local heads.
        position_mesh = Mesh(
            build_rank_layout(ParallelConfig()), name="dreamzero_position_bias"
        )
        self.token_embedding = VocabParallelEmbedding(
            self.config.vocab,
            self.config.dim,
            padding_multiple=1,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix="token_embedding",
        )
        self.pos_embedding = (
            T5RelativeEmbedding(
                self.config.num_buckets,
                self.config.num_heads,
                params_dtype=params_dtype,
                device=device,
                mesh=position_mesh,
                prefix="pos_embedding",
            )
            if self.config.shared_pos
            else None
        )
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    self.config,
                    params_dtype=params_dtype,
                    device=device,
                    mesh=mesh,
                    position_mesh=position_mesh,
                    prefix=f"blocks.{index}",
                )
                for index in range(self.config.num_layers)
            ]
        )
        self.norm = RMSNorm(
            self.config.dim,
            eps=1e-6,
            dtype=params_dtype,
            device=device,
            cast_before_affine=True,
            prefix="norm",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D (B, L); got shape {tuple(input_ids.shape)}."
            )
        x = self.token_embedding(input_ids)
        pos_bias = (
            self.pos_embedding(x.size(1), x.size(1))
            if self.pos_embedding is not None
            else None
        )
        for block in self.blocks:
            x = block(x, mask=attention_mask, pos_bias=pos_bias)
        return self.norm(x)


__all__ = [
    "DreamZeroWanTextEncoder",
    "T5Attention",
    "T5FeedForward",
    "T5RelativeEmbedding",
    "T5SelfAttention",
    "dreamzero_text_encoder_weight_remap",
]
