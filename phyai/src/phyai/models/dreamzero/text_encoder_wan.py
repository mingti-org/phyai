"""DreamZero Wan2.1 UMT5 text encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.models.dreamzero.configuration_dreamzero import DreamZeroTextEncoderConfig
from phyai.weights.shards import replicated


def attach_replicated_text_encoder_weights(module: nn.Module) -> None:
    """Attach phyai weight-loader metadata to all text encoder parameters."""
    for name, param in module.named_parameters():
        param.hf_keys = [(name, None)]
        param.weight_loader = replicated()


def dreamzero_text_encoder_weight_remap(name: str) -> str | None:
    """Map DreamZero checkpoint keys to PHYAI text-encoder parameter keys."""
    if name.startswith("action_head.text_encoder."):
        return name.removeprefix("action_head.text_encoder.")
    if name.startswith("text_encoder."):
        return name.removeprefix("text_encoder.")
    if name.startswith(("token_embedding.", "blocks.", "norm.", "pos_embedding.")):
        return name
    return None


def fp16_clamp(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class T5GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class T5LayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=params_dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in {torch.float16, torch.bfloat16}:
            x = x.type_as(self.weight)
        return self.weight * x


class T5RelativeEmbedding(nn.Module):
    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        bidirectional: bool,
        max_dist: int = 128,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(
            num_buckets,
            num_heads,
            dtype=params_dtype,
            device=device,
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
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

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
        dropout: float = 0.0,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if dim_attn % num_heads != 0:
            raise ValueError(
                f"dim_attn={dim_attn} must be divisible by num_heads={num_heads}."
            )
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False, dtype=params_dtype, device=device)
        self.k = nn.Linear(dim, dim_attn, bias=False, dtype=params_dtype, device=device)
        self.v = nn.Linear(dim, dim_attn, bias=False, dtype=params_dtype, device=device)
        self.o = nn.Linear(dim_attn, dim, bias=False, dtype=params_dtype, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        batch_size = x.size(0)
        q = self.q(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k(context).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v(context).view(batch_size, -1, self.num_heads, self.head_dim)

        attn_bias = x.new_zeros(
            batch_size,
            self.num_heads,
            q.size(1),
            k.size(1),
        )
        if pos_bias is not None:
            attn_bias += pos_bias
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

        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        out = torch.einsum("bnij,bjnc->binc", attn, v)
        out = out.reshape(batch_size, -1, self.dim_attn)
        return self.dropout(self.o(out))


class T5FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_ffn: int,
        dropout: float = 0.0,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.gate = nn.Sequential(
            nn.Linear(dim, dim_ffn, bias=False, dtype=params_dtype, device=device),
            T5GELU(),
        )
        self.fc1 = nn.Linear(
            dim, dim_ffn, bias=False, dtype=params_dtype, device=device
        )
        self.fc2 = nn.Linear(
            dim_ffn, dim, bias=False, dtype=params_dtype, device=device
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class T5SelfAttention(nn.Module):
    def __init__(
        self,
        config: DreamZeroTextEncoderConfig,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.shared_pos = config.shared_pos
        self.norm1 = T5LayerNorm(config.dim, params_dtype=params_dtype, device=device)
        self.attn = T5Attention(
            config.dim,
            config.dim_attn,
            config.num_heads,
            config.dropout,
            params_dtype=params_dtype,
            device=device,
        )
        self.norm2 = T5LayerNorm(config.dim, params_dtype=params_dtype, device=device)
        self.ffn = T5FeedForward(
            config.dim,
            config.dim_ffn,
            config.dropout,
            params_dtype=params_dtype,
            device=device,
        )
        self.pos_embedding = (
            None
            if config.shared_pos
            else T5RelativeEmbedding(
                config.num_buckets,
                config.num_heads,
                bidirectional=True,
                params_dtype=params_dtype,
                device=device,
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
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        return fp16_clamp(x + self.ffn(self.norm2(x)))


def init_weights(module: nn.Module) -> None:
    if isinstance(module, T5LayerNorm):
        nn.init.ones_(module.weight)
    elif isinstance(module, T5FeedForward):
        nn.init.normal_(module.gate[0].weight, std=module.dim**-0.5)
        nn.init.normal_(module.fc1.weight, std=module.dim**-0.5)
        nn.init.normal_(module.fc2.weight, std=module.dim_ffn**-0.5)
    elif isinstance(module, T5Attention):
        nn.init.normal_(module.q.weight, std=(module.dim * module.dim_attn) ** -0.5)
        nn.init.normal_(module.k.weight, std=module.dim**-0.5)
        nn.init.normal_(module.v.weight, std=module.dim**-0.5)
        nn.init.normal_(
            module.o.weight,
            std=(module.num_heads * module.dim_attn) ** -0.5,
        )
    elif isinstance(module, T5RelativeEmbedding):
        nn.init.normal_(
            module.embedding.weight,
            std=(2 * module.num_buckets * module.num_heads) ** -0.5,
        )


class DreamZeroWanTextEncoder(nn.Module):
    """Wan2.1 UMT5 encoder used by DreamZero prompts."""

    def __init__(
        self,
        config: DreamZeroTextEncoderConfig | None = None,
        *,
        params_dtype: torch.dtype | None = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DreamZeroTextEncoderConfig()
        self.token_embedding = nn.Embedding(
            self.config.vocab,
            self.config.dim,
            dtype=params_dtype,
            device=device,
        )
        self.pos_embedding = (
            T5RelativeEmbedding(
                self.config.num_buckets,
                self.config.num_heads,
                bidirectional=True,
                params_dtype=params_dtype,
                device=device,
            )
            if self.config.shared_pos
            else None
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    self.config,
                    params_dtype=params_dtype,
                    device=device,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.norm = T5LayerNorm(
            self.config.dim,
            params_dtype=params_dtype,
            device=device,
        )
        self.apply(init_weights)
        attach_replicated_text_encoder_weights(self)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D (B, L); got shape {tuple(input_ids.shape)}."
            )
        x = self.dropout(self.token_embedding(input_ids))
        pos_bias = (
            self.pos_embedding(x.size(1), x.size(1))
            if self.pos_embedding is not None
            else None
        )
        for block in self.blocks:
            x = block(x, mask=attention_mask, pos_bias=pos_bias)
        x = self.norm(x)
        return self.dropout(x)


__all__ = [
    "DreamZeroWanTextEncoder",
    "T5Attention",
    "T5FeedForward",
    "T5LayerNorm",
    "T5RelativeEmbedding",
    "T5SelfAttention",
    "dreamzero_text_encoder_weight_remap",
]
