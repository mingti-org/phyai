"""Inference-only UNIS decoder components with optional paged attention."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.attention.paged.base import PagedAttnCtx
from phyai.layers.attention.paged.layer import PagedAttention
from phyai.layers.layer_norm import RMSNorm
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.rotary_embedding import RotaryEmbedding
from phyai.models.xiaomi_u0.configuration_unis import UNISConfig


class UNISRMSNorm(RMSNorm):
    """Preserve UNIS's cast-before-weight-multiply semantics."""

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize in FP32, cast back before weight multiplication and reject fused residuals."""
        if residual is not None:
            raise ValueError("UNIS requires explicit residual addition.")
        promoted = x.float()
        variance = promoted.pow(2).mean(dim=-1, keepdim=True)
        normalized = promoted * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(x.dtype)


class UNISAttention(nn.Module):
    """Causal self-attention with runner-supplied optional paged context."""

    def __init__(
        self,
        config: UNISConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
        layer_id: int,
        attention_backend: str | None = None,
        rope_backend: str | None = None,
    ) -> None:
        """Build attention projections, per-head Q/K norms and positional embeddings."""
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        for name, width in (
            ("q_proj", self.num_heads * self.head_dim),
            ("k_proj", self.num_kv_heads * self.head_dim),
            ("v_proj", self.num_kv_heads * self.head_dim),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    config.hidden_size,
                    width,
                    bias=False,
                    params_dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.o_proj = ReplicatedLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = UNISRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.q_norm",
        )
        self.k_norm = UNISRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.k_norm",
        )
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            interleave=False,
            backend=rope_backend,
            device=device,
        )
        self.attention = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            causal=True,
            backend=attention_backend,
        )
        self.paged_attention = PagedAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            causal=True,
            backend="flashinfer",
            kernel_role="u0_ar",
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        ctx: PagedAttnCtx | None = None,
    ) -> torch.Tensor:
        """Apply normalized rotary Q/K attention, using the supplied paged context if present."""
        batch, length, _ = x.shape
        if ctx is not None and batch != 1:
            raise ValueError("Initial paged UNIS supports one sequence only.")
        q, _ = self.q_proj(x)
        k, _ = self.k_proj(x)
        v, _ = self.v_proj(x)
        q = self.q_norm(q.reshape(batch, length, self.num_heads, self.head_dim))
        k = self.k_norm(k.reshape(batch, length, self.num_kv_heads, self.head_dim))
        v = v.reshape(batch, length, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(position_ids, q, k)
        if ctx is None:
            output = self.attention(q, k, v)
        else:
            output = self.paged_attention(
                q.reshape(length, self.num_heads, self.head_dim),
                k.reshape(length, self.num_kv_heads, self.head_dim),
                v.reshape(length, self.num_kv_heads, self.head_dim),
                ctx,
            )
        output, _ = self.o_proj(
            output.reshape(batch, length, self.num_heads * self.head_dim)
        )
        return output


class UNISMLP(nn.Module):
    """UNIS SiLU-gated feed-forward block."""

    def __init__(
        self,
        config: UNISConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        """Create the bias-free gate, expansion and output projections."""
        super().__init__()
        for name, input_size, output_size in (
            ("gate_proj", config.hidden_size, config.intermediate_size),
            ("up_proj", config.hidden_size, config.intermediate_size),
            ("down_proj", config.intermediate_size, config.hidden_size),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_size,
                    output_size,
                    bias=False,
                    params_dtype=dtype,
                    device=device,
                    prefix=f"{prefix}.{name}",
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the down projection of SiLU(gate(x)) multiplied by up(x)."""
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class UNISDecoderLayer(nn.Module):
    """One UNIS inference layer with explicit residual boundaries."""

    def __init__(
        self,
        config: UNISConfig,
        *,
        layer_idx: int,
        dtype: torch.dtype,
        device: torch.device | str,
        attention_backend: str | None = None,
        rope_backend: str | None = None,
    ) -> None:
        """Build one indexed decoder layer with separate attention and MLP norms."""
        super().__init__()
        if not 0 <= layer_idx < config.num_hidden_layers:
            raise ValueError("layer_idx is outside the configured layer range.")
        prefix = f"model.layers.{layer_idx}"
        self.hidden_size = config.hidden_size
        self.input_layernorm = UNISRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.input_layernorm",
        )
        self.self_attn = UNISAttention(
            config,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.self_attn",
            layer_id=layer_idx,
            attention_backend=attention_backend,
            rope_backend=rope_backend,
        )
        self.post_attention_layernorm = UNISRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.post_attention_layernorm",
        )
        self.mlp = UNISMLP(
            config,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        ctx: PagedAttnCtx | None = None,
    ) -> torch.Tensor:
        """Run the decoder in eval mode with explicit attention and MLP residual additions."""
        if self.training:
            raise RuntimeError("UNISDecoderLayer currently supports eval mode only.")
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape (batch, sequence, hidden).")
        x = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_ids, ctx
        )
        return x + self.mlp(self.post_attention_layernorm(x))
