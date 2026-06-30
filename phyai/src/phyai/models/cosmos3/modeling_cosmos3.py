"""Cosmos3 MoT flow matching transformer
paper: https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config, resolve_engine_defaults
from phyai.layers.attention.attention.layer import Attention
from phyai.layers.layer_norm import RMSNorm
from phyai.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.mlp.dense_mlp import DenseMLP
from phyai.layers.rotary_embedding import InterleavedMRotaryEmbedding, rotate_half
from phyai.layers.vocab_embedding import VocabParallelEmbedding
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3Config
from phyai.weights.shards import replicated


def compute_mrope_position_ids_text(
    num_tokens: int, temporal_offset: int
) -> tuple[torch.Tensor, int]:
    """Text tokens: all three axes share a monotonically increasing id."""
    ids = torch.arange(num_tokens, dtype=torch.long) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    enable_fps_modulation: bool = True,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """3-D (t, h, w) ids for vision tokens; spatial ids reset per frame.

    With fps modulation, the temporal axis is rescaled by
    ``base_fps/base_tcf`` over ``fps/tcf`` so videos at different frame rates
    share a comparable temporal coordinate. Flattened t-major.
    """
    fps_modulation = enable_fps_modulation and fps is not None
    if fps_modulation:
        tps = fps / temporal_compression_factor
        effective_base_tcf = (
            base_temporal_compression_factor
            if base_temporal_compression_factor is not None
            else temporal_compression_factor
        )
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = (
            ((frame_indices + start_frame_offset) / tps * base_tps + temporal_offset)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
        )
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
            + int(temporal_offset)
            + start_frame_offset
        )
    h_index = (
        torch.arange(grid_h, dtype=torch.long)
        .view(1, -1, 1)
        .expand(grid_t, -1, grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(grid_w, dtype=torch.long)
        .view(1, 1, -1)
        .expand(grid_t, grid_h, -1)
        .flatten()
    )
    if fps_modulation:
        mrope_ids = torch.stack(
            [t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0
        )
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)
    next_offset = math.floor(mrope_ids.max().item()) + 1
    return mrope_ids, next_offset


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x * cos) + (rotate_half(x) * sin)``. cos/sin broadcast as ``[B,S,1,D]``."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def local_head_counts(
    tp_size: int, num_attention_heads: int, num_key_value_heads: int
) -> tuple[int, int]:
    """Per-rank ``(num_q_heads, num_kv_heads)`` for a tensor-parallel attention block."""
    if num_attention_heads % tp_size != 0:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} not divisible by "
            f"tp_size={tp_size}."
        )
    if num_key_value_heads % tp_size != 0:
        raise ValueError(
            f"num_key_value_heads={num_key_value_heads} not divisible by "
            f"tp_size={tp_size}: cosmos3 uses separate K/V projections that cannot "
            f"replicate KV heads across ranks, so tensor parallelism requires "
            f"tp_size <= num_key_value_heads with an even split."
        )
    return num_attention_heads // tp_size, num_key_value_heads // tp_size


class TimestepEmbedder(nn.Module):
    """Runs in fp32 for precision."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if device is None:
            device = get_engine_config().device.target
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        self.linear_1 = ReplicatedLinear(
            frequency_embedding_size,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear_1" if prefix else "",
        )
        self.linear_2 = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear_2" if prefix else "",
        )
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=device)
            / half
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_freq = t_freq.to(self.linear_1.weight.dtype)
        h, _ = self.linear_1(t_freq)
        h = F.silu(h)
        out, _ = self.linear_2(h)
        return out


class DomainAwareLinear(nn.Module):
    """Per-embodiment-domain linear: one ``(out, in)`` weight + ``out`` bias per domain.

    The Cosmos3 action adapters condition on the robot embodiment ``domain_id``;
    weights/biases are stored as ``nn.Embedding(num_domains, ...)`` tables and a
    per-sample ``bmm`` selects the row. Checkpoint leaves are ``fc.weight``
    ``[num_domains, out*in]`` and ``bias.weight`` ``[num_domains, out]`` — the flat
    ``fc`` row is viewed ``(in, out)`` so ``x @ W`` applies it.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_domains: int,
        *,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if device is None:
            device = get_engine_config().device.target
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Parameter(
            torch.zeros(
                num_domains, output_size * input_size, dtype=params_dtype, device=device
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(num_domains, output_size, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.fc.hf_keys = [(f"{prefix}.fc.weight", None)]
            self.fc.weight_loader = replicated()
            self.bias.hf_keys = [(f"{prefix}.bias.weight", None)]
            self.bias.weight_loader = replicated()

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        w = self.fc[domain_id].view(b, self.input_size, self.output_size)
        bias = self.bias[domain_id]  # [B, out]
        if x.dim() == 2:  # [B, in]
            return torch.bmm(x.unsqueeze(1), w).squeeze(1) + bias
        # [B, T, in] -> [B, T, out]
        return torch.bmm(x, w) + bias.unsqueeze(1)


class Cosmos3CausalAttention(nn.Module):
    """UND pathway: causal self-attention; returns ``(out, K, V)``"""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attn_backend: str,
        norm_backend: str,
        params_dtype: torch.dtype | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        # Fused Q/K/V column-parallel projection: one matmul (one hidden read,
        # one kernel) instead of three separate to_q/to_k/to_v. The separate
        # checkpoint leaves (to_q/to_k/to_v) still load via the fused loader's
        # per-leg hf_legs, so the weight remap is unchanged.
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            bias=False,
            gather_output=False,
            params_dtype=params_dtype,
            hf_legs={"q": "to_q", "k": "to_k", "v": "to_v"},
            prefix=f"{prefix}.qkv_proj" if prefix else "",
        )
        self.to_out = RowParallelLinear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.to_out" if prefix else "",
        )
        # Per-rank Q/K/V widths come straight off the fused layer's partition
        # sizes (it already encodes the head split + any GQA replication).
        self.q_size, self.k_size, self.v_size = self.qkv_proj.output_partition_sizes
        self.num_local_heads = self.q_size // head_dim
        self.num_local_kv_heads = self.k_size // head_dim
        self.norm_q = RMSNorm(
            head_dim,
            eps=rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm_q" if prefix else "",
        )
        self.norm_k = RMSNorm(
            head_dim,
            eps=rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm_k" if prefix else "",
        )
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_local_kv_heads,
            causal=True,
            backend=attn_backend,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q = q.view(B, S, self.num_local_heads, self.head_dim)
        k = k.view(B, S, self.num_local_kv_heads, self.head_dim)
        v = v.view(B, S, self.num_local_kv_heads, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        out = self.attn(q, k, v)  # [B, S, H, D]  (4-D padded causal path)
        out = out.reshape(B, S, -1)
        out, _ = self.to_out(out)
        return out, k, v


class Cosmos3CrossAttention(nn.Module):
    """GEN pathway: visual Q attends to ``cat([K_und, K_gen])``"""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attn_backend: str,
        norm_backend: str,
        params_dtype: torch.dtype | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        # Fused Q/K/V column-parallel projection (see Cosmos3CausalAttention).
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            bias=False,
            gather_output=False,
            params_dtype=params_dtype,
            hf_legs={"q": "to_q", "k": "to_k", "v": "to_v"},
            prefix=f"{prefix}.qkv_proj" if prefix else "",
        )
        self.to_out = RowParallelLinear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.to_out" if prefix else "",
        )
        # Per-rank Q/K/V widths from the fused layer; the cached UND K/V are sharded
        # identically, so the cross-attention concat stays aligned.
        self.q_size, self.k_size, self.v_size = self.qkv_proj.output_partition_sizes
        self.num_local_heads = self.q_size // head_dim
        self.num_local_kv_heads = self.k_size // head_dim
        self.norm_q = RMSNorm(
            head_dim,
            eps=rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm_q" if prefix else "",
        )
        self.norm_k = RMSNorm(
            head_dim,
            eps=rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm_k" if prefix else "",
        )
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_dim=head_dim,
            num_kv_heads=self.num_local_kv_heads,
            causal=False,
            backend=attn_backend,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, S_gen, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        q = q.view(B, S_gen, self.num_local_heads, self.head_dim)
        k = k.view(B, S_gen, self.num_local_kv_heads, self.head_dim)
        v = v.view(B, S_gen, self.num_local_kv_heads, self.head_dim)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        k_all = torch.cat([k_und, k], dim=1)
        v_all = torch.cat([v_und, v], dim=1)
        out = self.attn(q, k_all, v_all)  # [B, S_gen, H, D]
        out = out.reshape(B, S_gen, -1)
        out, _ = self.to_out(out)
        return out


class Cosmos3UndDecoderLayer(nn.Module):
    """UND layer: pre-norm causal self-attn (returns K/V) + SwiGLU MLP."""

    def __init__(
        self,
        *,
        config: Cosmos3Config,
        attn_backend: str,
        norm_backend: str,
        params_dtype: torch.dtype | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Cosmos3CausalAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            params_dtype=params_dtype,
            prefix=f"{prefix}.self_attn" if prefix else "",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.input_layernorm" if prefix else "",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.post_attention_layernorm" if prefix else "",
        )
        self.mlp = DenseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="silu",
            gated=True,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp" if prefix else "",
        )

    def forward(
        self, hidden_states: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        cos, sin = freqs
        attn_out, k, v = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, k, v


class Cosmos3GenDecoderLayer(nn.Module):
    """GEN layer: pre-norm cross-attn (to UND K/V) + SwiGLU MLP."""

    def __init__(
        self,
        *,
        config: Cosmos3Config,
        attn_backend: str,
        norm_backend: str,
        params_dtype: torch.dtype | None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.cross_attention = Cosmos3CrossAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            params_dtype=params_dtype,
            prefix=f"{prefix}.cross_attention" if prefix else "",
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.input_layernorm" if prefix else "",
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.post_attention_layernorm" if prefix else "",
        )
        self.mlp = DenseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="silu",
            gated=True,
            bias=False,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp" if prefix else "",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.cross_attention(
            hidden_states, k_und, v_und, freqs_cos, freqs_sin
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Cosmos3LanguageModel(nn.Module):
    """UND causal LM. Returns per-layer ``(K, V)`` for the GEN cross-attention."""

    def __init__(
        self,
        config: Cosmos3Config,
        *,
        attn_backend: str,
        norm_backend: str,
        params_dtype: torch.dtype | None,
        device: torch.device | str | None,
        prefix: str = "language_model",
    ) -> None:
        super().__init__()
        if device is None:
            device = get_engine_config().device.target
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.embed_tokens" if prefix else "",
        )
        self.rotary_emb = InterleavedMRotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            mrope_section=config.mrope_section,
            rope_theta=config.rope_theta,
            backend="eager",
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                Cosmos3UndDecoderLayer(
                    config=config,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    params_dtype=params_dtype,
                    prefix=f"{prefix}.layers.{i}" if prefix else "",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # Reserved for a future prompt upsampler; present in the checkpoint.
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm" if prefix else "",
        )

    def forward(
        self, text_ids: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        hidden = self.embed_tokens(text_ids)
        cached_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden, k, v = layer(hidden, freqs)
            cached_kv.append((k, v))
        return cached_kv


@dataclass
class Cosmos3Condition:
    """Timestep-independent denoise condition: UND text K/V + GEN rope freqs."""

    cached_kv: list[tuple[torch.Tensor, torch.Tensor]]
    freqs_gen: tuple[torch.Tensor, torch.Tensor]
    video_shape: tuple[int, int, int]
    action_len: int = 0
    sound_len: int = 0


class Cosmos3Transformer(nn.Module):
    """Cosmos3 MoT flow-matching transformer."""

    def __init__(
        self,
        config: Cosmos3Config,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        if device is None:
            device = get_engine_config().device.target
        self.config = config
        self.latent_patch_size = config.latent_patch_size
        self.latent_channel_size = config.latent_channel
        self.timestep_scale = config.timestep_scale
        self.base_fps = config.base_fps
        self.enable_fps_modulation = config.enable_fps_modulation
        self.temporal_compression_factor = config.temporal_compression_factor
        self.temporal_modality_margin = config.temporal_modality_margin
        self.action_gen = config.action_gen
        self.sound_gen = config.sound_gen
        self.sound_dim = config.sound_dim
        self.action_dim = config.action_dim

        self.language_model = Cosmos3LanguageModel(
            config,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            params_dtype=params_dtype,
            device=device,
            prefix="language_model",
        )
        self.gen_layers = nn.ModuleList(
            [
                Cosmos3GenDecoderLayer(
                    config=config,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    params_dtype=params_dtype,
                    prefix=f"gen_layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm_moe_gen = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix="norm_moe_gen",
        )
        self.proj_in = ReplicatedLinear(
            config.patch_latent_dim,
            config.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix="proj_in",
        )
        self.proj_out = ReplicatedLinear(
            config.hidden_size,
            config.patch_latent_dim,
            bias=True,
            params_dtype=params_dtype,
            prefix="proj_out",
        )
        self.time_embedder = TimestepEmbedder(
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix="time_embedder",
        )

        # Action modality (policy / forward- & inverse-dynamics): domain-aware
        # in/out projections + an additive modality embedding. Built only when the
        # checkpoint carries them (action_gen); the GEN tower is shared.
        if self.action_gen:
            self.action_proj_in = DomainAwareLinear(
                config.action_dim,
                config.hidden_size,
                config.num_embodiment_domains,
                params_dtype=params_dtype,
                device=device,
                prefix="action_proj_in",
            )
            self.action_proj_out = DomainAwareLinear(
                config.hidden_size,
                config.action_dim,
                config.num_embodiment_domains,
                params_dtype=params_dtype,
                device=device,
                prefix="action_proj_out",
            )
            self.action_modality_embed = nn.Parameter(
                torch.zeros(config.hidden_size, dtype=params_dtype, device=device),
                requires_grad=False,
            )
            self.action_modality_embed.hf_keys = [("action_modality_embed", None)]
            self.action_modality_embed.weight_loader = replicated()

        # Sound stream adapters (T2VS / I2VS): plain linears 64<->hidden + an
        # additive modality embed; the AVAE that decodes the sound latent is a
        # separate module (avae_sound.py). Gated on sound_gen so non-sound
        # checkpoints stay byte-identical.
        if config.sound_gen:
            self.audio_proj_in = ReplicatedLinear(
                config.sound_dim,
                config.hidden_size,
                bias=True,
                params_dtype=params_dtype,
                prefix="audio_proj_in",
            )
            self.audio_proj_out = ReplicatedLinear(
                config.hidden_size,
                config.sound_dim,
                bias=True,
                params_dtype=params_dtype,
                prefix="audio_proj_out",
            )
            self.audio_modality_embed = nn.Parameter(
                torch.zeros(config.hidden_size, dtype=params_dtype, device=device),
                requires_grad=False,
            )
            self.audio_modality_embed.hf_keys = [("audio_modality_embed", None)]
            self.audio_modality_embed.weight_loader = replicated()

    def _pad_to_patch_size(self, h: int, w: int) -> tuple[int, int, int, int]:
        p = self.latent_patch_size
        h_pad = ((h + p - 1) // p) * p
        w_pad = ((w + p - 1) // p) * p
        return h_pad // p, w_pad // p, h_pad, w_pad

    def patchify(self, latents: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        B = latents.shape[0]
        p = self.latent_patch_size
        c = self.latent_channel_size
        hp, wp, h_pad, w_pad = self._pad_to_patch_size(h, w)
        if h_pad != h or w_pad != w:
            latents = F.pad(latents, (0, w_pad - w, 0, h_pad - h))
        x = latents.reshape(B, c, t, hp, p, wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)  # [B, t, hp, wp, p, p, C]
        return x.reshape(B, t * hp * wp, p * p * c)

    def unpatchify(self, tokens: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        B = tokens.shape[0]
        p = self.latent_patch_size
        c = self.latent_channel_size
        hp, wp, h_pad, w_pad = self._pad_to_patch_size(h, w)
        x = tokens.reshape(B, t, hp, wp, p, p, c)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)  # [B, C, t, hp, p, wp, p]
        x = x.reshape(B, c, t, h_pad, w_pad)
        if h_pad != h or w_pad != w:
            x = x[:, :, :, :h, :w]
        return x

    def _compute_rope_freqs(
        self,
        text_mask: torch.Tensor,
        t: int,
        hp: int,
        wp: int,
        fps: float | None,
        device: torch.device,
        dtype: torch.dtype,
        action_len: int = 0,
        sound_len: int = 0,
        sound_fps: float | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """mRoPE cos/sin for the UND text and GEN (video [+ action / + sound]) pathways.

        Returns ``(freqs_und, freqs_gen)``, each ``(cos, sin)`` of shape
        ``[B, S, 1, D]`` (the ``1`` broadcasts over the head axis). When
        ``action_len > 0`` the GEN positions are ``cat([video, action])`` — action
        shares the video's media temporal offset, a ``(T,1,1)`` grid at tcf=1 with
        ``start_frame_offset=1`` (the action packing convention). When
        ``sound_len > 0`` a sound ``(T,1,1)`` grid (tcf=1, ``start_frame_offset=0``,
        modulated by ``sound_fps``) is appended after the video/action positions
        (the sound packing convention).
        """
        B = text_mask.shape[0]
        s_text = text_mask.shape[1]
        text_lengths = text_mask.sum(dim=1).long()
        effective_fps = fps if fps is not None and t > 1 else None
        effective_sfps = sound_fps if sound_fps is not None and sound_len > 1 else None

        text_pos_list = []
        gen_pos_list = []
        for b in range(B):
            real_len = int(text_lengths[b].item())
            t_pos, t_offset = compute_mrope_position_ids_text(
                real_len, temporal_offset=0
            )
            media_temporal_offset = t_offset + self.temporal_modality_margin
            v_pos, _ = compute_mrope_position_ids_vision(
                t,
                hp,
                wp,
                temporal_offset=media_temporal_offset,
                fps=effective_fps,
                base_fps=self.base_fps,
                temporal_compression_factor=self.temporal_compression_factor,
                enable_fps_modulation=self.enable_fps_modulation,
            )
            if real_len < s_text:
                t_pos = torch.cat(
                    [t_pos, torch.zeros(3, s_text - real_len, dtype=t_pos.dtype)], dim=1
                )
            text_pos_list.append(t_pos)
            if action_len > 0:
                a_pos, _ = compute_mrope_position_ids_vision(
                    action_len,
                    1,
                    1,
                    temporal_offset=media_temporal_offset,
                    fps=effective_fps,
                    base_fps=self.base_fps,
                    temporal_compression_factor=1,
                    # Action runs at frame rate (tcf=1) but its fps modulation is
                    # normalized against the VIDEO temporal compression factor, so
                    # action and video share a comparable temporal coordinate.
                    # Omitting base_temporal_compression_factor here would scale the
                    # action temporal positions by ``temporal_compression_factor``.
                    base_temporal_compression_factor=self.temporal_compression_factor,
                    enable_fps_modulation=self.enable_fps_modulation,
                    start_frame_offset=1,
                )
                v_pos = torch.cat([v_pos, a_pos.to(v_pos.dtype)], dim=1)
            if sound_len > 0:
                s_pos, _ = compute_mrope_position_ids_vision(
                    sound_len,
                    1,
                    1,
                    temporal_offset=media_temporal_offset,
                    fps=effective_sfps,
                    base_fps=self.base_fps,
                    temporal_compression_factor=1,
                    enable_fps_modulation=self.enable_fps_modulation,
                    start_frame_offset=0,
                )
                v_pos = torch.cat([v_pos, s_pos.to(v_pos.dtype)], dim=1)
            gen_pos_list.append(v_pos)

        text_pos_ids = torch.stack(text_pos_list, dim=1).to(device)  # [3, B, S_text]
        gen_pos_ids = torch.stack(gen_pos_list, dim=1).to(device)  # [3, B, S_gen]

        rope = self.language_model.rotary_emb
        cos_und, sin_und = rope.get_cos_sin(text_pos_ids)
        cos_gen, sin_gen = rope.get_cos_sin(gen_pos_ids)
        freqs_und = (cos_und.unsqueeze(2).to(dtype), sin_und.unsqueeze(2).to(dtype))
        freqs_gen = (cos_gen.unsqueeze(2).to(dtype), sin_gen.unsqueeze(2).to(dtype))
        return freqs_und, freqs_gen

    @staticmethod
    def _gate_timestep(
        time_embed: torch.Tensor, mask: torch.Tensor | None, per_frame: int
    ) -> torch.Tensor:
        """Timestep contribution ``[B, S, H]`` to add — gated to noised frames.

        ``time_embed`` is ``[B, H]``; ``mask`` is ``[B, n_frames]`` (1=noised) and
        each frame owns ``per_frame`` tokens (token order is frame-outer). ``None``
        mask → add to every token (all-noised).
        """
        if mask is None:
            return time_embed.unsqueeze(1)
        gate = (
            mask.to(time_embed.dtype).repeat_interleave(per_frame, dim=1).unsqueeze(-1)
        )
        return time_embed.unsqueeze(1) * gate

    def encode_condition(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float | None = None,
        *,
        action_len: int = 0,
        sound_len: int = 0,
        sound_fps: float | None = None,
    ) -> Cosmos3Condition:
        """Encode the timestep-independent denoise condition for one prompt.

        Runs the UND text tower **once** (a single full prefill over the prompt)
        and builds the GEN-pathway rope freqs. The result depends only on the text
        + media grid ``video_shape`` + ``fps`` (and the auxiliary token counts),
        never on the diffusion timestep or the noisy latents, so the caller
        (:class:`~phyai.models.cosmos3.model_runner_cosmos3.Cosmos3T2VRunner`)
        caches it per CFG branch and reuses it across every denoise step.

        ``action_len`` / ``sound_len`` extend the GEN rope positions with the
        action / sound ``(T, 1, 1)`` grid; they must match the ``action_latents`` /
        ``sound_latents`` later passed to :meth:`forward`.
        """
        t, h, w = video_shape
        hp, wp, _, _ = self._pad_to_patch_size(h, w)
        text_lengths = text_mask.sum(dim=1)
        max_real_len = int(text_lengths.max().item())
        if int(text_lengths.min().item()) != max_real_len:
            raise ValueError(
                "Cosmos3 requires identical real text lengths within a batch."
            )
        dtype = next(self.parameters()).dtype
        freqs_und, freqs_gen = self._compute_rope_freqs(
            text_mask,
            t,
            hp,
            wp,
            fps,
            text_ids.device,
            dtype,
            action_len=action_len,
            sound_len=sound_len,
            sound_fps=sound_fps,
        )
        cached_kv_full = self.language_model(text_ids, freqs_und)
        cached_kv = [
            (k[:, :max_real_len], v[:, :max_real_len]) for k, v in cached_kv_full
        ]
        return Cosmos3Condition(
            cached_kv=cached_kv,
            freqs_gen=freqs_gen,
            video_shape=video_shape,
            action_len=action_len,
            sound_len=sound_len,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        condition: Cosmos3Condition,
        *,
        noisy_frame_mask: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        action_domain_id: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        sound_noisy_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Velocity prediction for video [+ action | + sound] at one timestep.

        Stateless: the timestep-independent UND text K/V + GEN rope freqs are
        supplied via ``condition`` (see :meth:`encode_condition`), so this is only
        the per-step GEN pass. Build the condition once per CFG branch and reuse it
        across every denoise step.

        Args:
            hidden_states: ``[B, C, t, h, w]`` noisy video latents.
            timestep: ``[B]`` diffusion timestep (shared across modalities).
            condition: precomputed :class:`Cosmos3Condition` (UND K/V + GEN freqs +
                ``video_shape`` + auxiliary token lengths).
            noisy_frame_mask: optional ``[B, t]`` (1=noised/0=clean) — I2V/V2V; the
                additive timestep is applied only to noised video frames.
            action_latents: optional ``[B, chunk, action_dim]`` action latents
                (policy / dynamics). When given, action tokens are packed after the
                video tokens and a ``(video_velocity, action_velocity)`` pair is
                returned; ``condition.action_len`` must equal ``chunk``.
            action_domain_id: ``[B]`` embodiment domain id for the action adapters.
            action_noisy_mask: optional ``[B, chunk]`` (1=noised/0=clean) action mask.
            sound_latents: optional ``[B, T_sound, sound_dim]`` token-major sound
                latents (T2VS / I2VS). When given, sound tokens are packed after the
                video tokens and a ``(video_velocity, sound_velocity)`` pair is
                returned; ``condition.sound_len`` must equal ``T_sound``. Mutually
                exclusive with ``action_latents``.
            sound_noisy_mask: optional ``[B, T_sound]`` (1=noised/0=clean) sound mask.

        Returns:
            ``[B, C, t, h, w]`` video velocity; or ``(video_velocity, aux_velocity)``
            where ``aux`` is the ``[B, chunk, action_dim]`` action or
            ``[B, T_sound, sound_dim]`` sound velocity, when that stream is given.
        """
        t, h, w = condition.video_shape
        hp, wp, _, _ = self._pad_to_patch_size(h, w)

        # Patchify + project + additive timestep embedding (video tokens).
        hidden_video, _ = self.proj_in(self.patchify(hidden_states, t, h, w))
        time_embed = self.time_embedder(timestep * self.timestep_scale)
        time_embed = time_embed.to(hidden_video.dtype)
        hidden_gen = hidden_video + self._gate_timestep(
            time_embed, noisy_frame_mask, hp * wp
        )
        s_video = hidden_gen.shape[1]

        # Action stream (policy / forward- & inverse-dynamics): domain-aware
        # projection + modality embed + gated timestep, packed after the video.
        if action_latents is not None:
            if action_latents.shape[1] != condition.action_len:
                raise ValueError(
                    f"action_latents length {action_latents.shape[1]} != "
                    f"condition.action_len {condition.action_len}."
                )
            action_tok = self.action_proj_in(action_latents, action_domain_id)
            action_tok = action_tok + self.action_modality_embed.to(action_tok.dtype)
            action_tok = action_tok + self._gate_timestep(
                time_embed, action_noisy_mask, 1
            )
            hidden_gen = torch.cat([hidden_gen, action_tok], dim=1)

        # Sound stream (T2VS / I2VS): plain projection + modality embed + gated
        # timestep, packed after the video. Mutually exclusive with action.
        if sound_latents is not None:
            if action_latents is not None:
                raise NotImplementedError(
                    "Cosmos3 does not co-generate action and sound in one forward."
                )
            if sound_latents.shape[1] != condition.sound_len:
                raise ValueError(
                    f"sound_latents length {sound_latents.shape[1]} != "
                    f"condition.sound_len {condition.sound_len}."
                )
            sound_tok, _ = self.audio_proj_in(sound_latents)
            sound_tok = sound_tok + self.audio_modality_embed.to(sound_tok.dtype)
            sound_tok = sound_tok + self._gate_timestep(time_embed, sound_noisy_mask, 1)
            hidden_gen = torch.cat([hidden_gen, sound_tok], dim=1)

        # GEN pathway: cross-attend to the precomputed UND K/V at the GEN positions.
        freqs_cos, freqs_sin = condition.freqs_gen
        for layer, (k_und, v_und) in zip(
            self.gen_layers, condition.cached_kv, strict=True
        ):
            hidden_gen = layer(hidden_gen, k_und, v_und, freqs_cos, freqs_sin)

        hidden_gen = self.norm_moe_gen(hidden_gen)
        video_out, _ = self.proj_out(hidden_gen[:, :s_video])
        video_vel = self.unpatchify(video_out, t, h, w)
        if action_latents is not None:
            action_vel = self.action_proj_out(hidden_gen[:, s_video:], action_domain_id)
            return video_vel, action_vel
        if sound_latents is not None:
            sound_vel, _ = self.audio_proj_out(hidden_gen[:, s_video:])
            return video_vel, sound_vel
        return video_vel


# Per-layer leaf-name map: checkpoint source leaf -> phyai param leaf. The
# source ``layers.{i}.*`` is one physical layer holding BOTH experts; it fans
# out to ``language_model.layers.{i}`` (UND) and ``gen_layers.{i}`` (GEN).
_UND_LEAF = {
    "self_attn.to_q": "self_attn.to_q",
    "self_attn.to_k": "self_attn.to_k",
    "self_attn.to_v": "self_attn.to_v",
    "self_attn.to_out": "self_attn.to_out",
    "self_attn.norm_q": "self_attn.norm_q",
    "self_attn.norm_k": "self_attn.norm_k",
    "input_layernorm": "input_layernorm",
    "post_attention_layernorm": "post_attention_layernorm",
    "mlp.gate_proj": "mlp.gate_proj",
    "mlp.up_proj": "mlp.up_proj",
    "mlp.down_proj": "mlp.down_proj",
}
_GEN_LEAF = {
    "self_attn.add_q_proj": "cross_attention.to_q",
    "self_attn.add_k_proj": "cross_attention.to_k",
    "self_attn.add_v_proj": "cross_attention.to_v",
    "self_attn.to_add_out": "cross_attention.to_out",
    "self_attn.norm_added_q": "cross_attention.norm_q",
    "self_attn.norm_added_k": "cross_attention.norm_k",
    "input_layernorm_moe_gen": "input_layernorm",
    "post_attention_layernorm_moe_gen": "post_attention_layernorm",
    "mlp_moe_gen.gate_proj": "mlp.gate_proj",
    "mlp_moe_gen.up_proj": "mlp.up_proj",
    "mlp_moe_gen.down_proj": "mlp.down_proj",
}
# Top-level (non-layer) keys kept for the build: source -> phyai. embed_tokens/
# norm go under the UND language_model; proj/time/action stay at root.
_TOP_PREFIXES = (
    "proj_in.",
    "proj_out.",
    "time_embedder.",
    "norm_moe_gen.",
    "action_proj_in.",
    "action_proj_out.",
    "action_modality_embed",
    "audio_proj_in.",
    "audio_proj_out.",
    "audio_modality_embed",
)
# Top-level keys DROPPED: only the UND prompt-upsampler head -> None (the action
# and sound adapters are now constructed and kept).
_DROP_PREFIXES = ("lm_head.",)


def cosmos3_weight_remap(key: str) -> str | None:
    """Map a Cosmos3-Nano ``transformer/`` checkpoint key to a phyai param name.

    Returns the remapped name, or ``None`` to drop the key. The shared
    ``layers.{i}.<leaf>`` source fans out to UND (``language_model.layers.{i}``)
    and GEN (``gen_layers.{i}``) per :data:`_UND_LEAF` / :data:`_GEN_LEAF`.
    Only ``lm_head`` is dropped (the T2V/gen build has no prompt-upsampler head);
    action and sound projections are kept so a strict load is clean.
    """
    for pref in _DROP_PREFIXES:
        if key.startswith(pref):
            return None
    if key.startswith("embed_tokens.") or key.startswith("norm."):
        return f"language_model.{key}"
    for pref in _TOP_PREFIXES:
        if key.startswith(pref):
            return key
    if key.startswith("layers."):
        rest = key[len("layers.") :]
        idx, leaf = rest.split(".", 1)
        # Strip the trailing ``.weight`` / ``.bias`` to match against leaf maps.
        for suffix in (".weight", ".bias"):
            if leaf.endswith(suffix):
                base, tail = leaf[: -len(suffix)], suffix
                break
        else:
            base, tail = leaf, ""
        if base in _UND_LEAF:
            return f"language_model.layers.{idx}.{_UND_LEAF[base]}{tail}"
        if base in _GEN_LEAF:
            return f"gen_layers.{idx}.{_GEN_LEAF[base]}{tail}"
        return None
    return None


__all__ = [
    "Cosmos3Transformer",
    "Cosmos3Condition",
    "Cosmos3LanguageModel",
    "Cosmos3UndDecoderLayer",
    "Cosmos3GenDecoderLayer",
    "Cosmos3CausalAttention",
    "Cosmos3CrossAttention",
    "TimestepEmbedder",
    "cosmos3_weight_remap",
    "compute_mrope_position_ids_text",
    "compute_mrope_position_ids_vision",
    "local_head_counts",
]
