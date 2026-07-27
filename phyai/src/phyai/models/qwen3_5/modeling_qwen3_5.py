"""Prefill-only Qwen3.5 modeling."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config, resolve_engine_defaults
from phyai.layers.attention.attention.layer import Attention
from phyai.layers.attention.gdn import GatedDeltaNet
from phyai.layers.conv import Conv1d, Conv3d
from phyai.layers.layer_norm import GemmaRMSNorm, LayerNorm
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.rotary_embedding import (
    RotaryEmbedding,
    compute_qwen3vl_mrope_cos_sin_from_inv_freq,
    rotate_half,
)
from phyai.layers.vocab_embedding import ParallelLMHead, VocabParallelEmbedding
from phyai.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from phyai.weights.shards import replicated


if TYPE_CHECKING:
    from phyai.layers.attention import ARAttnCtx, GatedDeltaNetCtx


def _prepare_prefill_attention_mask(
    attention_mask: torch.Tensor | None, input_shape: torch.Size
) -> None:
    if attention_mask is None:
        return None
    if attention_mask.shape != input_shape:
        raise ValueError(
            f"attention_mask must have shape {tuple(input_shape)}, got "
            f"{tuple(attention_mask.shape)}."
        )
    if not bool(attention_mask.bool().all()):
        raise NotImplementedError(
            "Qwen3.5 prefill does not support padding because full-attention "
            "layers require a padding-aware attention context."
        )
    return None


def qwen3_5_weight_remap(name: str) -> str | None:
    """Drop checkpoint components that are outside the main prefill graph."""
    if name.startswith("mtp."):
        return None
    return name


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


def get_vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    position_ids: list[torch.Tensor] = []
    device = grid_thw.device
    merge = spatial_merge_size
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w)
        hpos = (
            hpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        )
        wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1)
        wpos = (
            wpos.reshape(h // merge, merge, w // merge, merge).transpose(1, 2).flatten()
        )
        position_ids.append(torch.stack((hpos, wpos), dim=-1).repeat(t, 1))
    return torch.cat(position_ids)


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    side = num_grid_per_side
    merge = spatial_merge_size
    device = grid_thw.device
    index_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor, w_floor = h_grid.int(), w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        dh, dw = h_grid - h_floor, w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        indices = (
            (h_floor_offset[:, None] + w_floor[None]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None]).flatten(),
        )
        weights = (
            ((1 - dh)[:, None] * (1 - dw)[None]).flatten(),
            ((1 - dh)[:, None] * dw[None]).flatten(),
            (dh[:, None] * (1 - dw)[None]).flatten(),
            (dh[:, None] * dw[None]).flatten(),
        )
        h_idx = torch.arange(h, device=device).view(h // merge, merge)
        w_idx = torch.arange(w, device=device).view(w // merge, merge)
        reorder = (
            (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
            .transpose(1, 2)
            .flatten()
            .repeat(t)
        )
        for i in range(4):
            index_parts[i].append(indices[i][reorder])
            weight_parts[i].append(weights[i][reorder])
    return (
        torch.stack([torch.cat(part) for part in index_parts]),
        torch.stack([torch.cat(part) for part in weight_parts]),
    )


class Qwen3_5RMSNormGated(nn.Module):
    """Qwen3.5 head-wise RMSNorm followed by a SiLU gate."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        *,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.weight.hf_keys = [(f"{prefix}.weight", None)]
        self.weight.weight_loader = replicated()

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Qwen3.5 projections, causal convolution, and GDN core."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        params_dtype: torch.dtype,
        gdn_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = ReplicatedLinear(
            config.hidden_size,
            self.conv_dim,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.in_proj_qkv",
        )
        self.in_proj_z = ReplicatedLinear(
            config.hidden_size,
            self.value_dim,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.in_proj_z",
        )
        self.in_proj_b = ReplicatedLinear(
            config.hidden_size,
            self.num_value_heads,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.in_proj_b",
        )
        self.in_proj_a = ReplicatedLinear(
            config.hidden_size,
            self.num_value_heads,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.in_proj_a",
        )
        self.conv1d = Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel_size,
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim,
            bias=False,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.conv1d",
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_value_heads, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.A_log.hf_keys = [(f"{prefix}.A_log", None)]
        self.A_log.weight_loader = replicated()
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_value_heads, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        self.dt_bias.hf_keys = [(f"{prefix}.dt_bias", None)]
        self.dt_bias.weight_loader = replicated()
        self.gdn = GatedDeltaNet(
            self.num_value_heads,
            self.value_head_dim,
            num_key_heads=self.num_value_heads,
            num_value_heads=self.num_value_heads,
            backend=gdn_backend,
        )
        self.norm = Qwen3_5RMSNormGated(
            self.value_head_dim,
            config.rms_norm_eps,
            device=device,
            prefix=f"{prefix}.norm",
        )
        self.out_proj = ReplicatedLinear(
            self.value_dim,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.out_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        gdn_ctx: "GatedDeltaNetCtx | None" = None,
    ) -> torch.Tensor:
        if attention_mask is not None and hidden_states.shape[0] > 1:
            hidden_states = hidden_states * attention_mask[..., None]
        batch_size, seq_len, _ = hidden_states.shape
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        mixed_qkv = self.conv1d(mixed_qkv.transpose(1, 2))[:, :, :seq_len]
        mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        query = query.reshape(
            batch_size, seq_len, self.num_key_heads, self.key_head_dim
        )
        key = key.reshape(batch_size, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(
            batch_size, seq_len, self.num_value_heads, self.value_head_dim
        )
        if self.num_value_heads != self.num_key_heads:
            repeats = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        a, _ = self.in_proj_a(hidden_states)
        b, _ = self.in_proj_b(hidden_states)
        core_output = self.gdn(
            query, key, value, a, b, self.A_log, self.dt_bias, ctx=gdn_ctx
        )
        z, _ = self.in_proj_z(hidden_states)
        core_output = self.norm(
            core_output.reshape(-1, self.value_head_dim),
            z.reshape(-1, self.value_head_dim),
        ).reshape(batch_size, seq_len, self.value_dim)
        output, _ = self.out_proj(core_output)
        return output


class Qwen3_5Attention(nn.Module):
    """Gated Qwen3.5 full attention."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        rotary_emb: RotaryEmbedding,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_emb = rotary_emb
        self.q_proj = ReplicatedLinear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ReplicatedLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ReplicatedLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.v_proj",
        )
        self.q_norm = GemmaRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.q_norm",
        )
        self.k_norm = GemmaRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.k_norm",
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_kv_heads,
            causal=True,
            backend=attn_backend,
        )
        self.o_proj = ReplicatedLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: "ARAttnCtx | None" = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_gate, _ = self.q_proj(hidden_states)
        query, gate = query_gate.view(
            batch_size, seq_len, self.num_heads, self.head_dim * 2
        ).chunk(2, dim=-1)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)
        query = self.q_norm(query)
        key = self.k_norm(
            key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        )
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        query, key = self.rotary_emb.apply(query, key, cos, sin)
        output = self.attn(query, key, value, ctx=attn_ctx)
        output = output.reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate.reshape(batch_size, seq_len, -1))
        output, _ = self.o_proj(output)
        return output


class Qwen3_5MLP(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_proj = ReplicatedLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ReplicatedLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = ReplicatedLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int,
        *,
        rotary_emb: RotaryEmbedding,
        params_dtype: torch.dtype,
        attn_backend: str,
        gdn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                params_dtype=params_dtype,
                gdn_backend=gdn_backend,
                device=device,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(
                config,
                rotary_emb=rotary_emb,
                params_dtype=params_dtype,
                attn_backend=attn_backend,
                norm_backend=norm_backend,
                device=device,
                prefix=f"{prefix}.self_attn",
            )
        self.mlp = Qwen3_5MLP(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.input_layernorm",
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.post_attention_layernorm",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_ctx: "ARAttnCtx | None" = None,
        gdn_ctx: "GatedDeltaNetCtx | None" = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, attention_mask=attention_mask, gdn_ctx=gdn_ctx
            )
        else:
            hidden_states = self.self_attn(
                hidden_states, cos=cos, sin=sin, attn_ctx=attn_ctx
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class Qwen3_5TextModel(nn.Module):
    """Qwen3.5 hybrid text decoder for stateless prefill."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        gdn_backend: str = "flashinfer",
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "model.language_model",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        if device is None:
            device = get_engine_config().device.target
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.embed_tokens",
        )
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_type=config.rope_type,
            partial_rotary_factor=config.partial_rotary_factor,
            backend="eager",
            device=device,
        )
        del self.rotary_emb.cos_sin_cache
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(
                    config,
                    layer_idx,
                    rotary_emb=self.rotary_emb,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    gdn_backend=gdn_backend,
                    norm_backend=norm_backend,
                    device=device,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm",
        )

    def get_position_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, *position_ids.shape)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        if position_ids.shape != (3, batch_size, seq_len):
            raise ValueError(
                f"position_ids must have shape (3, B, S) or (4, B, S), got "
                f"{tuple(position_ids.shape)}."
            )
        cos, sin = compute_qwen3vl_mrope_cos_sin_from_inv_freq(
            position_ids,
            self.rotary_emb.inv_freq,
            self.config.mrope_section,
        )
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attn_ctx: "ARAttnCtx | None" = None,
        gdn_ctxs: list["GatedDeltaNetCtx | None"] | None = None,
    ) -> torch.Tensor:
        _prepare_prefill_attention_mask(attention_mask, inputs_embeds.shape[:2])
        attention_mask = None
        cos, sin = self.get_position_embeddings(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            gdn_ctx = None if gdn_ctxs is None else gdn_ctxs[layer_idx]
            hidden_states = layer(
                hidden_states,
                cos=cos,
                sin=sin,
                attention_mask=attention_mask,
                attn_ctx=attn_ctx,
                gdn_ctx=gdn_ctx,
            )
        return self.norm(hidden_states)


class Qwen3_5VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.to(position_ids.device)
        return freqs.flatten(1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q = q.float() * cos + rotate_half(q.float()) * sin
    k = k.float() * cos + rotate_half(k.float()) * sin
    return q.to(q_dtype), k.to(k_dtype)


class Qwen3_5VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel,
            stride=kernel,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(
            -1, self.embed_dim
        )


class Qwen3_5VisionPatchMerger(nn.Module):
    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        params_dtype: torch.dtype,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.merged_dim = config.hidden_size * config.spatial_merge_size**2
        self.norm = LayerNorm(
            config.hidden_size,
            eps=1e-6,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm",
        )
        self.linear_fc1 = ReplicatedLinear(
            self.merged_dim,
            self.merged_dim,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_fc1",
        )
        self.linear_fc2 = ReplicatedLinear(
            self.merged_dim,
            config.out_hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states).view(-1, self.merged_dim)
        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states, _ = self.linear_fc2(hidden_states)
        return hidden_states


class Qwen3_5VisionMLP(nn.Module):
    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.linear_fc1 = ReplicatedLinear(
            config.hidden_size,
            config.intermediate_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_fc1",
        )
        self.linear_fc2 = ReplicatedLinear(
            config.intermediate_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.linear_fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states, _ = self.linear_fc2(hidden_states)
        return hidden_states


class Qwen3_5VisionAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size * 3,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.qkv",
        )
        self.proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.proj",
        )
        self.attn = Attention(
            self.num_heads, self.head_dim, causal=False, backend=attn_backend
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        qkv, _ = self.qkv(hidden_states)
        q, k, v = (
            qkv.view(seq_len, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        output = self.attn(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens)
        output, _ = self.proj(output.reshape(seq_len, -1))
        return output


class Qwen3_5VisionBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(
            config.hidden_size,
            eps=1e-6,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm1",
        )
        self.norm2 = LayerNorm(
            config.hidden_size,
            eps=1e-6,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm2",
        )
        self.attn = Qwen3_5VisionAttention(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            device=device,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Qwen3_5VisionMLP(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, position_embeddings
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3_5VisionModel(nn.Module):
    """Qwen3.5 native ViT vision tower."""

    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "model.visual",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        if device is None:
            device = get_engine_config().device.target
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = config.num_grid_per_side
        self.patch_embed = Qwen3_5VisionPatchEmbed(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.patch_embed",
        )
        self.pos_embed_weight = nn.Parameter(
            torch.empty(
                config.num_position_embeddings,
                config.hidden_size,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.pos_embed_weight.hf_keys = [(f"{prefix}.pos_embed.weight", None)]
        self.pos_embed_weight.weight_loader = replicated()
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(config.head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen3_5VisionBlock(
                    config,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    device=device,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(config.depth)
            ]
        )
        self.merger = Qwen3_5VisionPatchMerger(
            config,
            params_dtype=params_dtype,
            norm_backend=norm_backend,
            device=device,
            prefix=f"{prefix}.merger",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.pos_embed_weight.dtype

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        grid_thw = grid_thw.to(device="cpu")
        indices, weights = get_vision_bilinear_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        device = self.pos_embed_weight.device
        indices = indices.to(device=device)
        weights = weights.to(device=device, dtype=self.pos_embed_weight.dtype)
        position_ids = position_ids.to(device=device)
        cu_seqlens = cu_seqlens.to(device=device)
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (self.pos_embed_weight[indices] * weights[..., None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        rotary = self.rotary_pos_emb(position_ids)
        rotary = rotary.reshape(hidden_states.shape[0], -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = rotary.cos(), rotary.sin()
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)
        return self.merger(hidden_states)


class Qwen3_5Model(nn.Module):
    """Vision tower and hybrid language model for multimodal prefill."""

    def __init__(
        self,
        config: Qwen3_5Config,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        gdn_backend: str = "flashinfer",
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        vision_attn_backend: str | None = None,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        self.visual = Qwen3_5VisionModel(
            config.vision,
            params_dtype=vision_params_dtype or params_dtype,
            attn_backend=vision_attn_backend or attn_backend,
            norm_backend=norm_backend,
            device=device,
        )
        self.language_model = Qwen3_5TextModel(
            config.text,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            gdn_backend=gdn_backend,
            norm_backend=norm_backend,
            device=device,
        )

    def get_image_features(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        return self.visual(pixel_values.to(self.visual.dtype), grid_thw)

    @staticmethod
    def get_vision_position_ids(
        start_position: int,
        grid_thw: torch.Tensor | list[int] | tuple[int, ...],
        spatial_merge_size: int,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if isinstance(grid_thw, torch.Tensor):
            if device is None:
                device = grid_thw.device
            grid_thw = grid_thw.tolist()
        t, h, w = (int(x) for x in grid_thw)
        h //= spatial_merge_size
        w //= spatial_merge_size
        temporal = torch.arange(t, device=device).repeat_interleave(h * w)
        height = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        width = torch.arange(w, device=device).repeat(h * t)
        return torch.stack((temporal, height, width)) + start_position

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        *,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_grids = None if image_grid_thw is None else image_grid_thw.tolist()
        video_grids = None
        if video_grid_thw is not None:
            video_grids = []
            for t, h, w in video_grid_thw.tolist():
                video_grids.extend([(1, h, w)] * int(t))
        grid_iters = {
            1: iter(image_grids) if image_grids is not None else None,
            2: iter(video_grids) if video_grids is not None else None,
        }
        position_ids = torch.zeros(
            3,
            *input_ids.shape,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        token_type_rows = mm_token_type_ids.to(device="cpu").tolist()
        for batch_idx, token_types in enumerate(token_type_rows):
            mask = None if attention_mask is None else attention_mask[batch_idx].bool()
            if mask is not None:
                mask_values = mask.to(device="cpu").tolist()
                token_types = [
                    kind for kind, keep in zip(token_types, mask_values) if keep
                ]
            groups = []
            for kind, group in itertools.groupby(
                enumerate(token_types), lambda item: item[1]
            ):
                group = list(group)
                groups.append((kind, group[0][0], group[-1][0] + 1))
            current_position = 0
            parts = []
            for kind, start, end in groups:
                if kind == 0:
                    length = end - start
                    part = torch.arange(length, device=input_ids.device)
                    parts.append(part.view(1, -1).expand(3, -1) + current_position)
                    current_position += length
                else:
                    grid_iter = grid_iters.get(kind)
                    if grid_iter is None:
                        raise ValueError(
                            f"Missing grid metadata for token type {kind}."
                        )
                    try:
                        grid = next(grid_iter)
                    except StopIteration as error:
                        raise ValueError(
                            f"Not enough grid entries for token type {kind}."
                        ) from error
                    parts.append(
                        self.get_vision_position_ids(
                            current_position,
                            grid,
                            self.config.vision.spatial_merge_size,
                            device=input_ids.device,
                        )
                    )
                    current_position += (
                        int(max(grid[1], grid[2]))
                        // self.config.vision.spatial_merge_size
                    )
            positions = torch.cat(parts, dim=1)
            if mask is None:
                position_ids[:, batch_idx] = positions
            else:
                position_ids[:, batch_idx, mask] = positions
        for kind, grid_iter in grid_iters.items():
            if grid_iter is None:
                continue
            try:
                next(grid_iter)
            except StopIteration:
                continue
            raise ValueError(f"Unused grid entries for token type {kind}.")
        return position_ids

    def _validate_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> None:
        modalities = (
            (
                "image",
                1,
                pixel_values,
                image_grid_thw,
            ),
            (
                "video",
                2,
                pixel_values_videos,
                video_grid_thw,
            ),
        )
        for name, _, pixels, grid in modalities:
            if (pixels is None) != (grid is None):
                raise ValueError(
                    f"{name} pixels and {name}_grid_thw must be provided together."
                )

        if mm_token_type_ids is not None and mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                f"mm_token_type_ids must have shape {tuple(input_ids.shape)}, got "
                f"{tuple(mm_token_type_ids.shape)}."
            )

        input_rows = input_ids.to(device="cpu").tolist()
        expected_type_rows = [
            [
                1
                if token == self.config.image_token_id
                else 2
                if token == self.config.video_token_id
                else 0
                for token in row
            ]
            for row in input_rows
        ]
        if mm_token_type_ids is not None:
            type_rows = mm_token_type_ids.to(device="cpu").tolist()
            if type_rows != expected_type_rows:
                raise ValueError(
                    "mm_token_type_ids must match image and video token placeholders."
                )

        group_lengths: dict[int, list[int]] = {1: [], 2: []}
        for type_row in expected_type_rows:
            for kind, group in itertools.groupby(type_row):
                length = sum(1 for _ in group)
                if kind in group_lengths:
                    group_lengths[kind].append(length)

        merge = self.config.vision.spatial_merge_size
        patch_elements = (
            self.config.vision.in_channels
            * self.config.vision.temporal_patch_size
            * self.config.vision.patch_size**2
        )
        for name, kind, pixels, grid in modalities:
            expected_lengths: list[int] = []
            expected_patches = 0
            if grid is not None:
                if grid.ndim != 2 or grid.shape[1] != 3:
                    raise ValueError(
                        f"{name}_grid_thw must have shape (N, 3), got "
                        f"{tuple(grid.shape)}."
                    )
                for t, h, w in grid.tolist():
                    t, h, w = int(t), int(h), int(w)
                    if min(t, h, w) <= 0:
                        raise ValueError(f"{name}_grid_thw values must be positive.")
                    if h % merge or w % merge:
                        raise ValueError(
                            f"{name}_grid_thw spatial dimensions must be divisible "
                            f"by spatial_merge_size={merge}."
                        )
                    expected_patches += t * h * w
                    merged_spatial = (h // merge) * (w // merge)
                    if kind == 2:
                        expected_lengths.extend([merged_spatial] * t)
                    else:
                        expected_lengths.append(t * merged_spatial)
            if (
                pixels is not None
                and pixels.numel() != expected_patches * patch_elements
            ):
                raise ValueError(
                    f"{name} pixels contain the wrong number of patch elements for "
                    f"{name}_grid_thw."
                )
            if group_lengths[kind] != expected_lengths:
                raise ValueError(
                    f"{name} token placeholder groups do not match {name}_grid_thw."
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        attn_ctx: "ARAttnCtx | None" = None,
        gdn_ctxs: list["GatedDeltaNetCtx | None"] | None = None,
    ) -> torch.Tensor:
        _prepare_prefill_attention_mask(attention_mask, input_ids.shape)
        attention_mask = None
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device="cpu")
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device="cpu")
        has_multimodal = any(
            value is not None
            for value in (
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
            )
        )
        if has_multimodal:
            self._validate_multimodal_inputs(
                input_ids,
                mm_token_type_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        for pixel_data, grid, token_id in (
            (pixel_values, image_grid_thw, self.config.image_token_id),
            (pixel_values_videos, video_grid_thw, self.config.video_token_id),
        ):
            if pixel_data is None:
                continue
            features = self.get_image_features(pixel_data, grid)
            token_mask = (input_ids == token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[token_mask].numel() != features.numel():
                raise ValueError(
                    f"Vision features and token placeholders differ for token {token_id}."
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                token_mask, features.to(inputs_embeds.dtype)
            )
        if position_ids is None and has_multimodal:
            if mm_token_type_ids is None:
                raise ValueError(
                    "mm_token_type_ids is required for multimodal prefill."
                )
            position_ids = self.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        return self.language_model(
            inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            attn_ctx=attn_ctx,
            gdn_ctxs=gdn_ctxs,
        )


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Qwen3.5 multimodal model with a tied LM head."""

    def __init__(
        self,
        config: Qwen3_5Config,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        gdn_backend: str = "flashinfer",
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        vision_attn_backend: str | None = None,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        self.model = Qwen3_5Model(
            config,
            params_dtype=params_dtype,
            vision_params_dtype=vision_params_dtype,
            attn_backend=attn_backend,
            gdn_backend=gdn_backend,
            norm_backend=norm_backend,
            device=device,
            vision_attn_backend=vision_attn_backend,
        )
        tied_weight = (
            self.model.language_model.embed_tokens.weight
            if config.tie_word_embeddings
            else None
        )
        self.lm_head = ParallelLMHead(
            config.text.hidden_size,
            config.text.vocab_size,
            bias=False,
            tied_weight=tied_weight,
            params_dtype=params_dtype,
            device=device,
            prefix="lm_head",
        )

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.model(input_ids, **kwargs)
        return self.lm_head(hidden_states)


__all__ = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5Model",
    "Qwen3_5TextModel",
    "Qwen3_5VisionModel",
    "qwen3_5_weight_remap",
]
