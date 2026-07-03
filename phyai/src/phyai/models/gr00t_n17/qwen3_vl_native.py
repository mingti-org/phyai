"""Native Qwen3-VL pieces used by GR00T-N1.7.

This file is an inference-focused port of the Qwen3-VL model structure
from Hugging Face Transformers' Apache-2.0 implementation:
``transformers.models.qwen3_vl.modeling_qwen3_vl``.

Scope for this first native pass:
* GR00T prefill/image path, no generation logits needed by the scheduler.
* No KV cache path.
* Image inputs are supported; video hooks are intentionally left out until
  GR00T needs them.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.attention import Attention
from phyai.layers.linear.layers import ReplicatedLinear
from phyai.layers.rotary_embedding import rotate_half
from phyai.models.qwen3_vl.modeling_qwen3_vl import (
    get_vision_cu_seqlens,
    get_vision_position_ids,
)
from phyai.weights.shards import replicated


# NOTE: the Qwen3-VL backbone is stateless. CUDA-graph capture/replay for the
# ViT and LLM cores is owned by ``GR00TN17BackboneRunner`` (model_runner), which
# passes a graph context into ``forward``; when it is ``None`` the eager path
# runs (byte-identical to the reference).


# ============================================================================ #
#  Qwen3-VL native common layers                                              #
# ============================================================================ #


def _attach_replicated_hf_key(param: nn.Parameter, hf_key: str) -> None:
    param.hf_keys = [(hf_key, None)]
    param.weight_loader = replicated()


def _replicated_linear(linear: ReplicatedLinear, x: torch.Tensor) -> torch.Tensor:
    out, bias = ReplicatedLinear.forward(linear, x)
    if bias is not None:
        return out + bias
    return out


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    scaling: float,
    num_key_value_groups: int = 1,
    is_causal: bool = False,
) -> torch.Tensor:
    key = _repeat_kv(key, num_key_value_groups)
    value = _repeat_kv(value, num_key_value_groups)
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scaling,
    )
    return out.transpose(1, 2).contiguous()


def _attention_backend_kwargs(backend: str) -> dict[str, Any] | None:
    return {"compile": False} if backend == "sdpa" else None


class GR00TN17QwenLinear(ReplicatedLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        prefix: str,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            prefix=prefix,
            params_dtype=params_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _replicated_linear(self, x)


class GR00TN17QwenRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        prefix: str,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.variance_epsilon = float(eps)
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        _attach_replicated_hf_key(self.weight, f"{prefix}.weight")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(device=hidden_states.device, dtype=input_dtype) * hidden_states.to(
            input_dtype
        )


class GR00TN17QwenLayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        prefix: str,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.variance_epsilon = float(eps)
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_size, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        _attach_replicated_hf_key(self.weight, f"{prefix}.weight")
        _attach_replicated_hf_key(self.bias, f"{prefix}.bias")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            (self.hidden_size,),
            self.weight.to(device=x.device),
            self.bias.to(device=x.device),
            self.variance_epsilon,
        )


class GR00TN17QwenEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        prefix: str,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        _attach_replicated_hf_key(self.weight, f"{prefix}.weight")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class GR00TN17QwenConv3dPatchEmbed(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        prefix: str,
        params_dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = int(config.patch_size)
        self.temporal_patch_size = int(config.temporal_patch_size)
        self.in_channels = int(config.in_channels)
        self.embed_dim = int(config.hidden_size)
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.weight = nn.Parameter(
            torch.empty(
                self.embed_dim,
                self.in_channels,
                *kernel,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.empty(self.embed_dim, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        _attach_replicated_hf_key(self.weight, f"{prefix}.weight")
        _attach_replicated_hf_key(self.bias, f"{prefix}.bias")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = F.conv3d(
            hidden_states.to(dtype=self.weight.dtype),
            self.weight,
            self.bias,
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
        )
        return hidden_states.view(-1, self.embed_dim)


# ============================================================================ #
# Qwen3-VL vision tower                                                      #
# ============================================================================ #


class GR00TN17QwenVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.theta = float(theta)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class GR00TN17QwenVisionMLP(nn.Module):
    def __init__(self, config: Any, *, prefix: str, params_dtype=None, device=None) -> None:
        super().__init__()
        self.linear_fc1 = GR00TN17QwenLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            prefix=f"{prefix}.linear_fc1",
            params_dtype=params_dtype,
            device=device,
        )
        self.linear_fc2 = GR00TN17QwenLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            prefix=f"{prefix}.linear_fc2",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(_gelu_tanh(self.linear_fc1(hidden_state)))


class GR00TN17QwenVisionPatchMerger(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        prefix: str,
        use_postshuffle_norm: bool,
        params_dtype=None,
        device=None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = bool(use_postshuffle_norm)
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = GR00TN17QwenLayerNorm(
            norm_size,
            eps=1e-6,
            prefix=f"{prefix}.norm",
            params_dtype=params_dtype,
            device=device,
        )
        self.linear_fc1 = GR00TN17QwenLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            prefix=f"{prefix}.linear_fc1",
            params_dtype=params_dtype,
            device=device,
        )
        self.linear_fc2 = GR00TN17QwenLinear(
            self.hidden_size,
            config.out_hidden_size,
            bias=True,
            prefix=f"{prefix}.linear_fc2",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x)
        x = x.view(-1, self.hidden_size)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class GR00TN17QwenVisionAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        prefix: str,
        params_dtype=None,
        device=None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.dim = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_backend = attention_backend
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            causal=False,
            backend=attention_backend,
            backend_kwargs=_attention_backend_kwargs(attention_backend),
        )
        self.qkv = GR00TN17QwenLinear(
            self.dim,
            self.dim * 3,
            bias=True,
            prefix=f"{prefix}.qkv",
            params_dtype=params_dtype,
            device=device,
        )
        self.proj = GR00TN17QwenLinear(
            self.dim,
            self.dim,
            bias=True,
            prefix=f"{prefix}.proj",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        query_states, key_states, value_states = (
            qkv.reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )
        if self.attention_backend == "sdpa" or (
            self.attention_backend == "flashinfer" and not query_states.is_cuda
        ):
            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)
            # Per-image (block-diagonal) attention. ``seqlens`` is the precomputed
            # python segment-size list; when provided this avoids the
            # ``cu_seqlens.tolist()`` host sync and is CUDA-graph-capturable.
            # Byte-identical to deriving the lengths inline (same torch.split).
            if seqlens is None:
                seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            splits = [
                torch.split(tensor, seqlens, dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            outs = [
                _attention_forward(
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                )
                for q, k, v in zip(*splits)
            ]
            out = torch.cat(outs, dim=1).reshape(seq_length, -1).contiguous()
        else:
            out = self.attn(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
            ).reshape(seq_length, -1).contiguous()
        return self.proj(out)


class GR00TN17QwenVisionBlock(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        prefix: str,
        params_dtype=None,
        device=None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.norm1 = GR00TN17QwenLayerNorm(
            config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.norm1",
            params_dtype=params_dtype,
            device=device,
        )
        self.norm2 = GR00TN17QwenLayerNorm(
            config.hidden_size,
            eps=1e-6,
            prefix=f"{prefix}.norm2",
            params_dtype=params_dtype,
            device=device,
        )
        self.attn = GR00TN17QwenVisionAttention(
            config,
            prefix=f"{prefix}.attn",
            params_dtype=params_dtype,
            device=device,
            attention_backend=attention_backend,
        )
        self.mlp = GR00TN17QwenVisionMLP(
            config,
            prefix=f"{prefix}.mlp",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            seqlens=seqlens,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


@dataclass
class GR00TN17QwenVisionOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: list[torch.Tensor]


class GR00TN17QwenVisionModel(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        prefix: str,
        params_dtype=None,
        device=None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = int(config.spatial_merge_size)
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = GR00TN17QwenConv3dPatchEmbed(
            config,
            prefix=f"{prefix}.patch_embed.proj",
            params_dtype=params_dtype,
            device=device,
        )
        self.pos_embed = GR00TN17QwenEmbedding(
            config.num_position_embeddings,
            config.hidden_size,
            prefix=f"{prefix}.pos_embed",
            params_dtype=params_dtype,
            device=device,
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = GR00TN17QwenVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                GR00TN17QwenVisionBlock(
                    config,
                    prefix=f"{prefix}.blocks.{idx}",
                    params_dtype=params_dtype,
                    device=device,
                    attention_backend=attention_backend,
                )
                for idx in range(config.depth)
            ]
        )
        self.merger = GR00TN17QwenVisionPatchMerger(
            config,
            prefix=f"{prefix}.merger",
            use_postshuffle_norm=False,
            params_dtype=params_dtype,
            device=device,
        )
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [
                GR00TN17QwenVisionPatchMerger(
                    config,
                    prefix=f"{prefix}.deepstack_merger_list.{idx}",
                    use_postshuffle_norm=True,
                    params_dtype=params_dtype,
                    device=device,
                )
                for idx in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.weight.dtype

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        max_hw = max(max(h, w) for _, h, w in grid_thw.tolist())
        freq_table = self.rotary_pos_emb(max_hw)
        # (total_patches, 2) merge-block-ordered (row, col) ids. Identical
        # construction to the (now-removed) inline block-expand loop — same
        # ordering and values, verified byte-for-byte — reused from the shared
        # qwen3_vl helper. Indexing ``freq_table[pos_ids]`` is unchanged.
        pos_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        return freq_table[pos_ids.to(freq_table.device)].flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid_thw_list:
            del t
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=self.pos_embed.weight.dtype, device=device
        )
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws)]
        )
        out = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            out.append(pos_embed)
        return torch.cat(out)

    def _vision_preamble(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Grid-only vision preamble — NOT CUDA-graph-capturable.

        Depends solely on ``grid_thw`` (constant for a fixed image size), so it
        is hoisted out of the capturable core: ``fast_pos_embed_interpolate``
        (``.tolist()``), ``rot_pos_emb`` (Python loop), and the
        ``repeat_interleave`` cu_seqlens each force a host sync.
        """
        pos_embed = self.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        # Per-frame attention boundaries — identical to the (now-removed) inline
        # ``repeat_interleave(...).cumsum(...).pad`` (verified byte-for-byte),
        # reused from the shared qwen3_vl helper.
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        # Per-image segment sizes as a Python list (host sync here, in the
        # non-capturable preamble) so the capturable core's attention can
        # ``torch.split`` without a runtime ``.tolist()``.
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        return pos_embed, rotary_pos_emb, cu_seqlens, seqlens

    def _vision_core(
        self,
        hidden_states: torch.Tensor,
        pos_embed: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqlens: list[int] | None = None,
    ) -> GR00TN17QwenVisionOutput:
        """Pure-tensor vision core — CUDA-graph-capturable.

        Same ops/order as the original monolithic forward; only the grid-only
        preamble has been hoisted to :meth:`_vision_preamble`.
        """
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embed
        seq_len, _ = hidden_states.size()
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        deepstack = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                seqlens=seqlens,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack.append(self.deepstack_merger_list[idx](hidden_states))
        merged = self.merger(hidden_states)
        return GR00TN17QwenVisionOutput(
            last_hidden_state=hidden_states,
            pooler_output=merged,
            deepstack_features=deepstack,
        )

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> GR00TN17QwenVisionOutput:
        """Eager ViT forward. The CUDA-graph path captures ``_vision_core`` as
        part of the merged backbone core (see
        :meth:`GR00TN17NativeQwen3VLModel._backbone_core`); this module stays
        stateless."""
        pos_embed, rotary_pos_emb, cu_seqlens, seqlens = self._vision_preamble(grid_thw)
        return self._vision_core(
            hidden_states, pos_embed, rotary_pos_emb, cu_seqlens, seqlens=seqlens
        )


# ============================================================================ #
#  Qwen3-VL text model                                                        #
# ============================================================================ #


class GR00TN17QwenTextRotaryEmbedding(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0
        self.mrope_section = list(config.mrope_section)

    @staticmethod
    def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        freqs_t = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = self.inv_freq.to(device=x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def _apply_text_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GR00TN17QwenTextAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_idx: int,
        *,
        prefix: str,
        params_dtype=None,
        device=None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_backend = attention_backend
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            causal=True,
            backend=attention_backend,
            backend_kwargs=_attention_backend_kwargs(attention_backend),
        )
        bias = bool(config.attention_bias)
        self.q_proj = GR00TN17QwenLinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=bias,
            prefix=f"{prefix}.q_proj",
            params_dtype=params_dtype,
            device=device,
        )
        self.k_proj = GR00TN17QwenLinear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
            prefix=f"{prefix}.k_proj",
            params_dtype=params_dtype,
            device=device,
        )
        self.v_proj = GR00TN17QwenLinear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bias,
            prefix=f"{prefix}.v_proj",
            params_dtype=params_dtype,
            device=device,
        )
        self.o_proj = GR00TN17QwenLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=bias,
            prefix=f"{prefix}.o_proj",
            params_dtype=params_dtype,
            device=device,
        )
        self.q_norm = GR00TN17QwenRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}.q_norm",
            params_dtype=params_dtype,
            device=device,
        )
        self.k_norm = GR00TN17QwenRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}.k_norm",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.num_heads, self.head_dim)
        kv_shape = (*input_shape, self.num_key_value_heads, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(q_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(kv_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_text_rotary(q, k, cos, sin)
        if (
            attention_mask is None
            and (self.attention_backend != "flashinfer" or q.is_cuda)
        ):
            out = self.attn(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
            )
        else:
            out = _attention_forward(
                q,
                k,
                v,
                attention_mask=attention_mask,
                scaling=self.scaling,
                num_key_value_groups=self.num_key_value_groups,
                is_causal=attention_mask is None,
            )
        out = out.reshape(*input_shape, -1).contiguous()
        return self.o_proj(out)


class GR00TN17QwenTextMLP(nn.Module):
    def __init__(self, config: Any, *, prefix: str, params_dtype=None, device=None) -> None:
        super().__init__()
        self.gate_proj = GR00TN17QwenLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            prefix=f"{prefix}.gate_proj",
            params_dtype=params_dtype,
            device=device,
        )
        self.up_proj = GR00TN17QwenLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            prefix=f"{prefix}.up_proj",
            params_dtype=params_dtype,
            device=device,
        )
        self.down_proj = GR00TN17QwenLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            prefix=f"{prefix}.down_proj",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class GR00TN17QwenTextDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        layer_idx: int,
        *,
        prefix: str,
        params_dtype=None,
        device=None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.self_attn = GR00TN17QwenTextAttention(
            config,
            layer_idx,
            prefix=f"{prefix}.self_attn",
            params_dtype=params_dtype,
            device=device,
            attention_backend=attention_backend,
        )
        self.mlp = GR00TN17QwenTextMLP(
            config,
            prefix=f"{prefix}.mlp",
            params_dtype=params_dtype,
            device=device,
        )
        self.input_layernorm = GR00TN17QwenRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            params_dtype=params_dtype,
            device=device,
        )
        self.post_attention_layernorm = GR00TN17QwenRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class GR00TN17QwenTextModel(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        num_layers: int,
        prefix: str,
        params_dtype=None,
        device=None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = GR00TN17QwenEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
            params_dtype=params_dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                GR00TN17QwenTextDecoderLayer(
                    config,
                    idx,
                    prefix=f"{prefix}.layers.{idx}",
                    params_dtype=params_dtype,
                    device=device,
                    attention_backend=attention_backend,
                )
                for idx in range(num_layers)
            ]
        )
        self.norm = GR00TN17QwenRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}.norm",
            params_dtype=params_dtype,
            device=device,
        )
        self.rotary_emb = GR00TN17QwenTextRotaryEmbedding(config)

    def _causal_mask(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        text_position_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if text_position_ids is None and (
            attention_mask is None or bool(attention_mask.bool().all())
        ):
            return None
        batch, seq_len, _ = inputs_embeds.shape
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        if text_position_ids is None:
            causal = torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=inputs_embeds.device,
            ).triu(diagonal=1)
            causal = causal[None, None, :, :].expand(batch, 1, -1, -1)
        else:
            causal = text_position_ids[:, None, :] > text_position_ids[:, :, None]
            causal = causal[:, None, :, :].to(device=inputs_embeds.device)
        mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        mask = mask.masked_fill(causal, min_dtype)
        if attention_mask is not None:
            key_padding = attention_mask[:, None, None, :].to(device=inputs_embeds.device).bool()
            mask = mask.masked_fill(~key_padding, min_dtype)
        return mask

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_index: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Add ``visual_embeds`` to the visual-token rows of ``hidden_states``.

        Uses ``index_add_`` with a precomputed integer ``visual_index`` instead
        of boolean-mask indexing (``hidden_states[mask] += ...``), which calls
        ``nonzero`` and is not CUDA-graph-capturable. Byte-identical: each visual
        row receives exactly its one ``visual_embeds`` row, same order.
        """
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat.index_add_(0, visual_index, visual_embeds)
        return hidden_states

    def _llm_core(
        self,
        inputs_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        visual_index: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        """Pure-tensor decoder stack — CUDA-graph-capturable.

        ``attention_mask`` is either ``None`` for the original no-padding fast
        path or a precomputed 4-D additive causal/key-padding mask for graph
        sequence buckets. Computing the mask stays in the host-sync preamble;
        the captured region only reads the static mask buffer.
        """
        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            if layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_index, deepstack_visual_embeds[layer_idx]
                )
        return SimpleNamespace(
            last_hidden_state=self.norm(hidden_states),
            pre_norm_hidden_state=hidden_states,
            hidden_states=None,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
        output_hidden_states: bool,
    ) -> SimpleNamespace:
        """Eager decoder forward. The CUDA-graph path captures ``_llm_core`` as
        part of the merged backbone core (see
        :meth:`GR00TN17NativeQwen3VLModel._backbone_core`); this stays stateless."""
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required.")
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
            # Footgun guard: in the 4-row layout, row 0 is the causal-mask key
            # (token i attends j iff row0[j] <= row0[i] in _causal_mask), so it must
            # be strictly increasing per sample, i.e. real text/sequence positions.
            # Passing an M-RoPE coordinate row here (e.g. the temporal row, where every
            # image patch in a frame shares one value) silently makes image tokens
            # attend each other bidirectionally and diverges from the standard-causal
            # backbone. Production never enters this branch (the runner feeds 3-row
            # M-RoPE via prepare_position_ids); fail loudly rather than mis-attend.
            if text_position_ids.shape[-1] > 1 and not bool(
                (text_position_ids[:, 1:] > text_position_ids[:, :-1]).all()
            ):
                raise ValueError(
                    "GR00TN17QwenTextModel got 4-row position_ids whose row 0 is not "
                    "strictly increasing. Row 0 must be the text/sequence positions used "
                    "for causal masking; you likely passed an M-RoPE coordinate row "
                    "(e.g. temporal) by mistake. Feed 3-row M-RoPE position_ids "
                    "(as backbone.prepare_position_ids does), or a valid text-position row."
                )
        elif position_ids.ndim == 3:
            text_position_ids = None
            rope_position_ids = position_ids
        else:
            text_position_ids = position_ids
            rope_position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        attention_mask_4d = self._causal_mask(
            inputs_embeds, attention_mask, text_position_ids
        )
        position_embeddings = self.rotary_emb(inputs_embeds, rope_position_ids)
        # Flat integer indices of visual-token rows (host sync here, outside any
        # captured region) for the capture-safe deepstack add.
        visual_index = None
        if deepstack_visual_embeds is not None and visual_pos_masks is not None:
            visual_index = (
                visual_pos_masks.reshape(-1)
                .to(inputs_embeds.device)
                .nonzero(as_tuple=True)[0]
            )
        hidden_states = inputs_embeds
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask_4d,
            )
            if visual_index is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_index,
                    deepstack_visual_embeds[layer_idx],
                )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        final_hidden_states = self.norm(hidden_states)
        return SimpleNamespace(
            last_hidden_state=final_hidden_states,
            # Pre-final-norm features — what the GR00T backbone consumes. Returned
            # directly (instead of a forward_pre_hook on ``norm``) so the value
            # survives CUDA-graph replay, where Python hooks do not re-fire.
            pre_norm_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


class GR00TN17NativeQwen3VLModel(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        select_layer: int,
        params_dtype=None,
        device=None,
        attention_backend: str = "sdpa",
        prefix: str = "backbone.model.model",
    ) -> None:
        super().__init__()
        self.config = config
        self.visual = GR00TN17QwenVisionModel(
            config.vision,
            prefix=f"{prefix}.visual",
            params_dtype=params_dtype,
            device=device,
            attention_backend=attention_backend,
        )
        num_layers = config.text.num_hidden_layers if select_layer < 0 else select_layer
        self.language_model = GR00TN17QwenTextModel(
            config.text,
            num_layers=num_layers,
            prefix=f"{prefix}.language_model",
            params_dtype=params_dtype,
            device=device,
            attention_backend=attention_backend,
        )

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = position_width.repeat(llm_grid_h * llm_grid_t)
        position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
        position_temporal = (
            position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
        )
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision.spatial_merge_size
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        mrope_position_deltas = []
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_type = input_type[attention_mask[batch_idx].bool()]
            groups = []
            for key, group in itertools.groupby(enumerate(input_type.tolist()), lambda x: x[1]):
                group = list(group)
                groups.append((key, group[0][0], group[-1][0] + 1))
            current_pos = 0
            pos_list = []
            for modality_type, start_idx, end_idx in groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    pos_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos,
                        grid_thw,
                        1,
                        spatial_merge_size,
                        device=input_ids.device,
                    )
                    pos_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(pos_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        return position_ids, torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> GR00TN17QwenVisionOutput:
        vision_output = self.visual(
            pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw
        )
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_output.pooler_output = torch.split(vision_output.pooler_output, split_sizes)
        return vision_output

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor,
    ) -> torch.Tensor:
        mask = input_ids == self.config.image_token_id
        expanded = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if inputs_embeds[expanded].numel() != image_features.numel():
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={int(mask.sum())}, features={tuple(image_features.shape)}"
            )
        return expanded

    def _backbone_core(
        self,
        *,
        pixel_values: torch.Tensor,
        pos_embed: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_ids: torch.Tensor,
        image_index: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask_4d: torch.Tensor | None,
        seqlens: list[int],
    ) -> SimpleNamespace:
        """Merged ViT + multimodal fuse + LLM decoder — ONE capturable region.

        Every host-sync preamble (vision grid geometry, 3-D M-RoPE position ids
        -> cos/sin, the image-token ``image_index``) is computed by the caller
        and passed in as buffers, so this entire forward is pure tensor ops and
        captures as a single CUDA graph. ``index_copy_`` (capture-safe) replaces
        the boolean ``masked_scatter`` for vision-token fusion. Deepstack
        features flow from the ViT taps into the decoder within the same graph.
        """
        vision_out = self.visual._vision_core(
            pixel_values, pos_embed, rotary_pos_emb, cu_seqlens, seqlens=seqlens
        )
        merged = vision_out.pooler_output  # (total_vision_tokens, hidden)
        deepstack = vision_out.deepstack_features
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        inputs_embeds.view(-1, inputs_embeds.shape[-1]).index_copy_(
            0, image_index, merged.to(inputs_embeds.dtype)
        )
        return self.language_model._llm_core(
            inputs_embeds, (cos, sin), image_index, deepstack, attention_mask_4d
        )

    def backbone_graph_plan(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        graph_seq_len_buckets: tuple[int, ...] | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> tuple[Any, dict[str, torch.Tensor], tuple, dict[str, torch.Tensor]] | None:
        """Host-sync preamble for the single-graph backbone path.

        Returns ``(core_fn, buffers, key, model_inputs)`` ready for the
        **runner** to capture/replay :meth:`_backbone_core`, or ``None`` when the
        input is not graph-eligible (then the runner falls back to the eager
        :meth:`forward`).
        Eligible == on CUDA, 3-row M-RoPE position ids, dumping off. When
        ``graph_seq_len_buckets`` is provided, left-pad text tensors to the
        smallest bucket that fits the current sequence, preserving the official
        Qwen tokenizer's left-padding semantics.
        No capture happens here, so this module holds no graph state.
        """
        if not pixel_values.is_cuda or output_hidden_states:
            return None
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_token_type_ids = mm_token_type_ids.masked_fill(input_ids == self.config.image_token_id, 1)
        seq_len = int(input_ids.shape[1])
        if graph_seq_len_buckets is not None:
            bucket = next((b for b in graph_seq_len_buckets if seq_len <= b), None)
            if bucket is None:
                return None
            pad = int(bucket) - seq_len
            if pad > 0:
                input_ids = F.pad(input_ids, (pad, 0), value=0)
                attention_mask = F.pad(attention_mask, (pad, 0), value=0)
                mm_token_type_ids = F.pad(mm_token_type_ids, (pad, 0), value=0)
                if position_ids is not None:
                    position_ids = F.pad(position_ids, (pad, 0), value=0)
        if position_ids is None:
            position_ids, _ = self.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        if not (
            position_ids.ndim == 3
            and position_ids.shape[0] == 3
        ):
            return None
        pixel_values = pixel_values.type(self.visual.dtype)
        device = pixel_values.device
        pos_embed, rotary_pos_emb, cu_seqlens, seqlens = self.visual._vision_preamble(
            image_grid_thw
        )
        rotary_pos_emb = rotary_pos_emb.to(device)
        # 3-D M-RoPE cos/sin (pixel_values is only a dtype/device reference here).
        cos, sin = self.language_model.rotary_emb(pixel_values, position_ids)
        mask_ref = torch.empty(
            (*input_ids.shape, 1),
            dtype=self.language_model.embed_tokens.weight.dtype,
            device=input_ids.device,
        )
        attention_mask_4d = self.language_model._causal_mask(
            mask_ref,
            attention_mask,
            text_position_ids=None,
        )
        image_index = (input_ids.reshape(-1) == self.config.image_token_id).nonzero(
            as_tuple=True
        )[0]
        buffers = {
            "pixel_values": pixel_values,
            "pos_embed": pos_embed,
            "rotary_pos_emb": rotary_pos_emb,
            "cu_seqlens": cu_seqlens,
            "input_ids": input_ids,
            "image_index": image_index,
            "cos": cos,
            "sin": sin,
        }
        if attention_mask_4d is not None:
            buffers["attention_mask_4d"] = attention_mask_4d
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
            "position_ids": position_ids,
        }
        key = (
            tuple(pixel_values.shape),
            pixel_values.dtype,
            tuple(input_ids.shape),
            tuple(cos.shape),
            tuple(image_index.shape),
            None if attention_mask_4d is None else tuple(attention_mask_4d.shape),
            tuple(seqlens),
        )

        # ``seqlens`` is a Python list (not a graph buffer); bake it into the core
        # via a closure (the key includes it, so reuse stays shape-correct).
        def core(**buf: torch.Tensor) -> SimpleNamespace:
            return self._backbone_core(
                pixel_values=buf["pixel_values"],
                pos_embed=buf["pos_embed"],
                rotary_pos_emb=buf["rotary_pos_emb"],
                cu_seqlens=buf["cu_seqlens"],
                input_ids=buf["input_ids"],
                image_index=buf["image_index"],
                cos=buf["cos"],
                sin=buf["sin"],
                attention_mask_4d=buf.get("attention_mask_4d"),
                seqlens=seqlens,
            )

        return core, buffers, key, model_inputs

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> SimpleNamespace:
        """Eager backbone forward (byte-identical reference). The CUDA-graph path
        captures :meth:`_backbone_core` from the runner via
        :meth:`backbone_graph_plan`; this module holds no graph state."""
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_token_type_ids = mm_token_type_ids.masked_fill(input_ids == self.config.image_token_id, 1)
        if position_ids is None:
            position_ids, _ = self.get_rope_index(
                input_ids,
                mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask,
            )
        inputs_embeds = self.get_input_embeddings()(input_ids)
        image_outputs = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_embeds)
        # Capture-safe scatter (matches the graph path): place each merged vision
        # token onto its image-token row by precomputed index. Byte-identical to
        # the boolean ``masked_scatter`` (same positions/order).
        image_index = (input_ids.reshape(-1) == self.config.image_token_id).nonzero(
            as_tuple=True
        )[0]
        inputs_embeds.view(-1, inputs_embeds.shape[-1]).index_copy_(
            0, image_index, image_embeds
        )
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = image_outputs.deepstack_features
        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            output_hidden_states=output_hidden_states,
        )


class GR00TN17NativeQwen3VLForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Any,
        *,
        select_layer: int,
        params_dtype=None,
        device=None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = GR00TN17NativeQwen3VLModel(
            config,
            select_layer=select_layer,
            params_dtype=params_dtype,
            device=device,
            attention_backend=attention_backend,
        )
        self.lm_head = GR00TN17QwenLinear(
            config.text.hidden_size,
            config.text.vocab_size,
            bias=False,
            prefix="backbone.model.lm_head",
            params_dtype=params_dtype,
            device=device,
        )

    def forward(self, **kwargs: Any) -> SimpleNamespace:
        return self.model(**kwargs)


__all__ = ["GR00TN17NativeQwen3VLForConditionalGeneration"]
