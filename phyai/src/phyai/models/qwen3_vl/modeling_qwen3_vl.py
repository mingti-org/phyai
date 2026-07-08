"""Qwen3-VL Modeling"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.engine_config import get_engine_config, resolve_engine_defaults
from phyai.layers.attention.attention.layer import Attention
from phyai.layers.conv import Conv3d
from phyai.layers.layer_norm import LayerNorm, RMSNorm
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.rotary_embedding import InterleavedMRotaryEmbedding, rotate_half
from phyai.layers.transformer_block import TransformerBlock
from phyai.layers.vocab_embedding import ParallelLMHead, VocabParallelEmbedding
from phyai.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from phyai.weights.shards import replicated


if TYPE_CHECKING:
    from phyai.layers.attention import ARAttnCtx


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Per-frame attention boundaries (``cu_seqlens``) for the packed patches.

    The vision tower flattens every image / video frame into one ragged sequence
    of patches and runs **block-diagonal** attention: each temporal frame (an
    ``h * w`` patch block) is its own window, so a patch only attends to patches
    of the *same* frame, never across frames. This returns the cumulative-offset
    ``indptr`` marking those frame boundaries — exactly the ``cu_seqlens`` the
    attention op consumes (``self.attn(..., cu_seqlens_q=cu, cu_seqlens_kv=cu)``).

    Parameters
    ----------
    grid_thw:
        ``(num_images, 3)`` int tensor; row ``i`` is ``(t, h, w)`` for image /
        video ``i`` in **patch** units — ``t`` temporal frames, each ``h * w``
        patches. A still image has ``t = 1``; a video has ``t > 1``.

    Returns
    -------
    cu_seqlens:
        ``(num_frames + 1,)`` int32, ``num_frames = sum(t_i)``. Standard indptr
        ``[0, len_0, len_0 + len_1, ...]``; frame ``f`` occupies the packed slice
        ``[cu[f], cu[f + 1])`` and the final entry is the total patch count.

    Example
    -------
    ``grid_thw = [[1, 2, 2], [2, 2, 3]]`` — one image (1 frame, 2x2) plus one
    video (2 frames, 2x3):

    1. ``grid_thw[:, 1] * grid_thw[:, 2]`` -> patches per frame, per image:
       ``[4, 6]``.
    2. ``repeat_interleave(..., grid_thw[:, 0])`` repeats each by its frame count
       ``t`` -> per-frame counts ``[4, 6, 6]`` (the image's 1 frame, then the
       video's 2 frames).
    3. ``cumsum`` -> per-frame end offsets ``[4, 10, 16]``.
    4. ``F.pad(..., (1, 0))`` prepends 0 -> ``[0, 4, 10, 16]``.

    16 patches pack into 3 windows: frame 0 ``[0, 4)`` (image), frame 1 ``[4, 10)``
    and frame 2 ``[10, 16)`` (the video's two frames, mutually invisible). A
    single-frame image collapses to ``[0, h*w]`` — one window, a no-op block mask.
    """
    # Patches per frame (h * w), repeated once per temporal frame (t), then
    # accumulated into running per-frame end-offsets.
    cu = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    # Prepend 0 -> standard indptr [0, len_0, len_0 + len_1, ...].
    return F.pad(cu, (1, 0), value=0)


def get_vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    """(row, col) position ids per patch for the axial 2-D vision RoPE.

    Returns ``(total_patches, 2)`` long. Within each image the patches are laid
    out in spatial-merge-block order so the merger's ``view(-1, merge**2 * C)``
    groups the right 2x2 neighborhoods.
    """
    device = grid_thw.device
    position_ids: list[torch.Tensor] = []
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
        position_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def get_vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear-interpolation corner indices/weights into the learned pos-embed.

    The learned position embedding is a square ``num_grid_per_side`` grid;
    each patch position is bilinearly interpolated from its 4 grid corners.
    Returns ``(4, total_patches)`` long indices and ``(4, total_patches)`` float
    weights, reordered into spatial-merge-block order to match the patches.
    """
    side = num_grid_per_side
    merge = spatial_merge_size
    device = grid_thw.device
    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)
        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)
        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor
        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]
        h_idx = torch.arange(h, device=device).view(h // merge, merge)
        w_idx = torch.arange(w, device=device).view(w // merge, merge)
        reorder = (
            (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
            .transpose(1, 2)
            .flatten()
            .repeat(t)
        )
        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])
    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return bilinear_indices, bilinear_weights


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Axial 2-D vision RoPE. ``inv_freq`` over ``dim`` (= head_dim // 2)."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: (num_patches, 2) -> freqs (num_patches, 2 * (dim//2)).
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.to(position_ids.device)
        return freqs.flatten(1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half RoPE for the vision tower. q/k: ``(seq, heads, dim)``."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()  # (seq, 1, dim)
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Conv3d patchifier: ``(seq, C*tps*p*p) -> (seq, hidden)``."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel,
            stride=kernel,
            bias=True,
            dtype=params_dtype,
            prefix=f"{prefix}.proj" if prefix else "",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim
        )
        return hidden_states


class Qwen3VLVisionPatchMerger(nn.Module):
    """Merge ``spatial_merge_size**2`` patches into one ``out_hidden_size`` token.

    Two norm placements: pre-shuffle (``use_postshuffle_norm=False``) norms the
    per-patch ``hidden`` width before the view-merge (the main merger);
    post-shuffle norms the merged ``hidden * merge**2`` width (the deepstack
    mergers).
    """

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        use_postshuffle_norm: bool = False,
        params_dtype: torch.dtype | None = None,
        norm_backend: str = "flashinfer",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.merged_dim = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.merged_dim if use_postshuffle_norm else config.hidden_size
        self.norm = LayerNorm(
            norm_dim,
            eps=1e-6,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            prefix=f"{prefix}.norm" if prefix else "",
        )
        self.linear_fc1 = ReplicatedLinear(
            self.merged_dim,
            self.merged_dim,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear_fc1" if prefix else "",
        )
        self.linear_fc2 = ReplicatedLinear(
            self.merged_dim,
            config.out_hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear_fc2" if prefix else "",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.merged_dim) if self.use_postshuffle_norm else x)
        x = x.view(-1, self.merged_dim)
        x, _ = self.linear_fc1(x)
        x = F.gelu(x)
        x, _ = self.linear_fc2(x)
        return x


class Qwen3VLVisionMLP(nn.Module):
    """Plain ``fc1 -> act -> fc2`` MLP with bias."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.linear_fc1 = ReplicatedLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear_fc1" if prefix else "",
        )
        self.linear_fc2 = ReplicatedLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear_fc2" if prefix else "",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.linear_fc1(x)
        x = F.gelu(x, approximate="tanh")
        x, _ = self.linear_fc2(x)
        return x


class Qwen3VLVisionAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str = "flashinfer",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = ReplicatedLinear(
            self.dim,
            self.dim * 3,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.qkv" if prefix else "",
        )
        self.proj = ReplicatedLinear(
            self.dim,
            self.dim,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.proj" if prefix else "",
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            causal=False,
            backend=attn_backend,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv, _ = self.qkv(hidden_states)
        q, k, v = (
            qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )  # each (seq, num_heads, head_dim)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        # Attention expects ragged 3-D (N, H, D) with cu_seqlens for the
        # block-diagonal per-frame mask.
        out = self.attn(q, k, v, cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens)
        out = out.reshape(seq_length, -1)
        out, _ = self.proj(out)
        return out


class Qwen3VLVisionBlock(nn.Module):
    """Prenorm ViT block: ``h + attn(ln1(h))`` then ``h + mlp(ln2(h))``."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str = "flashinfer",
        norm_backend: str = "flashinfer",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            prefix=f"{prefix}.norm1" if prefix else "",
        )
        self.norm2 = LayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            prefix=f"{prefix}.norm2" if prefix else "",
        )
        self.attn = Qwen3VLVisionAttention(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            prefix=f"{prefix}.attn" if prefix else "",
        )
        self.mlp = Qwen3VLVisionMLP(
            config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.mlp" if prefix else "",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(nn.Module):
    """Qwen3-VL native ViT vision tower."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "visual",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        if device is None:
            device = get_engine_config().device.target
        self.config = config
        self.prefix = prefix
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_unit
        self.num_grid_per_side = config.num_grid_per_side
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.patch_embed" if prefix else "",
        )
        # Learned position embedding table — a plain replicated parameter,
        # indexed by bilinear corner ids (not a vocab-parallel token lookup).
        self.pos_embed_weight = nn.Parameter(
            torch.zeros(
                config.num_position_embeddings,
                config.hidden_size,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        if prefix:
            self.pos_embed_weight.hf_keys = [(f"{prefix}.pos_embed.weight", None)]
            self.pos_embed_weight.weight_loader = replicated()

        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(config.head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(
                    config,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    prefix=f"{prefix}.blocks.{i}" if prefix else "",
                )
                for i in range(config.depth)
            ]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            config,
            use_postshuffle_norm=False,
            params_dtype=params_dtype,
            norm_backend=norm_backend,
            prefix=f"{prefix}.merger" if prefix else "",
        )
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config,
                    use_postshuffle_norm=True,
                    params_dtype=params_dtype,
                    norm_backend=norm_backend,
                    prefix=f"{prefix}.deepstack_merger_list.{j}" if prefix else "",
                )
                for j in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.pos_embed_weight.dtype

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns ``(merged_tokens, deepstack_features)``.

        ``merged_tokens``: ``(num_patches // merge**2, out_hidden_size)``.
        ``deepstack_features``: list of the same-shaped per-tap features.
        """
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = (
            self.pos_embed_weight[bilinear_indices] * bilinear_weights[:, :, None]
        ).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)  # (num_patches, head_dim//2)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self.deepstack_merger_list[idx](hidden_states)
                )

        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states, deepstack_feature_lists


class Qwen3VLTextModel(nn.Module):
    """Qwen3 decoder with interleaved 3-D M-RoPE + DeepStack injection."""

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "model.language_model",
        text_attn_kind: str = "attention",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        if device is None:
            device = get_engine_config().device.target
        self.config = config
        self.prefix = prefix
        self.text_attn_kind = text_attn_kind

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=params_dtype,
            prefix=f"{prefix}.embed_tokens" if prefix else "",
        )
        # TODO(wch): rope kernel can be fused here. currently fail back to eager.
        self.rotary_emb = InterleavedMRotaryEmbedding(
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            mrope_section=config.mrope_section,
            rope_theta=config.rope_theta,
            backend="eager",
            device=device,
        )
        layer_prefix = f"{prefix}.layers" if prefix else ""
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    attn_kind=text_attn_kind,
                    layer_idx=i,
                    attn_causal=True,
                    attn_bias=config.attention_bias,
                    attn_qk_norm=True,
                    rope=self.rotary_emb,
                    precompute_rope=True,
                    mlp_gated=True,
                    mlp_activation=config.hidden_act,
                    mlp_bias=False,
                    norm_type="rmsnorm",
                    norm_eps=config.rms_norm_eps,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    params_dtype=params_dtype,
                    prefix=f"{layer_prefix}.{i}" if layer_prefix else "",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            backend=norm_backend,
            dtype=params_dtype,
            prefix=f"{prefix}.norm" if prefix else "",
        )

    @staticmethod
    def _deepstack_process(
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Add a deepstack feature onto the visual-token positions only."""
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        # No need to clone in inference framework
        # hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = (
            hidden_states[visual_pos_masks, :] + visual_embeds
        )
        return hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: "ARAttnCtx | None" = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        return_pre_norm_hidden_state: bool = False,
    ) -> torch.Tensor:
        """The caller computes ``(cos, sin)`` once via
        :meth:`RotaryEmbedding.get_cos_sin` and threads them through every layer.
        """
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)

        hidden_states = inputs_embeds
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cos=cos, sin=sin, attn_ctx=attn_ctx)
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        if return_pre_norm_hidden_state:
            return hidden_states
        return self.norm(hidden_states)


class Qwen3VLModel(nn.Module):
    """Vision tower + text decoder with multimodal merge and 3-D M-RoPE.

    The vision/text fusion mirrors HF: encode images/videos, ``masked_scatter``
    the merged vision tokens onto the placeholder positions of the text
    embedding, build 3-D M-RoPE position ids from the token-type layout, and run
    the decoder with DeepStack features injected into its first layers.
    """

    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        prefix: str = "model",
        text_attn_kind: str = "attention",
        vision_attn_backend: str | None = None,
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        self.spatial_merge_size = config.vision.spatial_merge_size
        self.visual = Qwen3VLVisionModel(
            config.vision,
            params_dtype=vision_params_dtype or params_dtype,
            attn_backend=vision_attn_backend or attn_backend,
            norm_backend=norm_backend,
            device=device,
            prefix=f"{prefix}.visual" if prefix else "visual",
        )
        self.language_model = Qwen3VLTextModel(
            config.text,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            device=device,
            prefix=f"{prefix}.language_model" if prefix else "language_model",
            text_attn_kind=text_attn_kind,
        )

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode images -> (per-image merged token lists, deepstack features)."""
        pixel_values = pixel_values.type(self.visual.dtype)
        merged, deepstack = self.visual(pixel_values, image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.spatial_merge_size**2).tolist()
        return list(torch.split(merged, split_sizes)), deepstack

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(position_ids[3,B,S], mrope_deltas[B,1])``.

        Token types are derived directly from ``input_ids`` against the
        configured image/video token ids.
        """
        config = self.config
        spatial_merge_size = self.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id

        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw = video_grid_thw.clone()
            video_grid_thw[:, 0] = 1

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_iter = iter(image_grid_thw) if image_grid_thw is not None else None
        video_iter = iter(video_grid_thw) if video_grid_thw is not None else None

        for batch_idx, current_input_ids in enumerate(input_ids):
            ids = current_input_ids
            if attention_mask is not None:
                ids = ids[attention_mask[batch_idx].bool()]
            token_type = torch.zeros_like(ids)
            token_type[ids == image_token_id] = 1
            token_type[ids == video_token_id] = 2

            groups = []
            for key, group in itertools.groupby(
                enumerate(token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                groups.append((key, group[0][0], group[-1][0] + 1))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in groups:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(image_iter if modality_type == 1 else video_iter)
                    llm_pos_ids_list.append(
                        self._vision_position_ids(
                            current_pos, grid_thw, spatial_merge_size
                        )
                    )
                    current_pos += (
                        int(max(grid_thw[1], grid_thw[2])) // spatial_merge_size
                    )
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                    llm_positions.to(position_ids.device)
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(ids))
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas

    @staticmethod
    def _vision_position_ids(
        start_position: int,
        grid_thw: torch.Tensor,
        spatial_merge_size: int,
        time_interval: int = 1,
    ) -> torch.Tensor:
        """3-D (t, h, w) positions for one image/video's vision tokens."""
        device = grid_thw.device
        llm_grid_t = int(grid_thw[0])
        llm_grid_h = int(grid_thw[1]) // spatial_merge_size
        llm_grid_w = int(grid_thw[2]) // spatial_merge_size
        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_width = torch.arange(llm_grid_w, device=device) + start_position
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = position_width.repeat(llm_grid_h * llm_grid_t)
        position_height = position_height.repeat_interleave(llm_grid_w).repeat(
            llm_grid_t
        )
        position_temporal = (
            position_temporal.repeat_interleave(llm_grid_h * llm_grid_w)
            + start_position
        )
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def embed_multimodal(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Text embeddings with vision tokens scattered in + DeepStack features."""
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            image_mask = image_mask.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_image_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
            video_mask = video_mask.expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks, deepstack_visual_embeds = self._merge_deepstack(
            image_mask, video_mask, deepstack_image_embeds, deepstack_video_embeds
        )
        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    @staticmethod
    def _merge_deepstack(
        image_mask: torch.Tensor | None,
        video_mask: torch.Tensor | None,
        deepstack_image_embeds: list[torch.Tensor] | None,
        deepstack_video_embeds: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """Build the joint visual mask + deepstack embed list (HF parity)."""
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            merged = []
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                joint = img_embed.new_zeros(
                    int(visual_pos_masks.sum()), img_embed.shape[-1]
                )
                joint[image_mask_joint, :] = img_embed
                joint[video_mask_joint, :] = vid_embed
                merged.append(joint)
            return visual_pos_masks, merged
        if image_mask is not None:
            return image_mask[..., 0], deepstack_image_embeds
        if video_mask is not None:
            return video_mask[..., 0], deepstack_video_embeds
        return None, None

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
        return_pre_norm_hidden_state: bool = False,
    ) -> torch.Tensor:
        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = (
            self.embed_multimodal(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
        )

        if position_ids is None:
            if image_grid_thw is not None or video_grid_thw is not None:
                position_ids, _ = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                )
            else:
                seq = torch.arange(input_ids.shape[1], device=input_ids.device)
                position_ids = seq.view(1, 1, -1).expand(3, input_ids.shape[0], -1)

        cos, sin = self.language_model.rotary_emb.get_cos_sin(position_ids)
        return self.language_model(
            inputs_embeds,
            cos=cos,
            sin=sin,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            return_pre_norm_hidden_state=return_pre_norm_hidden_state,
        )


class Qwen3VLForConditionalGeneration(nn.Module):
    """Qwen3-VL with the LM head.

    HF state-dict layout::

        model.visual.*           (vision tower)
        model.language_model.*   (text decoder)
        lm_head.weight
    """

    def __init__(
        self,
        config: Qwen3VLConfig,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
        device: torch.device | str | None = None,
        text_attn_kind: str = "attention",
        vision_attn_backend: str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        self.config = config
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = Qwen3VLModel(
            config,
            params_dtype=params_dtype,
            vision_params_dtype=vision_params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            device=device,
            prefix=model_prefix,
            text_attn_kind=text_attn_kind,
            vision_attn_backend=vision_attn_backend,
        )
        tied = (
            self.model.language_model.embed_tokens.weight
            if config.tie_word_embeddings
            else None
        )
        self.lm_head = ParallelLMHead(
            config.text.hidden_size,
            config.text.vocab_size,
            bias=False,
            tied_weight=tied,
            params_dtype=params_dtype,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
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
        return_pre_norm_hidden_state: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            return_pre_norm_hidden_state=return_pre_norm_hidden_state,
        )
        if return_pre_norm_hidden_state:
            return hidden_states
        logits = self.lm_head(hidden_states)
        return logits


__all__ = [
    "Qwen3VLVisionModel",
    "Qwen3VLTextModel",
    "Qwen3VLModel",
    "Qwen3VLForConditionalGeneration",
]
