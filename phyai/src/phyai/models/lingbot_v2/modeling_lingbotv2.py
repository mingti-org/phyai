"""LingBot-VLA 2.0 inference model modules.

This file owns only weight-bearing modules and tensor-geometry helpers.
Runners own attention planning, condition caching, and KV-cache storage;
the WS1 scheduler owns request packing and Euler integration.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

import phyai.parallel as P
from phyai.engine_config import get_engine_config, resolve_engine_defaults
from phyai.layers.attention.attention.layer import Attention
from phyai.layers.attention.diffusion import DiffusionAttention, DiffusionAttnCtx
from phyai.layers.conv import Conv3d
from phyai.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from phyai.layers.mlp import DenseMLP
from phyai.layers.quant import Bf16Spec
from phyai.layers.rotary_embedding import (
    InterleavedMRotaryEmbedding,
    apply_rotary_pos_emb,
)
from phyai.layers.transformer_block import TransformerBlock
from phyai.layers.vocab_embedding.layers import VocabParallelEmbedding
from phyai.models.lingbot_v2.configuration_lingbotv2 import (
    LingBotV2ExpertConfig,
    LingBotV2FlowMatchingConfig,
    LingBotV2MoEConfig,
    LingBotVLA2Config,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from phyai.parallel.state import resolve_mesh
from phyai.weights.shards import _Leg, fused, replicated, sharded

if TYPE_CHECKING:
    from phyai.layers.attention import ARAttnCtx


def _fp32_router_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Match the official LingBot FP32 router linear exactly."""
    with torch.amp.autocast(hidden_states.device.type, enabled=False):
        return F.linear(
            hidden_states.float(),
            weight.float(),
            None if bias is None else bias.float(),
        )


def _official_topk_router(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the released LingBot sigmoid top-k router exactly."""
    routing_scores = router_logits.sigmoid()
    scores_for_choice = routing_scores + correction_bias.unsqueeze(0)
    selected_experts = torch.topk(
        scores_for_choice,
        top_k,
        dim=-1,
    ).indices
    routing_weights = routing_scores.gather(1, selected_experts)
    routing_weights = routing_weights / (
        routing_weights.sum(dim=-1, keepdim=True) + 1e-20
    )
    if routed_scaling_factor != 1.0:
        routing_weights = routing_weights * routed_scaling_factor
    return routing_weights, selected_experts


def _flashinfer_fused_moe(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
) -> torch.Tensor:
    """Run only selected SwiGLU experts with FlashInfer CUTLASS MoE."""
    from flashinfer.fused_moe import cutlass_fused_moe

    # LingBot rounds routing weights to the model dtype before expert compute,
    # while FlashInfer requires token_final_scales to be stored as FP32.
    token_final_scales = routing_weights.float()
    token_selected_experts = selected_experts.to(torch.int32)
    outputs = cutlass_fused_moe(
        hidden_states,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        hidden_states.dtype,
        quant_scales=None,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )
    return outputs[0]


def get_vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Per-frame attention boundaries (``cu_seqlens``) for the packed patches.

    The vision tower flattens every image / video frame into one ragged sequence
    of patches and runs **block-diagonal** attention: each temporal frame (an
    ``h * w`` patch block) is its own window, so a patch only attends to patches
    of the *same* frame, never across frames. This returns the cumulative-offset
    ``indptr`` marking those frame boundaries -- exactly the ``cu_seqlens`` the
    attention op consumes (``self.attn(..., cu_seqlens_q=cu, cu_seqlens_kv=cu)``).

    Parameters
    ----------
    grid_thw:
        ``(num_images, 3)`` int tensor; row ``i`` is ``(t, h, w)`` for image /
        video ``i`` in **patch** units -- ``t`` temporal frames, each ``h * w``
        patches. A still image has ``t = 1``; a video has ``t > 1``.

    Returns
    -------
    cu_seqlens:
        ``(num_frames + 1,)`` int32, ``num_frames = sum(t_i)``. Standard indptr
        ``[0, len_0, len_0 + len_1, ...]``; frame ``f`` occupies the packed slice
        ``[cu[f], cu[f + 1])`` and the final entry is the total patch count.

    Example
    -------
    ``grid_thw = [[1, 2, 2], [2, 2, 3]]`` -- one image (1 frame, 2x2) plus one
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
    single-frame image collapses to ``[0, h*w]`` -- one window, a no-op block mask.
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

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer(
            "inv_freq",
            inv_freq.to(dtype=dtype, device=device),
            persistent=False,
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: (num_patches, 2) -> freqs (num_patches, 2 * (dim//2)).
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.to(position_ids.device)
        return freqs.flatten(1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half RoPE for the vision tower. q/k: ``(seq, heads, dim)``."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q_embed, k_embed = apply_rotary_pos_emb(
        q.float(),
        k.float(),
        cos.float(),
        sin.float(),
        unsqueeze_dim=-2,
    )
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class LingBotV2VisionLayerNorm(nn.Module):
    """BF16-stored Qwen vision LayerNorm with FP32 normalization math."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()
            self.bias.hf_keys = [(f"{prefix}.bias", None)]
            self.bias.weight_loader = replicated()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.float()
        mean = hidden_states_fp32.mean(dim=-1, keepdim=True)
        centered = hidden_states_fp32 - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + self.variance_epsilon)
        return (normalized * self.weight.float() + self.bias.float()).to(input_dtype)


class LingBotV2RMSNorm(nn.Module):
    """Official Qwen RMSNorm: FP32 variance, BF16 affine and output."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float,
        dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.square().mean(dim=-1, keepdim=True)
        normalized = hidden_states_fp32 * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)


def _attach_fused_qkv_loaders(
    layer: ColumnParallelLinear,
    checkpoint_prefix: str,
) -> None:
    """Load a checkpoint's single ``[Q; K; V]`` tensor into TP-local legs.

    Qwen3-VL vision attention stores Q/K/V in one fused tensor, whereas
    :class:`QKVParallelLinear` expects three checkpoint tensors. This loader
    preserves the fused checkpoint contract while placing one local slice from
    each leg into the column-parallel destination.
    """

    if not checkpoint_prefix:
        return
    global_sizes = tuple(layer.output_sizes_global)
    local_sizes = tuple(layer.output_partition_sizes)
    rank = layer.tp_rank

    def load_fused_qkv(
        parameter: nn.Parameter,
        loaded: torch.Tensor,
        _shard_id=None,
    ) -> None:
        if loaded.shape[0] != sum(global_sizes):
            raise ValueError(
                f"{checkpoint_prefix} leading dim {loaded.shape[0]} does not "
                f"match fused QKV width {sum(global_sizes)}."
            )
        source_offset = 0
        destination_offset = 0
        for global_size, local_size in zip(global_sizes, local_sizes):
            source = loaded.narrow(
                0,
                source_offset + rank * local_size,
                local_size,
            )
            parameter.data.narrow(0, destination_offset, local_size).copy_(source)
            source_offset += global_size
            destination_offset += local_size

    layer.weight.hf_keys = [(f"{checkpoint_prefix}.weight", None)]
    layer.weight.weight_loader = load_fused_qkv
    if layer.bias is not None:
        layer.bias.hf_keys = [(f"{checkpoint_prefix}.bias", None)]
        layer.bias.weight_loader = load_fused_qkv


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Full-patch Conv3D or its mathematically equivalent GEMM."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        backend: str = "gemm",
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if backend not in {"conv3d", "gemm"}:
            raise ValueError(
                "vision patch-embed backend must be 'conv3d' or 'gemm', "
                f"got {backend!r}."
            )
        self.backend = backend
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.patch_vector_dim = self.in_channels * math.prod(kernel)
        projection_prefix = f"{prefix}.proj" if prefix else ""
        if backend == "conv3d":
            self.proj = Conv3d(
                self.in_channels,
                self.embed_dim,
                kernel_size=kernel,
                stride=kernel,
                bias=True,
                dtype=params_dtype,
                device=device,
                prefix=projection_prefix,
            )
        else:
            self.proj = ReplicatedLinear(
                self.patch_vector_dim,
                self.embed_dim,
                bias=True,
                params_dtype=params_dtype,
                spec=Bf16Spec(),
                device=device,
                prefix=projection_prefix,
            )

            def load_conv3d_weight_as_gemm(
                parameter: nn.Parameter,
                loaded: torch.Tensor,
                _shard_id=None,
            ) -> None:
                if loaded.numel() != parameter.numel():
                    raise ValueError(
                        "PatchEmbed checkpoint weight has "
                        f"{loaded.numel()} values, expected {parameter.numel()}."
                    )
                parameter.data.copy_(loaded.reshape(parameter.shape))

            self.proj.weight.weight_loader = load_conv3d_weight_as_gemm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        if self.backend == "gemm":
            hidden_states = hidden_states.reshape(-1, self.patch_vector_dim)
            hidden_states, _ = self.proj(hidden_states.to(dtype=target_dtype))
            return hidden_states

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
        if params_dtype is None:
            params_dtype = get_engine_config().device.params_dtype
        device = get_engine_config().device.target
        self.norm = LingBotV2VisionLayerNorm(
            norm_dim,
            eps=1e-6,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm" if prefix else "",
        )
        del norm_backend
        self.linear_fc1 = ColumnParallelLinear(
            self.merged_dim,
            self.merged_dim,
            gather_output=False,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear_fc1" if prefix else "",
        )
        self.linear_fc2 = RowParallelLinear(
            self.merged_dim,
            config.out_hidden_size,
            input_is_parallel=True,
            reduce_results=True,
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
        self.linear_fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            gather_output=False,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.linear_fc1" if prefix else "",
        )
        self.linear_fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            input_is_parallel=True,
            reduce_results=True,
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
        self.qkv = ColumnParallelLinear(
            self.dim,
            self.dim * 3,
            output_sizes=[self.dim, self.dim, self.dim],
            gather_output=False,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.qkv" if prefix else "",
        )
        _attach_fused_qkv_loaders(
            self.qkv,
            f"{prefix}.qkv" if prefix else "",
        )
        self.num_heads_local = self.qkv.output_partition_sizes[0] // self.head_dim
        self.proj = RowParallelLinear(
            self.dim,
            self.dim,
            input_is_parallel=True,
            reduce_results=True,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.proj" if prefix else "",
        )
        self.attn = Attention(
            num_heads=self.num_heads_local,
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
            qkv.reshape(seq_length, 3, self.num_heads_local, self.head_dim)
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
        if params_dtype is None:
            params_dtype = get_engine_config().device.params_dtype
        device = get_engine_config().device.target
        self.norm1 = LingBotV2VisionLayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm1" if prefix else "",
        )
        self.norm2 = LingBotV2VisionLayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm2" if prefix else "",
        )
        del norm_backend
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
        patch_embed_backend: str = "gemm",
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
            backend=patch_embed_backend,
            device=device,
            prefix=f"{prefix}.patch_embed" if prefix else "",
        )
        # Learned position embedding table -- a plain replicated parameter,
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

        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            config.head_dim // 2,
            dtype=params_dtype,
            device=device,
        )
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
        bilinear_weights = bilinear_weights.to(self.pos_embed_weight.dtype)
        pos_embeds = (
            self.pos_embed_weight[bilinear_indices] * bilinear_weights[:, :, None]
        ).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self.deepstack_merger_list[index](hidden_states)
                )

        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states, deepstack_feature_lists


class Qwen3VLTextBlock(TransformerBlock):
    """Qwen text block with the official FP32 joint-attention boundary."""

    def __init__(self, *args, **kwargs) -> None:
        params_dtype = kwargs.get("params_dtype")
        prefix = kwargs.get("prefix", "")
        norm_eps = kwargs.get("norm_eps", 1e-6)
        super().__init__(*args, **kwargs)
        if params_dtype is None:
            params_dtype = get_engine_config().device.params_dtype
        device = get_engine_config().device.target
        self.input_norm = LingBotV2RMSNorm(
            self.hidden_size,
            eps=norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.input_layernorm",
        )
        self.pre_ff_norm = LingBotV2RMSNorm(
            self.hidden_size,
            eps=norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.post_attention_layernorm",
        )
        if self.attn_qk_norm_enabled:
            self.q_norm = LingBotV2RMSNorm(
                self.head_dim,
                eps=norm_eps,
                dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.self_attn.q_norm",
            )
            self.k_norm = LingBotV2RMSNorm(
                self.head_dim,
                eps=norm_eps,
                dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.self_attn.k_norm",
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attn_ctx=None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        hidden_states = self.input_norm(x)
        fused, _ = self.qkv_proj(hidden_states)
        q_dim = self.q_heads_local * self.head_dim
        kv_dim = self.kv_heads_local * self.head_dim
        q, k, v = fused.split([q_dim, kv_dim, kv_dim], dim=-1)
        leading = x.shape[:-1]
        q = self.q_norm(q.reshape(*leading, self.q_heads_local, self.head_dim)).float()
        k = self.k_norm(k.reshape(*leading, self.kv_heads_local, self.head_dim)).float()
        v = v.reshape(*leading, self.kv_heads_local, self.head_dim).float()
        q, k = self._apply_rope(q, k, positions, cos, sin)
        attn_out = self._attn_forward(
            q,
            k,
            v,
            attn_ctx,
            cu_seqlens_q,
            cu_seqlens_kv,
        )
        attn_flat = attn_out.reshape(
            *attn_out.shape[:-2], self.q_heads_local * self.head_dim
        ).to(self.o_proj.weight.dtype)
        out, _ = self.o_proj(attn_flat)
        hidden_states = residual + self.post_attn_norm(out)

        residual = hidden_states
        hidden_states = self.pre_ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_ff_norm(hidden_states)
        return residual + hidden_states


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
        attn_causal: bool = True,
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
        self.rotary_emb.to(dtype=params_dtype)
        layer_prefix = f"{prefix}.layers" if prefix else ""
        self.layers = nn.ModuleList(
            [
                Qwen3VLTextBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    attn_kind=text_attn_kind,
                    layer_idx=i,
                    attn_causal=attn_causal,
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
        self.norm = LingBotV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
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
    ) -> torch.Tensor:
        """The caller computes ``(cos, sin)`` once via
        :meth:`RotaryEmbedding.get_cos_sin` and threads them through every layer.
        """
        if inputs_embeds.dim() == 2 and cos.dim() > 2:
            cos = cos.reshape(-1, cos.shape[-1])
            sin = sin.reshape(-1, sin.shape[-1])

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
        return self.norm(hidden_states)


class LingBotV2AdaRMSNorm(nn.Module):
    """Checkpoint-compatible ``RMSNorm + FiLM`` used by the action stream."""

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        *,
        eps: float,
        params_dtype: torch.dtype,
        norm_backend: str,
        prefix: str,
    ) -> None:
        super().__init__()
        device = get_engine_config().device.target
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        self.weight.hf_keys = [(f"{prefix}.weight", None)]
        self.weight.weight_loader = replicated()
        del norm_backend
        self.gamma = ReplicatedLinear(
            cond_dim,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.gamma",
        )
        self.beta = ReplicatedLinear(
            cond_dim,
            hidden_size,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.beta",
        )

        # The official implementation starts as an ordinary RMSNorm.
        with torch.no_grad():
            self.gamma.weight.zero_()
            if self.gamma.bias is not None:
                self.gamma.bias.zero_()
            self.beta.weight.zero_()
            if self.beta.bias is not None:
                self.beta.bias.zero_()

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normed = hidden_states.float()
        variance = normed.square().mean(dim=-1, keepdim=True)
        normed = normed * torch.rsqrt(variance + self.variance_epsilon)
        normed = self.weight.float() * normed
        gamma, _ = self.gamma(cond.to(self.gamma.weight.dtype))
        beta, _ = self.beta(cond.to(self.beta.weight.dtype))
        while gamma.dim() < normed.dim():
            gamma = gamma.unsqueeze(-2)
            beta = beta.unsqueeze(-2)
        return ((1.0 + gamma.float()) * normed + beta.float()).to(input_dtype)


class LingBotV2ExpertMLP(DenseMLP):
    """One TP-aware SwiGLU expert using PHYAI/FlashInfer operators."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation="silu",
            gated=True,
            bias=False,
            params_dtype=params_dtype,
            prefix=prefix,
        )


class LingBotV2TokenMoE(nn.Module):
    """LingBot MoE with FP32 routing math and BF16-stored parameters."""

    def __init__(
        self,
        hidden_size: int,
        config: LingBotV2MoEConfig,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        if params_dtype != torch.bfloat16:
            raise ValueError(
                "LingBotV2TokenMoE requires bfloat16 expert parameters because "
                "FlashInfer CUTLASS fused MoE consumes bfloat16 expert weights."
            )
        if config.router_activation != "sigmoid" or not config.normalize_topk_prob:
            raise NotImplementedError(
                "The FlashInfer LingBot router requires sigmoid routing with "
                "normalized top-k probabilities."
            )
        if config.shared_expert_gate:
            raise NotImplementedError(
                "The released LingBot V2 has no shared-expert gate; enabling it "
                "requires a fused sigmoid-multiply kernel."
            )
        if config.router_bias:
            raise NotImplementedError("The released LingBot V2 router is bias-free.")

        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = ReplicatedLinear(
            hidden_size,
            config.num_experts,
            bias=config.router_bias,
            # Official BF16 deployment stores the gate in BF16, then promotes
            # both operands for the routing GEMM.
            params_dtype=params_dtype,
            spec=Bf16Spec(),
            device=device,
            prefix=f"{prefix}.gate",
        )
        mesh = resolve_mesh("model")
        self.tp_size = mesh.axis_size("tp")
        self.tp_rank = mesh.axis_local_rank("tp")
        if config.moe_intermediate_size % self.tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size={config.moe_intermediate_size} must be "
                f"divisible by tp_size={self.tp_size}."
            )
        self.intermediate_size_per_partition = (
            config.moe_intermediate_size // self.tp_size
        )

        # FlashInfer SwiGLU expects FC1 in [up, gate] order. Fuse and TP-shard
        # the public checkpoint's separate [E, I, H] tensors at load time.
        self.expert_gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                2 * self.intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        local_intermediate = self.intermediate_size_per_partition
        self.expert_gate_up_proj.hf_keys = [
            (f"{prefix}.experts.up_proj", "up"),
            (f"{prefix}.experts.gate_proj", "gate"),
        ]
        self.expert_gate_up_proj.weight_loader = fused(
            fuse_dim=1,
            legs={
                "up": _Leg(
                    offset=0,
                    size=local_intermediate,
                    dim=1,
                    axis="tp",
                ),
                "gate": _Leg(
                    offset=local_intermediate,
                    size=local_intermediate,
                    dim=1,
                    axis="tp",
                ),
            },
            mesh=mesh,
        )
        self.expert_down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                hidden_size,
                self.intermediate_size_per_partition,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.expert_down_proj.hf_keys = [(f"{prefix}.experts.down_proj", None)]
        self.expert_down_proj.weight_loader = sharded(
            dim=2,
            axis="tp",
            mesh=mesh,
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.num_experts, dtype=params_dtype, device=device),
            requires_grad=False,
        )
        self.e_score_correction_bias.hf_keys = [
            (f"{prefix}.e_score_correction_bias", None)
        ]
        self.e_score_correction_bias.weight_loader = replicated()

        self.shared_expert = LingBotV2ExpertMLP(
            hidden_size,
            config.shared_expert_intermediate_size,
            params_dtype=params_dtype,
            prefix=f"{prefix}.shared_expert",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, original_shape[-1])
        router_logits = _fp32_router_linear(
            hidden_flat,
            self.gate.weight,
            self.gate.bias,
        )
        routing_weights, selected_experts = _official_topk_router(
            router_logits,
            self.e_score_correction_bias.float(),
            top_k=self.top_k,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        routing_weights = routing_weights.to(hidden_flat.dtype)
        routed = _flashinfer_fused_moe(
            hidden_flat,
            selected_experts,
            routing_weights,
            self.expert_gate_up_proj,
            self.expert_down_proj,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
        )
        if self.tp_size > 1:
            routed = P.all_reduce(routed, axis="tp")

        shared = self.shared_expert(hidden_flat)
        return (routed + shared).reshape(original_shape)


class LingBotV2ExpertLayer(nn.Module):
    """Qwen2 action layer backed by paged diffusion attention."""

    def __init__(
        self,
        config: LingBotV2ExpertConfig,
        moe_config: LingBotV2MoEConfig,
        layer_idx: int,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        attn_prefix = f"{prefix}.self_attn"
        self.input_layernorm = LingBotV2AdaRMSNorm(
            config.hidden_size,
            config.adarms_cond_dim,
            eps=config.rms_norm_eps,
            params_dtype=params_dtype,
            norm_backend=norm_backend,
            prefix=f"{prefix}.input_layernorm",
        )
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            bias=config.qkv_bias,
            params_dtype=params_dtype,
            prefix=f"{attn_prefix}.qkv_proj",
        )
        q_width, k_width, _ = self.qkv_proj.output_partition_sizes
        self.q_heads_local = q_width // config.head_dim
        self.kv_heads_local = k_width // config.head_dim
        self.o_proj = RowParallelLinear(
            in_features=config.joint_attention_dim,
            out_features=config.hidden_size,
            bias=config.output_bias,
            params_dtype=params_dtype,
            prefix=f"{attn_prefix}.o_proj",
        )
        self.attn = DiffusionAttention(
            num_heads=self.q_heads_local,
            head_dim=config.head_dim,
            layer_id=layer_idx,
            num_kv_heads=self.kv_heads_local,
            causal=False,
            backend=attn_backend,
        )
        self.post_attention_layernorm = LingBotV2AdaRMSNorm(
            config.hidden_size,
            config.adarms_cond_dim,
            eps=config.rms_norm_eps,
            params_dtype=params_dtype,
            norm_backend=norm_backend,
            prefix=f"{prefix}.post_attention_layernorm",
        )
        if layer_idx in moe_config.layer_indices:
            self.mlp: nn.Module = LingBotV2TokenMoE(
                config.hidden_size,
                moe_config,
                params_dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = LingBotV2ExpertMLP(
                config.hidden_size,
                config.intermediate_size,
                params_dtype=params_dtype,
                prefix=f"{prefix}.mlp",
            )

    def _split_qkv(
        self, fused: torch.Tensor, leading: torch.Size
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_dim = self.q_heads_local * self.head_dim
        kv_dim = self.kv_heads_local * self.head_dim
        q, k, v = fused.split([q_dim, kv_dim, kv_dim], dim=-1)
        return (
            q.reshape(*leading, self.q_heads_local, self.head_dim),
            k.reshape(*leading, self.kv_heads_local, self.head_dim),
            v.reshape(*leading, self.kv_heads_local, self.head_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cond: torch.Tensor,
        rope: InterleavedMRotaryEmbedding,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_ctx: DiffusionAttnCtx,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.input_layernorm(hidden_states, cond)
        fused, _ = self.qkv_proj(normed)
        q, k, v = self._split_qkv(fused, hidden_states.shape[:-1])
        # Official joint attention promotes every stream's Q/K/V before MRoPE.
        q, k, v = q.float(), k.float(), v.float()
        q, k = rope.apply(q, k, cos, sin)
        # AdaRMSNorm keeps the action stream as (B, S, H), matching the
        # official condition projection and token broadcast. Attention owns
        # the ragged-token boundary and therefore receives flattened Q/K/V.
        q = q.reshape(-1, self.q_heads_local, self.head_dim)
        k = k.reshape(-1, self.kv_heads_local, self.head_dim)
        v = v.reshape(-1, self.kv_heads_local, self.head_dim)
        attn_out = self.attn(q, k, v, attn_ctx).reshape(
            *hidden_states.shape[:-1],
            -1,
        )
        attn_out = attn_out.to(self.o_proj.weight.dtype)
        attn_out, _ = self.o_proj(attn_out)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, cond)
        return residual + self.mlp(hidden_states)


class LingBotV2ExpertStack(nn.Module):
    """All action layers plus the released plain final RMSNorm."""

    DEFAULT_PREFIX = "model.qwenvl_with_expert.qwen_expert.model"

    def __init__(
        self,
        config: LingBotV2ExpertConfig,
        moe_config: LingBotV2MoEConfig,
        *,
        params_dtype: torch.dtype,
        attn_backend: str,
        norm_backend: str,
        device: torch.device | str,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                LingBotV2ExpertLayer(
                    config,
                    moe_config,
                    layer_idx,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    device=device,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LingBotV2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm",
        )


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    *,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    """Flow-time sinusoidal embedding used both as token input and AdaNorm cond."""

    if time.dim() != 1 or dimension % 2:
        raise ValueError("time must be 1-D and dimension must be even.")
    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=time.device
    )
    period = min_period * (max_period / min_period) ** fraction
    phase = time.float().unsqueeze(1) * (2.0 * math.pi / period).unsqueeze(0)
    return torch.cat([phase.sin(), phase.cos()], dim=-1)


class LingBotV2ActionTimeHeads(nn.Module):
    """State/action projections and flow-time fusion heads."""

    def __init__(
        self,
        expert_config: LingBotV2ExpertConfig,
        flow_config: LingBotV2FlowMatchingConfig,
        *,
        params_dtype: torch.dtype,
        prefix: str = "model",
    ) -> None:
        super().__init__()
        self.expert_hidden = expert_config.hidden_size
        self.flow_config = flow_config
        # These boundary projections stay replicated intentionally: their
        # state/action side is only 55-wide and the expert stack consumes and
        # returns a full residual tensor on every TP rank.
        replicated_specs = (
            ("state_proj", flow_config.max_state_dim, expert_config.hidden_size),
            ("action_in_proj", flow_config.max_action_dim, expert_config.hidden_size),
            ("action_out_proj", expert_config.hidden_size, flow_config.max_action_dim),
        )
        for name, in_features, out_features in replicated_specs:
            setattr(
                self,
                name,
                ReplicatedLinear(
                    in_features,
                    out_features,
                    bias=True,
                    params_dtype=params_dtype,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.action_time_mlp_in = ColumnParallelLinear(
            expert_config.hidden_size * 2,
            expert_config.hidden_size,
            gather_output=False,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.action_time_mlp_in",
        )
        self.action_time_mlp_out = RowParallelLinear(
            expert_config.hidden_size,
            expert_config.hidden_size,
            input_is_parallel=True,
            reduce_results=True,
            bias=True,
            params_dtype=params_dtype,
            prefix=f"{prefix}.action_time_mlp_out",
        )

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        out, _ = self.state_proj(state.to(self.state_proj.weight.dtype))
        return out

    def embed_action_time(
        self, actions: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_emb, _ = self.action_in_proj(
            actions.to(self.action_in_proj.weight.dtype)
        )
        cond = create_sinusoidal_pos_embedding(
            time,
            self.expert_hidden,
            min_period=self.flow_config.time_embedding_min_period,
            max_period=self.flow_config.time_embedding_max_period,
        ).to(action_emb.dtype)
        cond_tokens = cond.unsqueeze(1).expand(-1, action_emb.shape[1], -1)
        fused = torch.cat([action_emb, cond_tokens], dim=-1)
        fused, _ = self.action_time_mlp_in(fused)
        fused, _ = self.action_time_mlp_out(F.silu(fused))
        return fused, cond

    def project_action(self, hidden_states: torch.Tensor) -> torch.Tensor:
        out, _ = self.action_out_proj(
            hidden_states.to(self.action_out_proj.weight.dtype)
        )
        return out


class LingBotV2DualQuery(nn.Module):
    """Build the eight current and eight future learned prefix queries."""

    def __init__(
        self,
        config,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str,
        prefix: str = "model",
    ) -> None:
        super().__init__()
        released_layout = {
            "use_future_depth": True,
            "use_future_video": True,
            "use_future_video_patch": True,
            "use_current_video_patch": True,
            "share_future_depth_query": True,
            "use_shared_future_task_proj": True,
            "use_current_shared_task_proj": True,
            "use_future_video_cls": False,
        }
        unsupported = [
            f"{name}={getattr(config, name)!r} (expected {expected!r})"
            for name, expected in released_layout.items()
            if getattr(config, name) != expected
        ]
        if unsupported:
            raise ValueError(
                "LingBotV2DualQuery supports only the released four-seed-table "
                "shared-projection layout; unsupported settings: "
                + ", ".join(unsupported)
            )
        self.config = config
        names = (
            "depth_align_embs",
            "current_video_align_embs",
            "future_depth_align_embs",
            "future_video_align_embs",
        )
        for name in names:
            parameter = nn.Parameter(
                torch.empty(
                    config.num_query_seeds,
                    config.query_hidden_size,
                    dtype=params_dtype,
                    device=device,
                ),
                requires_grad=False,
            )
            parameter.hf_keys = [(f"{prefix}.{name}", None)]
            parameter.weight_loader = replicated()
            setattr(self, name, parameter)
        # Each rank starts from the same pooled query tensor. Row parallelism
        # shards the wide 5120 input projection and reduces the full 2560-wide
        # prefix token back onto every rank for the text stack.
        self.current_shared_task_proj = RowParallelLinear(
            config.query_hidden_size * 2,
            config.query_hidden_size,
            input_is_parallel=False,
            reduce_results=True,
            bias=config.fusion_bias,
            params_dtype=params_dtype,
            prefix=f"{prefix}.current_shared_task_proj",
        )
        self.future_shared_task_proj = RowParallelLinear(
            config.query_hidden_size * 2,
            config.query_hidden_size,
            input_is_parallel=False,
            reduce_results=True,
            bias=config.fusion_bias,
            params_dtype=params_dtype,
            prefix=f"{prefix}.future_shared_task_proj",
        )

    def _pool(self, seeds: torch.Tensor) -> torch.Tensor:
        return seeds.view(
            self.config.num_task_tokens,
            self.config.seeds_per_task_token,
            self.config.query_hidden_size,
        ).mean(dim=1)

    def forward(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        current_depth = self._pool(self.depth_align_embs)
        current_video = self._pool(self.current_video_align_embs)
        current, _ = self.current_shared_task_proj(
            torch.cat([current_depth, current_video], dim=-1)
        )
        future_depth = self._pool(self.future_depth_align_embs)
        future_video = self._pool(self.future_video_align_embs)
        future, _ = self.future_shared_task_proj(
            torch.cat([future_depth, future_video], dim=-1)
        )
        return (
            current.unsqueeze(0).expand(batch_size, -1, -1),
            future.unsqueeze(0).expand(batch_size, -1, -1),
        )


def _vision_mrope_positions(
    start: int, grid_thw: torch.Tensor, merge_size: int
) -> torch.Tensor:
    t, h, w = (int(value) for value in grid_thw.tolist())
    h //= merge_size
    w //= merge_size
    temporal = torch.arange(t, device=grid_thw.device).repeat_interleave(h * w) + start
    height = (
        torch.arange(h, device=grid_thw.device).repeat_interleave(w).repeat(t) + start
    )
    width = torch.arange(w, device=grid_thw.device).repeat(h * t) + start
    return torch.stack([temporal, height, width], dim=0)


def build_lingbot_v2_mrope_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Build Qwen3-VL ``(3, B, S)`` positions for packed LingBot prefixes."""

    position_ids = torch.zeros(
        3, *input_ids.shape, dtype=torch.int64, device=input_ids.device
    )
    grid_index = 0
    for batch_index in range(input_ids.shape[0]):
        valid = attention_mask[batch_index].bool()
        ids = input_ids[batch_index, valid]
        token_types = (ids == image_token_id).to(torch.int64)
        current_position = 0
        parts: list[torch.Tensor] = []
        for modality, group in itertools.groupby(
            enumerate(token_types.tolist()), lambda item: item[1]
        ):
            group_list = list(group)
            length = group_list[-1][0] - group_list[0][0] + 1
            if modality == 0:
                part = torch.arange(length, device=input_ids.device).expand(3, -1)
                parts.append(part + current_position)
                current_position += length
            else:
                grid = image_grid_thw[grid_index]
                grid_index += 1
                part = _vision_mrope_positions(
                    current_position, grid, spatial_merge_size
                )
                if part.shape[1] != length:
                    raise ValueError(
                        f"image placeholder length {length} does not match merged "
                        f"grid token count {part.shape[1]}."
                    )
                parts.append(part)
                current_position += max(
                    int(grid[0]),
                    int(grid[1]) // spatial_merge_size,
                    int(grid[2]) // spatial_merge_size,
                )
        if parts:
            position_ids[:, batch_index, valid] = torch.cat(parts, dim=1)
    if grid_index != int(image_grid_thw.shape[0]):
        raise ValueError(
            f"consumed {grid_index} image grids but received {image_grid_thw.shape[0]}."
        )
    return position_ids


class LingBotV2Model(nn.Module):
    """Flat LingBot-VLA 2.0 parameter container, following PI0's split."""

    VISION_PREFIX = "model.qwenvl_with_expert.qwenvl.model.visual"
    TEXT_PREFIX = "model.qwenvl_with_expert.qwenvl.model.language_model"

    def __init__(
        self,
        config: LingBotVLA2Config,
        *,
        params_dtype: torch.dtype | None = None,
        vision_params_dtype: torch.dtype | None = None,
        vision_patch_embed_backend: str = "gemm",
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
        vision_dtype = vision_params_dtype or params_dtype
        if attn_backend != "flashinfer":
            raise NotImplementedError(
                "LingBot V2 paged prefix/expert attention currently requires "
                "attn_backend='flashinfer'."
            )
        self.config = config
        self.params_dtype = params_dtype
        self.vision_params_dtype = vision_dtype
        self.vision = Qwen3VLVisionModel(
            config.vision,
            params_dtype=vision_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            patch_embed_backend=vision_patch_embed_backend,
            device=device,
            prefix=self.VISION_PREFIX,
        )
        self.text = Qwen3VLTextModel(
            config.text,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            device=device,
            prefix=self.TEXT_PREFIX,
            text_attn_kind="ar",
            attn_causal=config.vlm_causal,
        )
        self.expert_stack = LingBotV2ExpertStack(
            config.expert,
            config.moe,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
            device=device,
        )
        self.dual_query = LingBotV2DualQuery(
            config.dual_query,
            params_dtype=params_dtype,
            device=device,
        )
        self.heads = LingBotV2ActionTimeHeads(
            config.expert,
            config.flow_matching,
            params_dtype=params_dtype,
        )

    @property
    def mrope(self) -> InterleavedMRotaryEmbedding:
        return self.text.rotary_emb

    def embed_language(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text.embed_tokens(input_ids)

    def embed_special(self, token_id: int) -> torch.Tensor:
        token = torch.tensor(
            [token_id], dtype=torch.int64, device=self.text.embed_tokens.weight.device
        )
        return self.embed_language(token)[0]


_DROP_WEIGHT_FRAGMENTS = (
    ".lm_head.",
    ".qwen_expert.model.embed_tokens.",
    "depth_align_head",
    "future_depth_align_head",
    "video_align_head",
    "future_video_cls_align_emb",
    "future_video_cls_head",
    "expert_visual",
)


def lingbot_v2_weight_remap(name: str) -> str | None:
    """Drop training-only tensors while keeping public V2 checkpoint keys intact."""

    while name.startswith("module."):
        name = name[len("module.") :]
    if any(fragment in name for fragment in _DROP_WEIGHT_FRAGMENTS):
        return None
    return name


__all__ = [
    "LingBotV2ActionTimeHeads",
    "LingBotV2AdaRMSNorm",
    "LingBotV2DualQuery",
    "LingBotV2ExpertLayer",
    "LingBotV2ExpertMLP",
    "LingBotV2ExpertStack",
    "LingBotV2Model",
    "LingBotV2TokenMoE",
    "Qwen3VLTextModel",
    "Qwen3VLVisionAttention",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionMLP",
    "Qwen3VLVisionModel",
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionPatchMerger",
    "Qwen3VLVisionRotaryEmbedding",
    "apply_rotary_pos_emb_vision",
    "build_lingbot_v2_mrope_position_ids",
    "create_sinusoidal_pos_embedding",
    "get_vision_bilinear_indices_and_weights",
    "get_vision_cu_seqlens",
    "get_vision_position_ids",
    "lingbot_v2_weight_remap",
]
