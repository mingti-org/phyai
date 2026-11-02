"""LingBot's Qwen3-VL vision tower and prepared positional inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from phyai.engine_config import get_engine_config, resolve_params_dtype
from phyai.layers.conv import Conv3d
from phyai.layers.quant import Bf16Spec
from phyai.layers.linear import (
    ReplicatedLinear,
    ColumnParallelLinear,
    RowParallelLinear,
)
from phyai.layers.attention.nocache.base import AttnCtx
from phyai.layers.attention.nocache.layer import Attention
from phyai.layers.backbones.qwen3_vl.modeling import (
    Qwen3VLVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)
from phyai.models.lingbot_v2.configuration_lingbotv2 import Qwen3VLVisionConfig
from phyai.models.lingbot_v2.norm import LingBotV2LayerNorm
from phyai.weights.shards import replicated


@dataclass(frozen=True)
class VisionPositionInputs:
    """Position tensors reused by every block for one packed image layout."""

    pos_embeds: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    cu_seqlens: torch.Tensor


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Full-patch Conv3D or its mathematically equivalent GEMM."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        backend: str = "conv3d",
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        params_dtype = resolve_params_dtype(params_dtype)
        if device is None:
            device = get_engine_config().device.target
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

            def _load_conv3d_weight_as_gemm(
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

            self.proj.weight.weight_loader = _load_conv3d_weight_as_gemm

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the selected Conv3D or flattened GEMM patch projection."""

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
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.merged_dim = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.merged_dim if use_postshuffle_norm else config.hidden_size
        params_dtype = resolve_params_dtype(params_dtype)
        if device is None:
            device = get_engine_config().device.target
        self.norm = LingBotV2LayerNorm(
            norm_dim,
            eps=1e-6,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm" if prefix else "",
        )
        self.linear_fc1 = ColumnParallelLinear(
            self.merged_dim,
            self.merged_dim,
            gather_output=False,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_fc1" if prefix else "",
        )
        self.linear_fc2 = RowParallelLinear(
            self.merged_dim,
            config.out_hidden_size,
            input_is_parallel=True,
            reduce_results=True,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_fc2" if prefix else "",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and merge each spatial patch group into one token."""

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
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        params_dtype = resolve_params_dtype(params_dtype)
        self.linear_fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            gather_output=False,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_fc1" if prefix else "",
        )
        self.linear_fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            input_is_parallel=True,
            reduce_results=True,
            bias=True,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.linear_fc2" if prefix else "",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two-layer vision feed-forward network."""

        x, _ = self.linear_fc1(x)
        x = F.gelu(x, approximate="tanh")
        x, _ = self.linear_fc2(x)
        return x


class Qwen3VLVisionAttention(nn.Module):
    """Non-causal self-attention for ragged Qwen3-VL image grids."""

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str = "flashinfer",
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        params_dtype = resolve_params_dtype(params_dtype)
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
            device=device,
            prefix=f"{prefix}.qkv" if prefix else "",
        )
        self.num_heads_local = self.qkv.output_partition_sizes[0] // self.head_dim
        self.proj = RowParallelLinear(
            self.dim,
            self.dim,
            input_is_parallel=True,
            reduce_results=True,
            bias=True,
            params_dtype=params_dtype,
            device=device,
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_ctx: AttnCtx,
    ) -> torch.Tensor:
        """Apply rotary self-attention independently to each image sequence."""

        seq_length = hidden_states.shape[0]
        qkv, _ = self.qkv(hidden_states)
        q, k, v = (
            qkv.reshape(seq_length, 3, self.num_heads_local, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )  # each (seq, num_heads, head_dim)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        out = self.attn(q, k, v, ctx=attn_ctx)
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
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        params_dtype = resolve_params_dtype(params_dtype)
        if device is None:
            device = get_engine_config().device.target
        self.norm1 = LingBotV2LayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm1" if prefix else "",
        )
        self.norm2 = LingBotV2LayerNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm2" if prefix else "",
        )
        self.attn = Qwen3VLVisionAttention(
            config,
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            device=device,
            prefix=f"{prefix}.attn" if prefix else "",
        )
        self.mlp = Qwen3VLVisionMLP(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.mlp" if prefix else "",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_ctx: AttnCtx,
    ) -> torch.Tensor:
        """Run one residual vision transformer block."""

        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            attn_ctx=attn_ctx,
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
        patch_embed_backend: str = "conv3d",
        device: torch.device | str | None = None,
        prefix: str = "visual",
    ) -> None:
        super().__init__()
        params_dtype = resolve_params_dtype(params_dtype)
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
            device=device,
        ).to(dtype=params_dtype)
        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(
                    config,
                    params_dtype=params_dtype,
                    attn_backend=attn_backend,
                    device=device,
                    prefix=f"{prefix}.blocks.{i}" if prefix else "",
                )
                for i in range(config.depth)
            ]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            config,
            use_postshuffle_norm=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.merger" if prefix else "",
        )
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    config,
                    use_postshuffle_norm=True,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.deepstack_merger_list.{j}" if prefix else "",
                )
                for j in range(len(self.deepstack_visual_indexes))
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype used by the learned vision parameters."""

        return self.pos_embed_weight.dtype

    def prepare_positions(self, grid_thw: torch.Tensor) -> VisionPositionInputs:
        """Prepare positions once per image layout, after loading weights."""
        if grid_thw.device.type != "cpu":
            raise ValueError("vision position preparation requires a CPU image grid")
        device = self.pos_embed_weight.device
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )
        bilinear_indices = bilinear_indices.to(device=device)
        bilinear_weights = bilinear_weights.to(device=device, dtype=self.dtype)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size).to(
            device=device
        )
        cu_seqlens = get_vision_cu_seqlens(grid_thw).to(device=device)
        corners = self.pos_embed_weight[bilinear_indices] * bilinear_weights[:, :, None]
        pos_embeds = corners[0] + corners[1]
        pos_embeds = pos_embeds + corners[2]
        pos_embeds = pos_embeds + corners[3]
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        return VisionPositionInputs(pos_embeds, (emb.cos(), emb.sin()), cu_seqlens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: VisionPositionInputs,
        *,
        attn_ctx: AttnCtx,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode packed patches using positions and attention planned by the runner."""
        hidden_states = self.patch_embed(hidden_states) + positions.pos_embeds

        deepstack_feature_lists: list[torch.Tensor] = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                position_embeddings=positions.position_embeddings,
                attn_ctx=attn_ctx,
            )
            if layer_num in self.deepstack_visual_indexes:
                index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature_lists.append(
                    self.deepstack_merger_list[index](hidden_states)
                )

        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states, deepstack_feature_lists
