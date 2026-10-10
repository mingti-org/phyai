"""DreamZero Wan2.1 CLIP image encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from phyai.engine_config import ParallelConfig
from phyai.layers.linear import ReplicatedLinear
from phyai.layers.mlp import DenseMLP
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroImageEncoderConfig
from phyai.parallel.layout import build_rank_layout
from phyai.parallel.mesh import Mesh
from phyai.weights.shards import replicated


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def attach_replicated_image_encoder_weights(module: nn.Module) -> None:
    """Attach phyai weight-loader metadata to all image encoder parameters."""
    for name, param in module.named_parameters():
        if not hasattr(param, "weight_loader"):
            param.hf_keys = [(name, None)]
            param.weight_loader = replicated()
    # Keep the official split fc1/fc2/fc3 keys and the fused gate/up loaders.
    for name, child in module.named_modules():
        if isinstance(child, DenseMLP) and child.gated:
            prefix = f"{name}." if name else ""
            for suffix in ("weight", "bias"):
                gate_up = getattr(child.gate_up_proj, suffix)
                down = getattr(child.down_proj, suffix)
                if gate_up is not None:
                    gate_up.hf_keys = [
                        (f"{prefix}fc1.{suffix}", 0),
                        (f"{prefix}fc2.{suffix}", 1),
                    ]
                if down is not None:
                    down.hf_keys = [(f"{prefix}fc3.{suffix}", None)]


def dreamzero_image_encoder_weight_remap(name: str) -> str | None:
    """Map DreamZero checkpoint keys to PHYAI image-encoder parameter keys."""
    if name.startswith("action_head.image_encoder."):
        return name.removeprefix("action_head.image_encoder.")
    if name == "log_scale" or name.startswith("visual."):
        return f"model.{name}"
    if name.startswith("model.log_scale") or name.startswith("model.visual."):
        return name
    return None


def pos_interpolate(pos: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Interpolate CLIP position embeddings for non-native image sizes."""
    if pos.size(1) == seq_len:
        return pos
    src_grid = int(math.sqrt(pos.size(1)))
    tar_grid = int(math.sqrt(seq_len))
    num_extra_tokens = pos.size(1) - src_grid * src_grid
    return torch.cat(
        [
            pos[:, :num_extra_tokens],
            F.interpolate(
                pos[:, num_extra_tokens:]
                .float()
                .reshape(1, src_grid, src_grid, -1)
                .permute(0, 3, 1, 2),
                size=(tar_grid, tar_grid),
                mode="bicubic",
                align_corners=False,
            )
            .flatten(2)
            .transpose(1, 2)
            .type_as(pos),
        ],
        dim=1,
    )


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPLayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).type_as(x)


class CLIPMLP(nn.Sequential):
    """Keep checkpoint indices while unpacking the phyai linear outputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self[0](x)
        x = self[1](x)
        x, _ = self[2](x)
        return self[3](x)


class CLIPSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        self.to_qkv = ReplicatedLinear(
            dim, dim * 3, params_dtype=params_dtype, device="cpu"
        )
        self.proj = ReplicatedLinear(dim, dim, params_dtype=params_dtype, device="cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        x, _ = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class CLIPAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        *,
        post_norm: bool = False,
        activation: str = "gelu",
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        if activation not in {"quick_gelu", "gelu", "swi_glu"}:
            raise ValueError(f"unsupported activation={activation!r}.")
        super().__init__()
        self.post_norm = post_norm
        self.norm1 = CLIPLayerNorm(dim, eps=norm_eps, dtype=params_dtype)
        self.attn = CLIPSelfAttention(
            dim,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            params_dtype=params_dtype,
        )
        self.norm2 = CLIPLayerNorm(dim, eps=norm_eps, dtype=params_dtype)
        mid_dim = int(dim * mlp_ratio)
        if activation == "swi_glu":
            # Encoders can run only on rank 0; never join the DiT TP collectives.
            encoder_mesh = Mesh(
                build_rank_layout(ParallelConfig()), name="dreamzero_image_encoder"
            )
            self.mlp = DenseMLP(
                dim,
                mid_dim,
                activation="silu",
                gated=True,
                bias=True,
                sequence_parallel=False,
                params_dtype=params_dtype,
                mesh=encoder_mesh,
            )
        else:
            self.mlp = CLIPMLP(
                ReplicatedLinear(dim, mid_dim, params_dtype=params_dtype, device="cpu"),
                QuickGELU() if activation == "quick_gelu" else nn.GELU(),
                ReplicatedLinear(mid_dim, dim, params_dtype=params_dtype, device="cpu"),
                nn.Dropout(proj_dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class CLIPAttentionPool(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        *,
        activation: str = "gelu",
        proj_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = nn.Parameter(
            gain * torch.randn(1, 1, dim, dtype=params_dtype)
        )
        self.to_q = ReplicatedLinear(dim, dim, params_dtype=params_dtype, device="cpu")
        self.to_kv = ReplicatedLinear(
            dim, dim * 2, params_dtype=params_dtype, device="cpu"
        )
        self.proj = ReplicatedLinear(dim, dim, params_dtype=params_dtype, device="cpu")
        self.norm = CLIPLayerNorm(dim, eps=norm_eps, dtype=params_dtype)
        mid_dim = int(dim * mlp_ratio)
        self.mlp = CLIPMLP(
            ReplicatedLinear(dim, mid_dim, params_dtype=params_dtype, device="cpu"),
            QuickGELU() if activation == "quick_gelu" else nn.GELU(),
            ReplicatedLinear(mid_dim, dim, params_dtype=params_dtype, device="cpu"),
            nn.Dropout(proj_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, dim = x.shape
        q, _ = self.to_q(self.cls_embedding)
        q = q.expand(batch, -1, -1)
        kv, _ = self.to_kv(x)
        k, v = kv.chunk(2, dim=-1)
        q = q.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(batch, 1, dim)
        x, _ = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class WanCLIPVisionTransformer(nn.Module):
    """OpenCLIP ViT-H/14 visual tower used by DreamZero Wan2.1."""

    def __init__(
        self,
        config: DreamZeroImageEncoderConfig,
        *,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches
        self.dim = config.vision_dim
        self.num_heads = config.vision_heads
        self.num_layers = config.vision_layers
        self.pool_type = config.vision_pool
        self.post_norm_enabled = config.vision_post_norm
        gain = 1.0 / math.sqrt(config.vision_dim)
        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            config.vision_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=not config.vision_pre_norm,
            dtype=params_dtype,
        )
        if config.vision_pool in {"token", "token_fc"}:
            self.cls_embedding = nn.Parameter(
                gain * torch.randn(1, 1, config.vision_dim, dtype=params_dtype)
            )
            num_tokens = self.num_patches + 1
        else:
            num_tokens = self.num_patches
        self.pos_embedding = nn.Parameter(
            gain * torch.randn(1, num_tokens, config.vision_dim, dtype=params_dtype)
        )
        self.dropout = nn.Dropout(config.embedding_dropout)
        self.pre_norm = (
            CLIPLayerNorm(config.vision_dim, eps=config.norm_eps, dtype=params_dtype)
            if config.vision_pre_norm
            else None
        )
        self.transformer = nn.Sequential(
            *[
                CLIPAttentionBlock(
                    config.vision_dim,
                    config.vision_mlp_ratio,
                    config.vision_heads,
                    post_norm=config.vision_post_norm,
                    activation=config.activation,
                    attn_dropout=config.attn_dropout,
                    proj_dropout=config.proj_dropout,
                    norm_eps=config.norm_eps,
                    params_dtype=params_dtype,
                )
                for _ in range(config.vision_layers)
            ]
        )
        self.post_norm = CLIPLayerNorm(
            config.vision_dim, eps=config.norm_eps, dtype=params_dtype
        )
        if config.vision_pool == "token":
            self.head = nn.Parameter(
                gain
                * torch.randn(config.vision_dim, config.embed_dim, dtype=params_dtype)
            )
        elif config.vision_pool == "token_fc":
            self.head = ReplicatedLinear(
                config.vision_dim,
                config.embed_dim,
                params_dtype=params_dtype,
                device="cpu",
            )
        else:
            self.head = CLIPAttentionPool(
                config.vision_dim,
                config.vision_mlp_ratio,
                config.vision_heads,
                activation=config.activation,
                proj_dropout=config.proj_dropout,
                norm_eps=config.norm_eps,
                params_dtype=params_dtype,
            )

    def forward(
        self,
        x: torch.Tensor,
        interpolation: bool = False,
        use_31_block: bool = False,
        feature_layer: int | None = None,
    ) -> torch.Tensor:
        batch = x.size(0)
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        if self.pool_type in {"token", "token_fc"}:
            x = torch.cat(
                [
                    self.cls_embedding.expand(batch, -1, -1).to(
                        dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )
        pos_embedding = (
            pos_interpolate(self.pos_embedding, x.size(1))
            if interpolation
            else self.pos_embedding
        )
        pos_embedding = pos_embedding.to(dtype=x.dtype, device=x.device)
        x = self.dropout(x + pos_embedding)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        if feature_layer is not None and not 0 < feature_layer <= self.num_layers:
            raise ValueError(
                f"feature_layer={feature_layer} must be in [1, {self.num_layers}]."
            )
        if feature_layer is not None:
            x = self.transformer[:feature_layer](x)
        elif use_31_block:
            if self.config.feature_layer == self.num_layers - 1:
                x = self.transformer[:-1](x)
            else:
                x = self.transformer[: self.config.feature_layer](x)
        else:
            x = self.transformer(x)
        return x


class WanXLMRobertaCLIPVisual(nn.Module):
    """Visual-only wrapper matching the official CLIP module parameter names."""

    def __init__(
        self,
        config: DreamZeroImageEncoderConfig,
        *,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.vision_dim = config.vision_dim
        self.vision_layers = config.vision_layers
        self.visual = WanCLIPVisionTransformer(config, params_dtype=params_dtype)
        self.log_scale = nn.Parameter(
            math.log(1 / 0.07) * torch.ones([], dtype=params_dtype)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values, use_31_block=True)


class DreamZeroWanImageEncoder(nn.Module):
    """DreamZero image encoder with official WanImageEncoder-compatible layout."""

    def __init__(
        self,
        config: DreamZeroImageEncoderConfig | None = None,
        *,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config or DreamZeroImageEncoderConfig()
        self.model = WanXLMRobertaCLIPVisual(self.config, params_dtype=params_dtype)
        self.register_buffer(
            "image_mean",
            torch.tensor(CLIP_IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(CLIP_IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        attach_replicated_image_encoder_weights(self)

    def preprocess(self, videos: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Resize and normalize DreamZero image tensors from [-1, 1] to CLIP input."""
        size = (self.config.image_size, self.config.image_size)
        if isinstance(videos, torch.Tensor):
            if videos.dim() == 4:
                frames = videos
            elif videos.dim() == 5:
                frames = torch.cat([u for u in videos], dim=0)
            else:
                raise ValueError(
                    f"videos tensor must be 4-D or 5-D; got shape {tuple(videos.shape)}."
                )
        else:
            frames = torch.cat(list(videos), dim=0)
        if frames.dim() != 4 or frames.size(1) != self.config.num_channels:
            raise ValueError(
                f"videos must flatten to (N, {self.config.num_channels}, H, W); "
                f"got shape {tuple(frames.shape)}."
            )
        frames = F.interpolate(
            frames,
            size=size,
            mode="bicubic",
            align_corners=False,
        )
        frames = frames.mul(0.5).add(0.5)
        mean = torch.tensor(
            CLIP_IMAGE_MEAN,
            dtype=frames.dtype,
            device=frames.device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            CLIP_IMAGE_STD,
            dtype=frames.dtype,
            device=frames.device,
        ).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames

    def encode_image(
        self,
        videos: torch.Tensor | list[torch.Tensor],
        *,
        preprocessed: bool = False,
    ) -> torch.Tensor:
        pixel_values = videos if preprocessed else self.preprocess(videos)
        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("preprocessed videos must be a Tensor.")
        dtype = next(iter(self.model.visual.parameters())).dtype
        pixel_values = pixel_values.to(
            device=self.model.visual.pos_embedding.device, dtype=dtype
        )
        return self.model.visual(pixel_values, use_31_block=True).clone()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.visual(pixel_values, use_31_block=True)


__all__ = [
    "CLIP_IMAGE_MEAN",
    "CLIP_IMAGE_STD",
    "DreamZeroWanImageEncoder",
    "WanCLIPVisionTransformer",
    "WanXLMRobertaCLIPVisual",
    "dreamzero_image_encoder_weight_remap",
]
