"""DreamZero Wan2.1 CLIP features for image conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from phyai.engine_config import get_engine_config
from phyai.layers import LayerNorm
from phyai.layers.attention import Attention, AttnCtx
from phyai.layers.linear import ColumnParallelLinear, RowParallelLinear
from phyai.layers.conv import Conv2d
from phyai.layers.mlp import DenseMLP
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroImageEncoderConfig
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import resolve_mesh
from phyai.weights.shards import replicated


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def dreamzero_image_encoder_weight_remap(
    name: str, *, config: DreamZeroImageEncoderConfig
) -> str | None:
    """Map the visual checkpoint and omit layers after the requested features."""
    component_key = name.startswith("action_head.image_encoder.")
    name = name.removeprefix("action_head.image_encoder.").removeprefix("model.")
    if name == "log_scale" or name == "visual.head":
        return None
    if name.startswith(("visual.head.", "visual.post_norm.")):
        return None
    if not component_key and not name.startswith("visual."):
        return None
    if name.startswith("visual.transformer."):
        index = name.split(".", 3)[2]
        if (
            index.isdecimal()
            and config.feature_layer <= int(index) < config.vision_layers
        ):
            return None
        if config.activation == "gelu":
            name = name.replace(".mlp.0.", ".mlp.fc1.").replace(".mlp.2.", ".mlp.fc2.")
        elif config.activation == "swi_glu":
            name = name.replace(".mlp.fc3.", ".mlp.down_proj.")
    return name


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class CLIPLayerNorm(LayerNorm):
    """Keep CLIP's residual dtype when autocast promotes torch LayerNorm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).to(dtype=x.dtype)


class CLIPQuickGELUMLP(nn.Module):
    """QuickGELU differs from the exact and tanh GELU forms in DenseMLP."""

    def __init__(
        self,
        dim: int,
        mid_dim: int,
        *,
        params_dtype: torch.dtype,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
        prefix: str,
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            dim,
            mid_dim,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.0",
        )
        self.activation = QuickGELU()
        self.fc2 = RowParallelLinear(
            mid_dim,
            dim,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.activation(x)
        x, _ = self.fc2(x)
        return x


class CLIPSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        params_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
        prefix: str,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        tp_size = resolve_mesh(mesh).group_size("dense_tp")
        if num_heads % tp_size:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by tp_size={tp_size}."
            )
        self.num_heads = num_heads // tp_size
        self.head_dim = dim // num_heads
        self.to_qkv = ColumnParallelLinear(
            dim,
            dim * 3,
            output_sizes=[dim, dim, dim],
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.to_qkv",
        )
        self.proj = RowParallelLinear(
            dim,
            dim,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.proj",
        )
        self.attention = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            causal=False,
            kernel_role="dreamzero.image_attention",
        )

    def forward(self, x: torch.Tensor, ctx: AttnCtx | None = None) -> torch.Tensor:
        qkv, _ = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)
        x = self.attention(q, k, v, ctx=ctx)
        x = rearrange(x, "b s n d -> b s (n d)")
        return self.proj(x)[0]


class CLIPAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int,
        num_heads: int,
        *,
        post_norm: bool = False,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
        params_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
        prefix: str,
    ) -> None:
        super().__init__()
        if activation not in {"quick_gelu", "gelu", "swi_glu"}:
            raise ValueError(f"unsupported activation={activation!r}.")
        self.post_norm = post_norm
        self.norm1 = CLIPLayerNorm(
            dim,
            eps=norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm1",
        )
        self.attn = CLIPSelfAttention(
            dim,
            num_heads,
            params_dtype=params_dtype,
            device=device,
            mesh=mesh,
            prefix=f"{prefix}.attn",
        )
        self.norm2 = CLIPLayerNorm(
            dim,
            eps=norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.norm2",
        )
        mid_dim = int(dim * mlp_ratio)
        if activation == "quick_gelu":
            self.mlp = CLIPQuickGELUMLP(
                dim,
                mid_dim,
                params_dtype=params_dtype,
                device=device,
                mesh=mesh,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DenseMLP(
                dim,
                mid_dim,
                activation="silu" if activation == "swi_glu" else "gelu",
                gated=activation == "swi_glu",
                gated_hf_legs=("fc1", "fc2"),
                bias=True,
                sequence_parallel=False,
                params_dtype=params_dtype,
                device=device,
                mesh=mesh,
                prefix=f"{prefix}.mlp",
            )

    def forward(self, x: torch.Tensor, ctx: AttnCtx | None = None) -> torch.Tensor:
        if self.post_norm:
            x = x + self.norm1(self.attn(x, ctx=ctx))
            return x + self.norm2(self.mlp(x))
        x = x + self.attn(self.norm1(x), ctx=ctx)
        return x + self.mlp(self.norm2(x))


class WanCLIPVisionTransformer(nn.Module):
    """CLIP visual tower truncated at the configured feature layer."""

    def __init__(
        self,
        config: DreamZeroImageEncoderConfig,
        *,
        params_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
        prefix: str = "visual",
    ) -> None:
        super().__init__()
        device = get_engine_config().device.target if device is None else device
        self.has_cls_token = config.vision_pool in {"token", "token_fc"}
        self.patch_embedding = Conv2d(
            config.num_channels,
            config.vision_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=not config.vision_pre_norm,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.patch_embedding",
        )
        if self.has_cls_token:
            self.cls_embedding = nn.Parameter(
                torch.empty(1, 1, config.vision_dim, dtype=params_dtype, device=device),
                requires_grad=False,
            )
            self.cls_embedding.hf_keys = [(f"{prefix}.cls_embedding", None)]
            self.cls_embedding.weight_loader = replicated()
        self.pos_embedding = nn.Parameter(
            torch.empty(
                1,
                config.num_patches + int(self.has_cls_token),
                config.vision_dim,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.pos_embedding.hf_keys = [(f"{prefix}.pos_embedding", None)]
        self.pos_embedding.weight_loader = replicated()
        self.pre_norm = (
            CLIPLayerNorm(
                config.vision_dim,
                eps=config.norm_eps,
                dtype=params_dtype,
                device=device,
                prefix=f"{prefix}.pre_norm",
            )
            if config.vision_pre_norm
            else None
        )
        self.transformer = nn.ModuleList(
            CLIPAttentionBlock(
                config.vision_dim,
                config.vision_mlp_ratio,
                config.vision_heads,
                post_norm=config.vision_post_norm,
                activation=config.activation,
                norm_eps=config.norm_eps,
                params_dtype=params_dtype,
                device=device,
                mesh=mesh,
                prefix=f"{prefix}.transformer.{index}",
            )
            for index in range(config.feature_layer)
        )

    def forward(
        self, x: torch.Tensor, *, attn_contexts: tuple[AttnCtx, ...] | None = None
    ) -> torch.Tensor:
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        if self.has_cls_token:
            x = torch.cat(
                [self.cls_embedding.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1
            )
        x = x + self.pos_embedding.to(x.dtype)
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        if attn_contexts is not None and len(attn_contexts) != len(self.transformer):
            raise ValueError("attention contexts must match the CLIP layers.")
        for index, layer in enumerate(self.transformer):
            x = layer(x, ctx=None if attn_contexts is None else attn_contexts[index])
        return x


class DreamZeroWanImageEncoder(nn.Module):
    """Preprocess observations and produce the CLIP features used by the DiT."""

    def __init__(
        self,
        config: DreamZeroImageEncoderConfig | None = None,
        *,
        params_dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        mesh: Mesh | str = "model",
    ) -> None:
        super().__init__()
        self.config = config or DreamZeroImageEncoderConfig()
        self.visual = WanCLIPVisionTransformer(
            self.config, params_dtype=params_dtype, device=device, mesh=mesh
        )

    def preprocess(self, videos: torch.Tensor) -> torch.Tensor:
        """Resize [-1, 1] RGB images shaped [B, C, H, W] or [B, T, C, H, W]."""
        if videos.ndim not in (4, 5):
            raise ValueError(f"videos must be 4-D or 5-D; got {tuple(videos.shape)}.")
        frames = videos.flatten(0, 1) if videos.ndim == 5 else videos
        if frames.shape[1] != 3:
            raise ValueError(f"CLIP expects three RGB channels; got {frames.shape[1]}.")
        frames = F.interpolate(
            frames,
            size=(self.config.image_size, self.config.image_size),
            mode="bicubic",
            align_corners=False,
        )
        frames = frames.mul(0.5).add(0.5)
        mean = torch.tensor(
            CLIP_IMAGE_MEAN, dtype=frames.dtype, device=frames.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            CLIP_IMAGE_STD, dtype=frames.dtype, device=frames.device
        ).view(1, 3, 1, 1)
        return (frames - mean) / std

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        attn_contexts: tuple[AttnCtx, ...] | None = None,
    ) -> torch.Tensor:
        return self.visual(pixel_values, attn_contexts=attn_contexts)


__all__ = [
    "CLIP_IMAGE_MEAN",
    "CLIP_IMAGE_STD",
    "DreamZeroWanImageEncoder",
    "WanCLIPVisionTransformer",
    "dreamzero_image_encoder_weight_remap",
]
