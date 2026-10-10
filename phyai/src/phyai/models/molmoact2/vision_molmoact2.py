"""MolmoAct2 patch encoder, attention pooling, and image projection."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from phyai.layers.activation import SiLU
from phyai.layers.mlp import DenseMLP
from phyai.engine_config import get_engine_config
from phyai.layers.linear import ReplicatedLinear
from phyai.weights.shards import replicated
from phyai.layers.attention import AttnMask, Attention
from phyai.layers.layer_norm import LayerNorm
from phyai.models.molmoact2.configuration_molmoact2 import (
    MolmoAct2ViTConfig,
    MolmoAct2AdapterConfig,
)


class ViTMultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        config,
        *,
        input_dim: int | None = None,
        params_dtype,
        device,
        prefix,
        attn_backend=None,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.float32_attention = config.float32_attention
        input_dim = input_dim or config.hidden_size
        for name, heads in (
            ("wq", self.num_heads),
            ("wk", self.num_key_value_heads),
            ("wv", self.num_key_value_heads),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_dim,
                    heads * self.head_dim,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.{name}",
                ),
            )
        self.wo = ReplicatedLinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.wo",
        )
        self.attention = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            causal=False,
            backend=attn_backend,
        )

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_kv = inputs_q if inputs_kv is None else inputs_kv
        query = self.wq(inputs_q)[0].reshape(
            *inputs_q.shape[:2], self.num_heads, self.head_dim
        )
        key = self.wk(inputs_kv)[0].reshape(
            *inputs_kv.shape[:2], self.num_key_value_heads, self.head_dim
        )
        value = self.wv(inputs_kv)[0].reshape_as(key)
        output_dtype = query.dtype
        if self.float32_attention:
            query, key, value = query.float(), key.float(), value.float()
        mask = None if key_mask is None else AttnMask.from_key_mask(key_mask)
        output = self.attention(query, key, value, mask=mask)
        return self.wo(output.to(output_dtype).flatten(2))[0]


class MolmoAct2VisionBlock(nn.Module):
    def __init__(
        self,
        config,
        *,
        params_dtype,
        device,
        prefix,
        attn_backend=None,
        norm_backend=None,
    ):
        super().__init__()
        self.attention = ViTMultiHeadDotProductAttention(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.attention",
            attn_backend=attn_backend,
        )
        self.feed_forward = DenseMLP(
            config.hidden_size,
            config.intermediate_size,
            activation="gelu_tanh",
            gated=False,
            bias=True,
            sequence_parallel=False,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.feed_forward",
        )
        for parameter in self.feed_forward.parameters():
            parameter.hf_keys = [
                (
                    key.replace(".feed_forward.fc1.", ".feed_forward.w1.").replace(
                        ".feed_forward.fc2.", ".feed_forward.w2."
                    ),
                    shard_id,
                )
                for key, shard_id in parameter.hf_keys
            ]
        self.attention_norm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.attention_norm",
            backend=norm_backend,
        )
        self.ffn_norm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.ffn_norm",
            backend=norm_backend,
        )

    def post_load(self) -> None:
        # FlashInfer stores affine weights in FP32; match the reference's parameter rounding.
        dtype = self.attention.wq.weight.dtype
        for norm in (self.attention_norm, self.ffn_norm):
            for parameter in norm.parameters():
                if parameter.dtype != dtype:
                    parameter.data.copy_(parameter.data.to(dtype))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.attention_norm(value))
        return value + self.feed_forward(self.ffn_norm(value))


class MolmoAct2VisionBlockCollection(nn.Module):
    def __init__(self, config, *, prefix, **kwargs):
        super().__init__()
        self.resblocks = nn.ModuleList(
            MolmoAct2VisionBlock(config, prefix=f"{prefix}.resblocks.{index}", **kwargs)
            for index in range(config.num_hidden_layers)
        )

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden_states = []
        for block in self.resblocks:
            value = block(value)
            hidden_states.append(value)
        return tuple(hidden_states)


class MolmoAct2VisionTransformer(nn.Module):
    def __init__(
        self,
        config: MolmoAct2ViTConfig,
        *,
        params_dtype=torch.bfloat16,
        device=None,
        prefix="model.vision_backbone.image_vit",
        attn_backend=None,
        norm_backend=None,
    ):
        super().__init__()
        device = device or get_engine_config().device.target
        self.config = config
        self.num_prefix_tokens = 0
        self.positional_embedding = nn.Parameter(
            torch.empty(
                config.image_num_pos,
                config.hidden_size,
                dtype=params_dtype,
                device=device,
            ),
            requires_grad=False,
        )
        self.positional_embedding.hf_keys = [(f"{prefix}.positional_embedding", None)]
        self.positional_embedding.weight_loader = replicated()
        self.patch_embedding = ReplicatedLinear(
            config.image_patch_size**2 * 3,
            config.hidden_size,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.patch_embedding",
        )
        self.transformer = MolmoAct2VisionBlockCollection(
            config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.transformer",
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )

    def add_pos_emb(
        self, value: torch.Tensor, patch_num: tuple[int, int]
    ) -> torch.Tensor:
        side = math.isqrt(self.config.image_num_pos)
        pos = self.positional_embedding.reshape(side, side, -1)
        if patch_num != (side, side):
            pos = F.interpolate(
                pos.permute(2, 0, 1)[None].float(),
                size=patch_num,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0].permute(1, 2, 0)
        return value + pos.reshape(-1, pos.shape[-1]).to(value.dtype)[None]

    def forward(
        self, images: torch.Tensor, patch_num: tuple[int, int] | None = None
    ) -> tuple[torch.Tensor, ...]:
        if patch_num is None:
            patch_num = tuple(
                size // self.config.image_patch_size
                for size in self.config.image_default_input_size
            )
        value = self.patch_embedding(images)[0]
        return self.transformer(self.add_pos_emb(value, patch_num))


class ImageProjectorMLP(nn.Module):
    def __init__(self, config, *, params_dtype, device, prefix):
        super().__init__()
        self.hidden_act = config.hidden_act
        self.act = SiLU()
        for name, input_dim, output_dim in (
            ("w1", config.hidden_size, config.intermediate_size),
            ("w2", config.intermediate_size, config.text_hidden_size),
            ("w3", config.hidden_size, config.intermediate_size),
        ):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    input_dim,
                    output_dim,
                    bias=False,
                    params_dtype=params_dtype,
                    device=device,
                    prefix=f"{prefix}.{name}",
                ),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(value)[0], self.w3(value)[0]
        if self.hidden_act != "silu":
            raise ValueError(f"Unsupported adapter activation: {self.hidden_act}")
        return self.w2(self.act(gate) * up)[0]


class MolmoAct2VisionBackbone(nn.Module):
    def __init__(
        self,
        vit_config: MolmoAct2ViTConfig,
        adapter_config: MolmoAct2AdapterConfig,
        *,
        params_dtype=torch.bfloat16,
        device=None,
        prefix="model.vision_backbone",
        attn_backend=None,
        norm_backend=None,
    ):
        super().__init__()
        device = device or get_engine_config().device.target
        self.vit_config = vit_config
        self.adapter_config = adapter_config
        self.vit_layers = tuple(
            index if index >= 0 else index + vit_config.num_hidden_layers
            for index in adapter_config.vit_layers
        )
        self.image_vit = MolmoAct2VisionTransformer(
            replace(vit_config, num_hidden_layers=max(self.vit_layers) + 1),
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.image_vit",
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            adapter_config,
            input_dim=vit_config.hidden_size * len(self.vit_layers),
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.image_pooling_2d",
            attn_backend=attn_backend,
        )
        self.image_projector = ImageProjectorMLP(
            adapter_config,
            params_dtype=params_dtype,
            device=device,
            prefix=f"{prefix}.image_projector",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.image_vit.patch_embedding.weight.device

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        batch, crops, patches, pixels = images.shape
        hidden = self.image_vit(images.reshape(batch * crops, patches, pixels))
        features = torch.cat(tuple(hidden[index] for index in self.vit_layers), dim=-1)
        return features.reshape(batch, crops, patches, -1)

    def forward(
        self, images: torch.Tensor, pooled_patches_idx: torch.Tensor
    ) -> torch.Tensor:
        if images.ndim != 4 or pooled_patches_idx.ndim != 3:
            raise ValueError("images and pooled_patches_idx must have rank 4 and 3")
        images = images.to(device=self.device)
        # The reference snaps processor floats back to the uint8 pixel grid.
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        elif images.is_floating_point():
            images = torch.round((images.float() + 1.0) * 0.5 * 255.0)
            images = images.clamp(0.0, 255.0) / 255.0
        else:
            raise ValueError("images must contain uint8 or normalized floating pixels")
        images = (images * 2.0 - 1.0).to(self.dtype)
        features = self.encode_image(images)
        batch, _, _, width = features.shape
        pooled_patches_idx = pooled_patches_idx.to(device=self.device)
        valid = pooled_patches_idx >= 0
        valid_token = valid.any(dim=-1)
        batch_idx = torch.arange(batch, device=self.device)[:, None, None]
        to_pool = features.reshape(batch, -1, width)[
            batch_idx, pooled_patches_idx.clamp_min(0)
        ]
        to_pool = to_pool * valid[..., None].to(self.dtype)
        to_pool = to_pool.reshape(-1, pooled_patches_idx.shape[-1], width)
        if self.adapter_config.pooling_attention_mask:
            key_mask = valid.reshape(-1, valid.shape[-1])
            denom = key_mask.float().sum(dim=-1).clamp_min(1)
            query = to_pool.sum(dim=-2, keepdim=True) / denom[:, None, None].to(
                to_pool.dtype
            )
        else:
            key_mask = None
            query = to_pool.mean(dim=-2, keepdim=True)
        pooled = self.image_pooling_2d(query, to_pool, key_mask=key_mask)
        projected = self.image_projector(pooled.reshape(batch, -1, pooled.shape[-1]))
        return projected.reshape(-1, projected.shape[-1])[valid_token.flatten()]
