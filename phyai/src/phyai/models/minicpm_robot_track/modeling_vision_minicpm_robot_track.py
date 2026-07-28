"""PhyAI-native DINOv3 and SigLIP vision towers for RobotTrack."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from phyai.engine_config import get_engine_config, resolve_engine_defaults
from phyai.layers.attention.attention import Attention
from phyai.layers.conv import Conv2d
from phyai.layers.layer_norm import LayerNorm
from phyai.layers.linear import QKVParallelLinear, RowParallelLinear
from phyai.layers.mlp.dense_mlp import DenseMLP
from phyai.layers.rotary_embedding import rotate_half
from phyai.layers.transformer_block import TransformerBlock
from phyai.models.minicpm_robot_track.configuration_minicpm_robot_track import (
    DINOv3TrackVisionConfig,
    SiglipTrackVisionConfig,
)
from phyai.weights import load_pretrained
from phyai.weights.shards import replicated

_SIGLIP_NORM_HF_NAMES = {
    "input_layernorm": "layer_norm1",
    "post_attention_layernorm": "layer_norm2",
}


def _replicated_parameter(
    shape: tuple[int, ...],
    *,
    hf_key: str,
    dtype: torch.dtype,
    device: torch.device | str,
) -> nn.Parameter:
    parameter = nn.Parameter(
        torch.empty(shape, dtype=dtype, device=device), requires_grad=False
    )
    parameter.hf_keys = [(hf_key, None)]
    parameter.weight_loader = replicated()
    return parameter


class _DINOv3Embeddings(nn.Module):
    def __init__(
        self,
        config: DINOv3TrackVisionConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.cls_token = _replicated_parameter(
            (1, 1, hidden_size),
            hf_key="embeddings.cls_token",
            dtype=dtype,
            device=device,
        )
        # The mask token is unused at inference, but loading it preserves a
        # strict one-to-one checkpoint contract with the upstream DINO tower.
        self.mask_token = _replicated_parameter(
            (1, 1, hidden_size),
            hf_key="embeddings.mask_token",
            dtype=dtype,
            device=device,
        )
        self.register_tokens = _replicated_parameter(
            (1, config.num_register_tokens, hidden_size),
            hf_key="embeddings.register_tokens",
            dtype=dtype,
            device=device,
        )
        self.patch_embeddings = Conv2d(
            config.num_channels,
            hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
            dtype=dtype,
            device=device,
            prefix="embeddings.patch_embeddings",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        batch_size = patches.shape[0]
        return torch.cat(
            (
                self.cls_token.expand(batch_size, -1, -1),
                self.register_tokens.expand(batch_size, -1, -1),
                patches,
            ),
            dim=1,
        )


class _DINOv3Attention(nn.Module):
    def __init__(
        self,
        config: DINOv3TrackVisionConfig,
        *,
        layer_index: int,
        dtype: torch.dtype,
        device: torch.device | str,
        attn_backend: str,
    ) -> None:
        super().__init__()
        prefix = f"layer.{layer_index}.attention"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads,
            bias=True,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.qkv_proj",
        )
        # DINOv3 has Q/V bias tensors but no K bias. Keep the fused QKV
        # parameter's zero-initialized K slice and load only the two supplied legs.
        if self.qkv_proj.bias is None:
            raise RuntimeError("DINOv3 fused QKV projection requires a bias buffer.")
        self.qkv_proj.bias.hf_keys = [
            (f"{prefix}.q_proj.bias", "q"),
            (f"{prefix}.v_proj.bias", "v"),
        ]
        self.attn = Attention(
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            causal=False,
            backend=attn_backend,
        )
        self.o_proj = RowParallelLinear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            input_is_parallel=True,
            bias=True,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.o_proj",
        )

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        fused, _ = self.qkv_proj(hidden_states)
        query, key, value = fused.chunk(3, dim=-1)
        shape = (*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        query = query.view(shape)
        key = key.view(shape)
        value = value.view(shape)

        prefix_count = query.shape[1] - cos.shape[1]
        query_prefix, query_patches = query.split((prefix_count, cos.shape[1]), dim=1)
        key_prefix, key_patches = key.split((prefix_count, cos.shape[1]), dim=1)
        query_patches = query_patches * cos + rotate_half(query_patches) * sin
        key_patches = key_patches * cos + rotate_half(key_patches) * sin
        query = torch.cat((query_prefix, query_patches), dim=1)
        key = torch.cat((key_prefix, key_patches), dim=1)

        output = self.attn(query, key, value).flatten(2)
        output, _ = self.o_proj(output)
        return output


class _DINOv3Layer(nn.Module):
    def __init__(
        self,
        config: DINOv3TrackVisionConfig,
        *,
        layer_index: int,
        dtype: torch.dtype,
        device: torch.device | str,
        attn_backend: str,
        norm_backend: str,
    ) -> None:
        super().__init__()
        prefix = f"layer.{layer_index}"
        self.norm1 = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.norm1",
        )
        self.attention = _DINOv3Attention(
            config,
            layer_index=layer_index,
            dtype=dtype,
            device=device,
            attn_backend=attn_backend,
        )
        self.layer_scale1 = _replicated_parameter(
            (config.hidden_size,),
            hf_key=f"{prefix}.layer_scale1.lambda1",
            dtype=dtype,
            device=device,
        )
        self.norm2 = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=dtype,
            device=device,
            prefix=f"{prefix}.norm2",
        )
        self.mlp = DenseMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="gelu",
            gated=False,
            bias=True,
            params_dtype=dtype,
            prefix=f"{prefix}.mlp",
        )
        # DenseMLP uses fc1/fc2 for plain MLPs; DINOv3 names the same
        # tensors up_proj/down_proj in its checkpoint.
        self.mlp.fc1.weight.hf_keys = [(f"{prefix}.mlp.up_proj.weight", None)]
        self.mlp.fc1.bias.hf_keys = [(f"{prefix}.mlp.up_proj.bias", None)]
        self.mlp.fc2.weight.hf_keys = [(f"{prefix}.mlp.down_proj.weight", None)]
        self.mlp.fc2.bias.hf_keys = [(f"{prefix}.mlp.down_proj.bias", None)]
        self.layer_scale2 = _replicated_parameter(
            (config.hidden_size,),
            hf_key=f"{prefix}.layer_scale2.lambda1",
            dtype=dtype,
            device=device,
        )

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = (
            hidden_states
            + self.attention(self.norm1(hidden_states), cos, sin) * self.layer_scale1
        )
        return hidden_states + self.mlp(self.norm2(hidden_states)) * self.layer_scale2


class MiniCPMRobotTrackDINOv3VisionModel(nn.Module):
    """DINOv3 ViT-S/16 tower returning the 24x24 patch-token grid."""

    def __init__(
        self,
        config: DINOv3TrackVisionConfig | None = None,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
    ) -> None:
        super().__init__()
        config = config or DINOv3TrackVisionConfig()
        if attn_backend is None:
            attn_backend = "sdpa"
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        device = get_engine_config().device.target
        self.config = config
        self.params_dtype = params_dtype
        self.embeddings = _DINOv3Embeddings(config, dtype=params_dtype, device=device)
        self.layer = nn.ModuleList(
            [
                _DINOv3Layer(
                    config,
                    layer_index=index,
                    dtype=params_dtype,
                    device=device,
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix="norm",
        )
        side = config.image_size // config.patch_size
        coords = torch.arange(0.5, side, dtype=torch.float32, device=device) / side
        patch_coords = torch.stack(
            torch.meshgrid(coords, coords, indexing="ij"), dim=-1
        ).flatten(0, 1)
        patch_coords = patch_coords * 2.0 - 1.0
        inv_freq = 1.0 / config.rope_theta ** torch.arange(
            0, 1, 4 / config.head_dim, dtype=torch.float32, device=device
        )
        angles = (2 * math.pi * patch_coords[:, :, None] * inv_freq).flatten(1, 2)
        angles = angles.tile(2)
        self.register_buffer(
            "rope_cos",
            torch.cos(angles).to(params_dtype)[None, :, None, :],
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            torch.sin(angles).to(params_dtype)[None, :, None, :],
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values.to(self.params_dtype))
        for layer in self.layer:
            hidden_states = layer(hidden_states, self.rope_cos, self.rope_sin)
        hidden_states = self.norm(hidden_states)
        prefix_tokens = 1 + self.config.num_register_tokens
        return hidden_states[:, prefix_tokens:, :]


class _PositionEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.weight = _replicated_parameter(
            (num_embeddings, embedding_dim),
            hf_key="vision_model.embeddings.position_embedding.weight",
            dtype=dtype,
            device=device,
        )

    def forward(self) -> torch.Tensor:
        return self.weight


class MiniCPMRobotTrackSiglipVisionModel(nn.Module):
    """SigLIP So400m/14@384 tower returning its 27x27 token grid."""

    def __init__(
        self,
        config: SiglipTrackVisionConfig | None = None,
        *,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
    ) -> None:
        super().__init__()
        config = config or SiglipTrackVisionConfig()
        if attn_backend is None:
            attn_backend = "sdpa"
        params_dtype, attn_backend, norm_backend = resolve_engine_defaults(
            params_dtype, attn_backend, norm_backend
        )
        device = get_engine_config().device.target
        self.config = config
        self.params_dtype = params_dtype
        self.patch_embedding = Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix="vision_model.embeddings.patch_embedding",
        )
        self.position_embedding = _PositionEmbedding(
            config.num_patches,
            config.hidden_size,
            dtype=params_dtype,
            device=device,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    attn_causal=False,
                    attn_bias=True,
                    attn_out_bias=True,
                    rope=None,
                    mlp_gated=False,
                    mlp_activation="gelu_pytorch_tanh",
                    mlp_bias=True,
                    norm_type="layernorm",
                    norm_eps=config.layer_norm_eps,
                    norm_bias=True,
                    norm_hf_names=_SIGLIP_NORM_HF_NAMES,
                    attn_out_hf_name="out_proj",
                    attn_backend=attn_backend,
                    norm_backend=norm_backend,
                    params_dtype=params_dtype,
                    prefix=f"vision_model.encoder.layers.{index}",
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.post_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            backend=norm_backend,
            bias=True,
            dtype=params_dtype,
            device=device,
            prefix="vision_model.post_layernorm",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embedding(pixel_values.to(self.params_dtype))
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.position_embedding()
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)


def _siglip_vision_weight_remap(key: str) -> str | None:
    if not key.startswith("vision_model.") or key.startswith("vision_model.head."):
        return None
    return key


class MiniCPMRobotTrackPhyAIVisionEncoder:
    """Own and load the two PhyAI-native RobotTrack vision towers."""

    def __init__(
        self,
        *,
        dino_checkpoint_dir: str | Path,
        siglip_checkpoint_dir: str | Path,
        params_dtype: torch.dtype | None = None,
        attn_backend: str | None = None,
        norm_backend: str | None = None,
    ) -> None:
        # FlashInfer single-prefill is numerically invalid for the released
        # SigLIP geometry (head_dim=72, sequence_length=729) on this stack.
        # SDPA remains behind PhyAI's Attention interface and selects a valid
        # CUDA attention kernel for both RobotTrack vision towers.
        if attn_backend is None:
            attn_backend = "sdpa"
        self.dino = MiniCPMRobotTrackDINOv3VisionModel(
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        dino_report = load_pretrained(
            self.dino, dino_checkpoint_dir, strict=True, progress=False
        )
        if dino_report.missing or dino_report.unexpected:
            raise RuntimeError(f"DINOv3 weight mismatch: {dino_report.summary()}")
        self.dino.eval()

        self.siglip = MiniCPMRobotTrackSiglipVisionModel(
            params_dtype=params_dtype,
            attn_backend=attn_backend,
            norm_backend=norm_backend,
        )
        siglip_report = load_pretrained(
            self.siglip,
            siglip_checkpoint_dir,
            remap=_siglip_vision_weight_remap,
            strict=True,
            progress=False,
        )
        if siglip_report.missing or siglip_report.unexpected:
            raise RuntimeError(f"SigLIP weight mismatch: {siglip_report.summary()}")
        self.siglip.eval()

    @staticmethod
    def pool_siglip(tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, hidden_size = tokens.shape
        input_side = round(token_count**0.5)
        if input_side * input_side != token_count:
            raise ValueError(f"SigLIP token count must be square, got {token_count}.")
        features = tokens.transpose(1, 2).reshape(
            batch_size, hidden_size, input_side, input_side
        )
        features = F.adaptive_avg_pool2d(features, (24, 24))
        return features.flatten(2).transpose(1, 2).contiguous()

    def close(self) -> None:
        self.dino = None
        self.siglip = None


__all__ = [
    "MiniCPMRobotTrackDINOv3VisionModel",
    "MiniCPMRobotTrackPhyAIVisionEncoder",
    "MiniCPMRobotTrackSiglipVisionModel",
]
