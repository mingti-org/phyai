"""Configs for DreamZero inference.

The public DreamZero-DROID checkpoint stores the policy configuration under
``action_head_cfg.config`` in the top-level ``config.json``. The dataclasses in
this module lift the inference-relevant fields into typed, frozen configs so the
modeling, runner, and scheduler layers can stay independent of the upstream
Hydra/Transformers schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phyai.models.configuration import PretrainedConfig


def _dig(data: dict[str, Any], path: str) -> tuple[bool, Any]:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


@dataclass(frozen=True)
class DreamZeroDiTConfig(PretrainedConfig):
    """Causal Wan DiT config used by DreamZero's action head."""

    nested_sources = {
        "pretrained_path": "diffusion_model_pretrained_path",
    }

    pretrained_path: str | None = None
    dim: int = 5120
    eps: float = 1e-6
    ffn_dim: int = 13824
    frame_seqlen: int = 880
    freq_dim: int = 256
    in_dim: int = 36
    max_chunk_size: int = 4
    model_type: str = "i2v"
    num_action_per_block: int = 24
    num_frame_per_block: int = 2
    num_heads: int = 40
    num_layers: int = 40
    num_state_per_block: int = 1
    out_dim: int = 16

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"dim={self.dim} must be positive.")
        if self.num_heads <= 0 or self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads={self.num_heads}."
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers={self.num_layers} must be positive.")
        if self.ffn_dim <= 0:
            raise ValueError(f"ffn_dim={self.ffn_dim} must be positive.")
        if self.frame_seqlen <= 0:
            raise ValueError(f"frame_seqlen={self.frame_seqlen} must be positive.")
        if self.freq_dim <= 0 or self.freq_dim % 2:
            raise ValueError(f"freq_dim={self.freq_dim} must be a positive even int.")
        if self.num_frame_per_block <= 0:
            raise ValueError(
                f"num_frame_per_block={self.num_frame_per_block} must be positive."
            )
        if self.num_action_per_block <= 0:
            raise ValueError(
                f"num_action_per_block={self.num_action_per_block} must be positive."
            )
        if self.num_state_per_block <= 0:
            raise ValueError(
                f"num_state_per_block={self.num_state_per_block} must be positive."
            )

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    def validate_tp_size(self, tp_size: int) -> None:
        """Validate the tensor-parallel degree planned for this DiT."""
        if tp_size <= 0:
            raise ValueError(f"tp_size={tp_size} must be positive.")
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads={self.num_heads} must be divisible by tp_size={tp_size}."
            )
        if self.ffn_dim % tp_size != 0:
            raise ValueError(
                f"ffn_dim={self.ffn_dim} must be divisible by tp_size={tp_size}."
            )
        if self.dim % tp_size != 0:
            raise ValueError(f"dim={self.dim} must be divisible by tp_size={tp_size}.")


@dataclass(frozen=True)
class DreamZeroTextEncoderConfig(PretrainedConfig):
    """DreamZero Wan/T5 text encoder checkpoint config."""

    nested_sources = {
        "pretrained_path": "text_encoder_pretrained_path",
    }

    pretrained_path: str | None = None
    vocab: int = 256384
    dim: int = 4096
    dim_attn: int = 4096
    dim_ffn: int = 10240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    shared_pos: bool = False
    dropout: float = 0.0
    max_length: int = 512

    def __post_init__(self) -> None:
        if self.vocab <= 0:
            raise ValueError(f"vocab={self.vocab} must be positive.")
        if self.dim <= 0:
            raise ValueError(f"dim={self.dim} must be positive.")
        if self.dim_attn <= 0:
            raise ValueError(f"dim_attn={self.dim_attn} must be positive.")
        if self.dim_ffn <= 0:
            raise ValueError(f"dim_ffn={self.dim_ffn} must be positive.")
        if self.num_heads <= 0 or self.dim_attn % self.num_heads != 0:
            raise ValueError(
                f"dim_attn={self.dim_attn} must be divisible by "
                f"num_heads={self.num_heads}."
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers={self.num_layers} must be positive.")
        if self.num_buckets <= 0:
            raise ValueError(f"num_buckets={self.num_buckets} must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout={self.dropout} must be in [0, 1).")
        if self.max_length <= 0:
            raise ValueError(f"max_length={self.max_length} must be positive.")

    @property
    def head_dim(self) -> int:
        return self.dim_attn // self.num_heads


@dataclass(frozen=True)
class DreamZeroImageEncoderConfig(PretrainedConfig):
    """DreamZero CLIP/XLM-R image encoder checkpoint config."""

    nested_sources = {
        "pretrained_path": "image_encoder_pretrained_path",
    }

    pretrained_path: str | None = None
    embed_dim: int = 1024
    image_size: int = 224
    patch_size: int = 14
    vision_dim: int = 1280
    vision_mlp_ratio: int = 4
    vision_heads: int = 16
    vision_layers: int = 32
    vision_pool: str = "token"
    vision_pre_norm: bool = True
    vision_post_norm: bool = False
    activation: str = "gelu"
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    embedding_dropout: float = 0.0
    norm_eps: float = 1e-5
    num_channels: int = 3
    feature_layer: int = 31

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size={self.image_size} must be positive.")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size={self.patch_size} must be positive.")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by "
                f"patch_size={self.patch_size}."
            )
        if self.vision_dim <= 0:
            raise ValueError(f"vision_dim={self.vision_dim} must be positive.")
        if self.vision_heads <= 0 or self.vision_dim % self.vision_heads != 0:
            raise ValueError(
                f"vision_dim={self.vision_dim} must be divisible by "
                f"vision_heads={self.vision_heads}."
            )
        if self.vision_layers <= 0:
            raise ValueError(f"vision_layers={self.vision_layers} must be positive.")
        if not 0 < self.feature_layer <= self.vision_layers:
            raise ValueError(
                f"feature_layer={self.feature_layer} must be in "
                f"[1, {self.vision_layers}]."
            )
        if self.vision_mlp_ratio <= 0:
            raise ValueError(
                f"vision_mlp_ratio={self.vision_mlp_ratio} must be positive."
            )
        if self.vision_pool not in {"token", "token_fc", "attn_pool"}:
            raise ValueError(f"unsupported vision_pool={self.vision_pool!r}.")
        if self.activation not in {"quick_gelu", "gelu", "swi_glu"}:
            raise ValueError(f"unsupported activation={self.activation!r}.")
        if self.num_channels <= 0:
            raise ValueError(f"num_channels={self.num_channels} must be positive.")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.vision_dim // self.vision_heads


@dataclass(frozen=True)
class DreamZeroVAEConfig(PretrainedConfig):
    """DreamZero Wan2.1 VAE checkpoint config."""

    nested_sources = {
        "pretrained_path": "vae_pretrained_path",
    }

    pretrained_path: str | None = None
    z_dim: int = 16
    base_dim: int = 96
    decoder_base_dim: int = 96
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    out_channels: int = 3
    patch_size: int = 1
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.dim_mult, tuple):
            object.__setattr__(self, "dim_mult", tuple(self.dim_mult))
        if not isinstance(self.temperal_downsample, tuple):
            object.__setattr__(
                self, "temperal_downsample", tuple(self.temperal_downsample)
            )
        if self.z_dim <= 0:
            raise ValueError(f"z_dim={self.z_dim} must be positive.")
        if self.base_dim <= 0:
            raise ValueError(f"base_dim={self.base_dim} must be positive.")
        if self.decoder_base_dim <= 0:
            raise ValueError(
                f"decoder_base_dim={self.decoder_base_dim} must be positive."
            )
        if self.out_channels <= 0:
            raise ValueError(f"out_channels={self.out_channels} must be positive.")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size={self.patch_size} must be positive.")
        if self.scale_factor_temporal <= 0:
            raise ValueError(
                f"scale_factor_temporal={self.scale_factor_temporal} must be positive."
            )
        if self.scale_factor_spatial <= 0:
            raise ValueError(
                f"scale_factor_spatial={self.scale_factor_spatial} must be positive."
            )

    @property
    def is_wan21(self) -> bool:
        return (
            self.z_dim == 16
            and self.base_dim == 96
            and self.decoder_base_dim == 96
            and self.out_channels == 3
            and self.patch_size == 1
            and self.scale_factor_spatial == 8
        )


@dataclass(frozen=True)
class DreamZeroConfig(PretrainedConfig):
    """Top-level DreamZero policy config."""

    nested_sources = {
        "action_dim": (
            "action_head_cfg.config.action_dim",
            "action_dim",
        ),
        "action_horizon": (
            "action_head_cfg.config.action_horizon",
            "action_horizon",
        ),
        "max_action_dim": "action_head_cfg.config.max_action_dim",
        "max_state_dim": "action_head_cfg.config.max_state_dim",
        "hidden_size": "action_head_cfg.config.hidden_size",
        "input_embedding_dim": "action_head_cfg.config.input_embedding_dim",
        "num_frames": "action_head_cfg.config.num_frames",
        "num_inference_timesteps": "action_head_cfg.config.num_inference_timesteps",
        "num_timestep_buckets": "action_head_cfg.config.num_timestep_buckets",
        "num_frame_per_block": "action_head_cfg.config.num_frame_per_block",
        "cfg_scale": "action_head_cfg.config.cfg_scale",
        "sigma_shift": "action_head_cfg.config.sigma_shift",
        "noise_s": "action_head_cfg.config.noise_s",
        "decouple_inference_noise": "action_head_cfg.config.decouple_inference_noise",
        "video_inference_final_noise": (
            "action_head_cfg.config.video_inference_final_noise"
        ),
        "tiled": "action_head_cfg.config.tiled",
        "tile_size_height": "action_head_cfg.config.tile_size_height",
        "tile_size_width": "action_head_cfg.config.tile_size_width",
        "tile_stride_height": "action_head_cfg.config.tile_stride_height",
        "tile_stride_width": "action_head_cfg.config.tile_stride_width",
        "dit": "action_head_cfg.config.diffusion_model_cfg",
        "text_encoder": "action_head_cfg.config.text_encoder_cfg",
        "image_encoder": "action_head_cfg.config.image_encoder_cfg",
        "vae": "action_head_cfg.config.vae_cfg",
    }

    dit: DreamZeroDiTConfig = field(default_factory=DreamZeroDiTConfig)
    text_encoder: DreamZeroTextEncoderConfig = field(
        default_factory=DreamZeroTextEncoderConfig
    )
    image_encoder: DreamZeroImageEncoderConfig = field(
        default_factory=DreamZeroImageEncoderConfig
    )
    vae: DreamZeroVAEConfig = field(default_factory=DreamZeroVAEConfig)

    action_dim: int = 32
    action_horizon: int = 24
    max_action_dim: int = 32
    max_state_dim: int = 64
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    num_frames: int = 33
    num_inference_timesteps: int = 4
    num_timestep_buckets: int = 1000
    num_frame_per_block: int = 2
    cfg_scale: float = 5.0
    sigma_shift: float = 5.0
    noise_s: float = 0.999
    decouple_inference_noise: bool = False
    video_inference_final_noise: float = 0.8

    tiled: bool = False
    tile_size_height: int = 34
    tile_size_width: int = 34
    tile_stride_height: int = 18
    tile_stride_width: int = 16

    torch_dtype: str = "bfloat16"
    model_dtype: str = "float32"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DreamZeroConfig":
        data = dict(data)
        found, value = _dig(
            data,
            "action_head_cfg.config.diffusion_model_cfg.hidden_size",
        )
        # Upstream top-level hidden_size belongs to the VLA backbone and is 0
        # for DreamZero-DROID. The DiT action/state MLP hidden is owned by
        # CausalWanModel and defaults to 1024 when absent from the config.
        data["hidden_size"] = int(value) if found else 1024
        return super().from_dict(data)

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError(f"action_dim={self.action_dim} must be positive.")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon={self.action_horizon} must be positive.")
        if self.max_action_dim < self.action_dim:
            raise ValueError(
                f"max_action_dim={self.max_action_dim} must be >= "
                f"action_dim={self.action_dim}."
            )
        if self.max_state_dim <= 0:
            raise ValueError(f"max_state_dim={self.max_state_dim} must be positive.")
        if self.num_frames <= 0:
            raise ValueError(f"num_frames={self.num_frames} must be positive.")
        if (
            self.num_inference_timesteps is not None
            and self.num_inference_timesteps <= 0
        ):
            raise ValueError(
                "num_inference_timesteps must be positive when it is configured."
            )
        if self.num_timestep_buckets <= 0:
            raise ValueError(
                f"num_timestep_buckets={self.num_timestep_buckets} must be positive."
            )
        if self.num_frame_per_block != self.dit.num_frame_per_block:
            raise ValueError(
                f"num_frame_per_block={self.num_frame_per_block} must match "
                f"dit.num_frame_per_block={self.dit.num_frame_per_block}."
            )
        if self.action_horizon != self.dit.num_action_per_block:
            raise ValueError(
                f"action_horizon={self.action_horizon} must match "
                f"dit.num_action_per_block={self.dit.num_action_per_block}."
            )


__all__ = [
    "DreamZeroConfig",
    "DreamZeroDiTConfig",
    "DreamZeroImageEncoderConfig",
    "DreamZeroTextEncoderConfig",
    "DreamZeroVAEConfig",
]
