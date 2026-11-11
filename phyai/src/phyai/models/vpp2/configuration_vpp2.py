"""Architecture and inference configuration for VPP2 release checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class VPP2ActionConfig(PretrainedConfig):
    hidden_dim: int = 1024
    action_dim: int = 16
    ffn_dim: int = 4096
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    num_heads: int = 40
    attn_head_dim: int = 128
    num_layers: int = 40
    qk_norm: bool = True
    cross_attn_norm: bool = False
    max_position_embeddings: int = 1024

    def __post_init__(self) -> None:
        for name in (
            "hidden_dim",
            "action_dim",
            "ffn_dim",
            "text_dim",
            "freq_dim",
            "num_heads",
            "attn_head_dim",
            "num_layers",
            "max_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.attn_head_dim % 2 or self.freq_dim % 2:
            raise ValueError("attn_head_dim and freq_dim must be even")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    @property
    def attn_hidden_dim(self) -> int:
        return self.num_heads * self.attn_head_dim

    def validate_tp_size(self, tp_size: int) -> None:
        if tp_size <= 0 or self.num_heads % tp_size or self.ffn_dim % tp_size:
            raise ValueError("TP size must divide the number of heads and FFN width")


@dataclass(frozen=True)
class VPP2VideoConfig(PretrainedConfig):
    hidden_dim: int = 5120
    ffn_dim: int = 13824
    text_dim: int = 4096
    freq_dim: int = 256
    eps: float = 1e-6
    num_heads: int = 40
    attn_head_dim: int = 128
    num_layers: int = 40
    in_dim: int = 36
    out_dim: int = 16
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    image_dim: int = 1280
    image_tokens: int = 257
    condition_channels: int = 20
    qk_norm: bool = True
    cross_attn_norm: bool = True
    max_position_embeddings: int = 1024
    video_attention_mask_mode: str = "bidirectional"
    model_type: str = "i2v"

    nested_sources: ClassVar[Mapping[str, str | tuple[str, ...]]] = {
        "hidden_dim": "dim",
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "patch_size", tuple(self.patch_size))
        for name in (
            "hidden_dim",
            "ffn_dim",
            "text_dim",
            "freq_dim",
            "num_heads",
            "attn_head_dim",
            "num_layers",
            "in_dim",
            "out_dim",
            "image_dim",
            "image_tokens",
            "max_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.patch_size) != 3 or any(p <= 0 for p in self.patch_size):
            raise ValueError("patch_size must contain three positive dimensions")
        if self.attn_head_dim < 6 or self.attn_head_dim % 2 or self.freq_dim % 2:
            raise ValueError("Video head dimensions must be even and at least six")
        if self.in_dim != self.out_dim + self.condition_channels:
            raise ValueError("in_dim must include latent and conditioning channels")
        if self.condition_channels != self.out_dim + 4:
            raise ValueError("Wan conditioning requires four mask channels")
        if self.model_type != "i2v":
            raise ValueError("VPP2 release checkpoints require model_type='i2v'")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.video_attention_mask_mode not in {
            "first_frame_causal",
            "per_frame_causal",
            "bidirectional",
            "three_frame",
        }:
            raise ValueError("Unsupported video attention mask mode")

    @property
    def attn_hidden_dim(self) -> int:
        return self.num_heads * self.attn_head_dim

    def validate_tp_size(self, tp_size: int) -> None:
        if tp_size <= 0 or self.num_heads % tp_size or self.ffn_dim % tp_size:
            raise ValueError("TP size must divide the number of heads and FFN width")


@dataclass(frozen=True)
class VPP2Config(PretrainedConfig):
    video: VPP2VideoConfig = field(default_factory=VPP2VideoConfig)
    action: VPP2ActionConfig = field(default_factory=VPP2ActionConfig)
    proprio_dim: int | None = 16
    action_chunk_size: int = 1
    action_visible_video_frames: int = 5
    action_only_first_frame_fast_path: bool = False
    video_num_train_timesteps: int = 1000
    action_num_train_timesteps: int = 1000
    video_infer_shift: float = 3.0
    action_infer_shift: float = 3.0
    vae_temporal_downsample_factor: int = 4

    nested_sources: ClassVar[Mapping[str, str | tuple[str, ...]]] = {
        "video": ("video_dit_config", "video_config", "video_dit"),
        "action": ("action_dit_config", "action_config", "action_dit"),
        "video_num_train_timesteps": "video_scheduler.num_train_timesteps",
        "action_num_train_timesteps": "action_scheduler.num_train_timesteps",
        "video_infer_shift": "video_scheduler.infer_shift",
        "action_infer_shift": "action_scheduler.infer_shift",
    }

    @classmethod
    def from_checkpoint_metadata(cls, data: dict[str, Any]) -> VPP2Config:
        resolved = data.get("resolved_config", data)
        if not isinstance(resolved, dict):
            raise ValueError("Checkpoint resolved_config must be a mapping")
        model = resolved.get("model", resolved)
        if not isinstance(model, dict):
            raise ValueError("Checkpoint model configuration must be a mapping")
        for name in ("use_video_tokens", "joint_denoising"):
            if model.get(name, False):
                raise ValueError(f"Unsupported experimental checkpoint option: {name}")
        for name in (
            "use_fixed_video_layers",
            "action_clean_first_frame_start_layer",
            "action_clean_first_frame_timestep_cutoff",
            "action_video_timestep_cutoff",
        ):
            if model.get(name) is not None:
                raise ValueError(f"Unsupported experimental checkpoint option: {name}")
        return cls.from_dict(model)

    def __post_init__(self) -> None:
        if (self.video.num_heads, self.video.attn_head_dim, self.video.num_layers) != (
            self.action.num_heads,
            self.action.attn_head_dim,
            self.action.num_layers,
        ):
            raise ValueError("Video and action experts must share heads and depth")
        if self.video.text_dim != self.action.text_dim:
            raise ValueError("Video and action text dimensions must match")
        if (
            self.action_chunk_size <= 0
            or self.action.action_dim % self.action_chunk_size
        ):
            raise ValueError("action_chunk_size must divide the action dimension")
        if self.action_visible_video_frames <= 0:
            raise ValueError("action_visible_video_frames must be positive")
        if self.proprio_dim is not None and self.proprio_dim <= 0:
            raise ValueError("proprio_dim must be positive or None")
        if min(self.video_num_train_timesteps, self.action_num_train_timesteps) <= 0:
            raise ValueError("Training timestep counts must be positive")
        if min(self.video_infer_shift, self.action_infer_shift) <= 0:
            raise ValueError("Inference shifts must be positive")
        if self.vae_temporal_downsample_factor <= 0:
            raise ValueError("VAE temporal downsampling factor must be positive")
