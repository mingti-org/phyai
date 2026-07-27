"""Configs for Qwen3.5."""

from __future__ import annotations

from dataclasses import dataclass, field

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class Qwen3_5VisionConfig(PretrainedConfig):
    """Config for the Qwen3.5 vision tower."""

    depth: int = 12
    hidden_size: int = 768
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 3072
    num_heads: int = 12
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 1024
    num_position_embeddings: int = 2304
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} not divisible by "
                f"num_heads={self.num_heads}."
            )
        if self.head_dim % 2:
            raise ValueError(f"vision head_dim={self.head_dim} must be even.")
        side = int(self.num_position_embeddings**0.5)
        if side * side != self.num_position_embeddings:
            raise ValueError(
                f"num_position_embeddings={self.num_position_embeddings} must be "
                "a perfect square."
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        return int(self.num_position_embeddings**0.5)


@dataclass(frozen=True)
class Qwen3_5TextConfig(PretrainedConfig):
    """Config for the Qwen3.5 hybrid text decoder."""

    nested_sources = {
        "mrope_section": (
            "rope_parameters.mrope_section",
            "rope_scaling.mrope_section",
        ),
        "partial_rotary_factor": (
            "rope_parameters.partial_rotary_factor",
            "rope_scaling.partial_rotary_factor",
        ),
        "rope_theta": (
            "rope_parameters.rope_theta",
            "rope_scaling.rope_theta",
        ),
        "rope_type": (
            "rope_parameters.rope_type",
            "rope_scaling.rope_type",
        ),
    }

    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    rope_type: str = "default"
    partial_rotary_factor: float = 0.25
    mrope_section: tuple[int, ...] = (11, 11, 10)
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True
    tie_word_embeddings: bool = True
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    layer_types: tuple[str, ...] = (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ) * 6

    def __post_init__(self) -> None:
        if not isinstance(self.mrope_section, tuple):
            object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
        if not isinstance(self.layer_types, tuple):
            object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads}."
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"len(layer_types)={len(self.layer_types)} must equal "
                f"num_hidden_layers={self.num_hidden_layers}."
            )
        valid_layer_types = {"linear_attention", "full_attention"}
        if not set(self.layer_types) <= valid_layer_types:
            raise ValueError(
                f"layer_types must contain only {sorted(valid_layer_types)}, got "
                f"{self.layer_types}."
            )
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(f"rotary_dim={rotary_dim} must be a positive even int.")
        if sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(self.mrope_section)} must equal "
                f"rotary_dim//2={rotary_dim // 2}."
            )
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError(
                "Qwen3.5 GDN requires equal key and value head dimensions in PHYAI."
            )
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError(
                f"linear_num_value_heads={self.linear_num_value_heads} must be a "
                f"multiple of linear_num_key_heads={self.linear_num_key_heads}."
            )
        if not self.attn_output_gate:
            raise ValueError("Qwen3.5 full attention requires attn_output_gate=True.")


@dataclass(frozen=True)
class Qwen3_5Config(PretrainedConfig):
    """Top-level Qwen3.5 multimodal config."""

    nested_sources = {"vision": "vision_config", "text": "text_config"}

    vision: Qwen3_5VisionConfig = field(default_factory=Qwen3_5VisionConfig)
    text: Qwen3_5TextConfig = field(default_factory=Qwen3_5TextConfig)
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vision.out_hidden_size != self.text.hidden_size:
            raise ValueError(
                f"vision.out_hidden_size={self.vision.out_hidden_size} must equal "
                f"text.hidden_size={self.text.hidden_size}."
            )


__all__ = ["Qwen3_5Config", "Qwen3_5TextConfig", "Qwen3_5VisionConfig"]
