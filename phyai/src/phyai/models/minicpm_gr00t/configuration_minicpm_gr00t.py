"""Configuration for the MiniCPM-V 4.6 GR00T policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class MiniCPMGR00TVisionConfig(PretrainedConfig):
    """SigLIP vision tower and the two MiniCPM-V merge stages."""

    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 980
    patch_size: int = 14
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    insert_layer_id: int = 6
    window_size: int = 2
    resampler_size: int = 2
    output_hidden_size: int = 1024

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by "
                f"patch_size={self.patch_size}."
            )
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"num_attention_heads={self.num_attention_heads}."
            )
        if not 0 <= self.insert_layer_id < self.num_hidden_layers:
            raise ValueError(
                f"insert_layer_id={self.insert_layer_id} must be in "
                f"[0, {self.num_hidden_layers})."
            )
        if self.window_size != 2 or self.resampler_size != 2:
            raise ValueError("The checkpoint requires 2x2 window and resampler merges.")
        if self.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(
                "The SigLIP checkpoint requires hidden_act='gelu_pytorch_tanh'."
            )
        if self.attention_dropout != 0.0:
            raise ValueError("Vision attention_dropout must be 0.0 for inference.")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_position_embeddings(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def merged_hidden_size(self) -> int:
        return self.hidden_size * self.window_size**2

    @property
    def merger_intermediate_size(self) -> int:
        return self.intermediate_size * self.window_size**2


@dataclass(frozen=True)
class MiniCPMGR00TTextConfig(PretrainedConfig):
    """Qwen3.5 hybrid text decoder used by MiniCPM-V 4.6."""

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

    vocab_size: int = 248094
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
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be divisible "
                f"by num_key_value_heads={self.num_key_value_heads}."
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"len(layer_types)={len(self.layer_types)} must equal "
                f"num_hidden_layers={self.num_hidden_layers}."
            )
        valid_types = {"linear_attention", "full_attention"}
        if not set(self.layer_types) <= valid_types:
            raise ValueError(
                f"layer_types must contain only {sorted(valid_types)}, got "
                f"{self.layer_types}."
            )
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError(f"rotary_dim={rotary_dim} must be positive and even.")
        if sum(self.mrope_section) != rotary_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(self.mrope_section)} must equal "
                f"rotary_dim//2={rotary_dim // 2}."
            )
        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError("PHYAI GatedDeltaNet requires equal key/value head dims.")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                "linear_num_value_heads must be divisible by linear_num_key_heads."
            )
        if self.hidden_act != "silu":
            raise ValueError("Qwen3.5 text requires hidden_act='silu'.")
        if not self.attn_output_gate:
            raise ValueError("Qwen3.5 full attention requires attn_output_gate=True.")


@dataclass(frozen=True)
class MiniCPMGR00TDiTConfig(PretrainedConfig):
    """FP32 DiT-B action transformer."""

    hidden_size: int = 768
    num_attention_heads: int = 12
    attention_head_dim: int = 64
    num_hidden_layers: int = 16
    cross_attention_dim: int = 1024
    intermediate_size: int = 3072
    output_dim: int = 1024
    timestep_input_dim: int = 256
    num_timestep_buckets: int = 1000
    norm_eps: float = 1e-5
    output_norm_eps: float = 1e-6
    dropout: float = 0.2
    final_dropout: bool = True
    activation: str = "gelu-approximate"
    interleave_self_attention: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.attention_head_dim:
            raise ValueError(
                f"hidden_size={self.hidden_size} must equal num_attention_heads * "
                f"attention_head_dim={self.num_attention_heads * self.attention_head_dim}."
            )
        if self.attention_head_dim % 2:
            raise ValueError("attention_head_dim must be even.")
        if self.num_hidden_layers <= 0 or self.num_hidden_layers % 2:
            raise ValueError("num_hidden_layers must be a positive even number.")
        if self.intermediate_size != self.hidden_size * 4:
            raise ValueError("The checkpoint requires a 4x DiT FFN.")
        if self.activation != "gelu-approximate":
            raise ValueError("The checkpoint requires activation='gelu-approximate'.")
        if not self.interleave_self_attention:
            raise ValueError("The checkpoint requires interleaved self-attention.")
        if self.num_timestep_buckets != 1000:
            raise ValueError("The checkpoint requires 1000 timestep buckets.")


@dataclass(frozen=True)
class MiniCPMGR00TActionConfig(PretrainedConfig):
    """Action/proprio boundary layers and clean-action policy dimensions."""

    dit: MiniCPMGR00TDiTConfig = field(default_factory=MiniCPMGR00TDiTConfig)
    action_dim: int = 80
    proprio_dim: int = 80
    action_horizon: int = 30
    num_future_tokens: int = 32
    max_position_embeddings: int = 1024
    num_inference_steps: int = 4
    max_num_embodiments: int = 1
    proprio_inject: str = "concat"
    prediction_type: str = "clean_action"

    def __post_init__(self) -> None:
        if self.action_dim <= 0 or self.proprio_dim <= 0:
            raise ValueError("action_dim and proprio_dim must be positive.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if self.num_future_tokens <= 0:
            raise ValueError("num_future_tokens must be positive.")
        if self.max_position_embeddings < self.action_horizon:
            raise ValueError(
                "max_position_embeddings must cover the complete action horizon."
            )
        if self.num_inference_steps != 4:
            raise ValueError("The checkpoint was trained for four clean-action steps.")
        if self.max_num_embodiments != 1:
            raise ValueError("This checkpoint contains a single shared embodiment.")
        if self.proprio_inject != "concat":
            raise ValueError("The checkpoint requires proprio_inject='concat'.")
        if self.prediction_type != "clean_action":
            raise ValueError("The checkpoint requires prediction_type='clean_action'.")

    @property
    def action_encoder_input_dim(self) -> int:
        return self.action_dim + self.proprio_dim

    @property
    def dit_sequence_length(self) -> int:
        return self.num_future_tokens + self.action_horizon


@dataclass(frozen=True)
class MiniCPMGR00TConfig(PretrainedConfig):
    """Top-level MiniCPM-V GR00T model configuration."""

    nested_sources = {"vision": "vision_config", "text": "text_config"}

    vision: MiniCPMGR00TVisionConfig = field(default_factory=MiniCPMGR00TVisionConfig)
    text: MiniCPMGR00TTextConfig = field(default_factory=MiniCPMGR00TTextConfig)
    action: MiniCPMGR00TActionConfig = field(default_factory=MiniCPMGR00TActionConfig)
    insert_layer_id: int = 6
    image_token_id: int = 248056
    video_token_id: int = 248057
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.insert_layer_id != self.vision.insert_layer_id:
            raise ValueError(
                f"insert_layer_id={self.insert_layer_id} must equal "
                f"vision.insert_layer_id={self.vision.insert_layer_id}."
            )
        if self.vision.output_hidden_size != self.text.hidden_size:
            raise ValueError(
                f"vision.output_hidden_size={self.vision.output_hidden_size} must "
                f"equal text.hidden_size={self.text.hidden_size}."
            )
        if self.action.dit.cross_attention_dim != self.text.hidden_size:
            raise ValueError(
                f"action.dit.cross_attention_dim="
                f"{self.action.dit.cross_attention_dim} must equal "
                f"text.hidden_size={self.text.hidden_size}."
            )
        if self.tie_word_embeddings != self.text.tie_word_embeddings:
            raise ValueError("Top-level and text tie_word_embeddings must agree.")


__all__ = [
    "MiniCPMGR00TActionConfig",
    "MiniCPMGR00TConfig",
    "MiniCPMGR00TDiTConfig",
    "MiniCPMGR00TTextConfig",
    "MiniCPMGR00TVisionConfig",
]
