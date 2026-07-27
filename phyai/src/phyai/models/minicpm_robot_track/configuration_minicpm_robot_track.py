"""Configuration for MiniCPM-RobotTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from phyai.models.configuration import PretrainedConfig

_MINICPM4_LONGROPE_FACTORS = (
    1.0004360675811768,
    1.0668443441390991,
    1.1631425619125366,
    1.3025742769241333,
    1.5040205717086792,
    1.7941505908966064,
    2.2101221084594727,
    2.802666664123535,
    3.6389970779418945,
    4.804192543029785,
    6.39855432510376,
    8.527148246765137,
    11.277542114257812,
    14.684998512268066,
    18.69317054748535,
    23.13019371032715,
    27.72362518310547,
    32.1606559753418,
    36.168827056884766,
    39.57627868652344,
    42.32667541503906,
    44.45526885986328,
    46.04962921142578,
    47.21482849121094,
    48.05115509033203,
    48.64370346069336,
    49.05967712402344,
    49.34980392456055,
    49.551246643066406,
    49.69068145751953,
    49.78697967529297,
    49.85338592529297,
)


@dataclass(frozen=True)
class MiniCPM4TrackConfig(PretrainedConfig):
    """MiniCPM4-0.5B backbone fields used by RobotTrack."""

    nested_sources: ClassVar[dict[str, str]] = {
        "rope_type": "rope_scaling.rope_type",
        "long_factor": "rope_scaling.long_factor",
        "short_factor": "rope_scaling.short_factor",
        "original_max_position_embeddings": (
            "rope_scaling.original_max_position_embeddings"
        ),
    }

    vocab_size: int = 73448
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_type: str = "longrope"
    short_factor: tuple[float, ...] = _MINICPM4_LONGROPE_FACTORS
    long_factor: tuple[float, ...] = _MINICPM4_LONGROPE_FACTORS
    original_max_position_embeddings: int = 32768
    scale_depth: float = 1.4

    def __post_init__(self) -> None:
        if not isinstance(self.short_factor, tuple):
            object.__setattr__(self, "short_factor", tuple(self.short_factor))
        if not isinstance(self.long_factor, tuple):
            object.__setattr__(self, "long_factor", tuple(self.long_factor))
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} must be divisible by "
                f"num_attention_heads={self.num_attention_heads}."
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be divisible "
                f"by num_key_value_heads={self.num_key_value_heads}."
            )
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be even.")
        if self.rope_type != "longrope":
            raise ValueError(
                f"MiniCPM-RobotTrack requires rope_type='longrope', got "
                f"{self.rope_type!r}."
            )
        expected_factors = self.head_dim // 2
        if len(self.short_factor) != expected_factors:
            raise ValueError(
                f"short_factor must contain {expected_factors} values, got "
                f"{len(self.short_factor)}."
            )
        if len(self.long_factor) != expected_factors:
            raise ValueError(
                f"long_factor must contain {expected_factors} values, got "
                f"{len(self.long_factor)}."
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True)
class MiniCPMRobotTrackConfig(PretrainedConfig):
    """Static-shape RobotTrack policy configuration."""

    nested_sources: ClassVar[dict[str, str]] = {"backbone": "backbone_config"}

    backbone: MiniCPM4TrackConfig = field(default_factory=MiniCPM4TrackConfig)
    vision_feature_dim: int = 1536
    history_frames: int = 31
    coarse_tokens_per_frame: int = 4
    fine_tokens_current_frame: int = 64
    num_waypoints: int = 8
    action_dim: int = 3
    max_text_tokens: int = 128
    max_time_steps: int = 4096
    trajectory_dropout: float = 0.4
    xy_scale: float = 2.0
    use_tanh_actions: bool = True
    input_seq_length: int = 256

    def __post_init__(self) -> None:
        if self.vision_feature_dim <= 0:
            raise ValueError("vision_feature_dim must be positive.")
        if self.history_frames <= 0:
            raise ValueError("history_frames must be positive.")
        for name, value in (
            ("coarse_tokens_per_frame", self.coarse_tokens_per_frame),
            ("fine_tokens_current_frame", self.fine_tokens_current_frame),
        ):
            side = round(value**0.5) if value > 0 else 0
            if side * side != value:
                raise ValueError(f"{name} must be a positive square number.")
        if self.num_waypoints < 2:
            raise ValueError("num_waypoints must be at least 2.")
        if self.action_dim != 3:
            raise ValueError("RobotTrack actions must use [x, y, yaw] format.")
        if self.max_text_tokens <= 0 or self.max_time_steps <= 0:
            raise ValueError("text and time limits must be positive.")
        if not 0.0 <= self.trajectory_dropout < 1.0:
            raise ValueError("trajectory_dropout must be in [0, 1).")
        if self.xy_scale <= 0.0:
            raise ValueError("xy_scale must be positive.")
        if self.text_capacity <= 0:
            raise ValueError(
                f"input_seq_length={self.input_seq_length} is too short for "
                f"{self.fixed_non_text_tokens} fixed non-text tokens."
            )
        if self.input_seq_length > self.backbone.max_position_embeddings:
            raise ValueError("input_seq_length exceeds the MiniCPM4 position limit.")

    @property
    def coarse_token_count(self) -> int:
        return self.history_frames * self.coarse_tokens_per_frame

    @property
    def history_sequence_tokens(self) -> int:
        return self.history_frames * (self.coarse_tokens_per_frame + 1)

    @property
    def current_sequence_tokens(self) -> int:
        return self.fine_tokens_current_frame + 1

    @property
    def fixed_non_text_tokens(self) -> int:
        return self.history_sequence_tokens + self.current_sequence_tokens + 1

    @property
    def text_capacity(self) -> int:
        return self.input_seq_length - self.fixed_non_text_tokens


__all__ = ["MiniCPM4TrackConfig", "MiniCPMRobotTrackConfig"]
