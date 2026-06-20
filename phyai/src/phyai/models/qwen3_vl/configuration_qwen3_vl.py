"""Configs for Qwen3-VL."""

from __future__ import annotations

from dataclasses import dataclass, field

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class Qwen3VLVisionConfig(PretrainedConfig):
    """Config for the Qwen3-VL vision tower.

    Defaults match the ``vision_config`` of the public Qwen3-VL checkpoints.
    """

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        # From JSON, deepstack_visual_indexes arrives as a list; coerce so the
        # frozen config stays hashable and equality with the tuple default holds.
        if not isinstance(self.deepstack_visual_indexes, tuple):
            object.__setattr__(
                self, "deepstack_visual_indexes", tuple(self.deepstack_visual_indexes)
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} not divisible by "
                f"num_heads={self.num_heads}."
            )
        if self.head_dim % 2:
            raise ValueError(
                f"vision head_dim={self.head_dim} must be even for axial 2-D RoPE."
            )
        side = int(self.num_position_embeddings**0.5)
        if side * side != self.num_position_embeddings:
            raise ValueError(
                f"num_position_embeddings={self.num_position_embeddings} must be a "
                f"perfect square (the learned pos-embed table is a square grid)."
            )
        if any(i < 0 or i >= self.depth for i in self.deepstack_visual_indexes):
            raise ValueError(
                f"deepstack_visual_indexes={self.deepstack_visual_indexes} must all "
                f"be in [0, depth={self.depth})."
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        """Side length of the square learned position-embedding grid."""
        return int(self.num_position_embeddings**0.5)

    @property
    def spatial_merge_unit(self) -> int:
        """Patches collapsed into one merged token (``spatial_merge_size**2``)."""
        return self.spatial_merge_size**2


@dataclass(frozen=True)
class Qwen3VLTextConfig(PretrainedConfig):
    """Config for the Qwen3-VL language model"""

    nested_sources = {
        "mrope_section": (
            "rope_scaling.mrope_section",
            "rope_parameters.mrope_section",
        ),
        "rope_theta": ("rope_scaling.rope_theta", "rope_parameters.rope_theta"),
    }

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)
    max_position_embeddings: int = 128000
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        # Lifted from JSON, mrope_section arrives as a list; coerce so the frozen
        # config stays hashable and equality with the tuple default holds.
        if not isinstance(self.mrope_section, tuple):
            object.__setattr__(self, "mrope_section", tuple(self.mrope_section))
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads}."
            )
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be a positive even int.")
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(self.mrope_section)} must equal "
                f"head_dim//2={self.head_dim // 2}; the temporal/height/width axes "
                f"partition the half-rotary frequency slots."
            )

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass(frozen=True)
class Qwen3VLConfig(PretrainedConfig):
    """Top-level Qwen3-VL config: vision tower + text decoder + special token ids."""

    nested_sources = {"vision": "vision_config", "text": "text_config"}

    vision: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    text: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.vision.out_hidden_size != self.text.hidden_size:
            raise ValueError(
                f"vision.out_hidden_size={self.vision.out_hidden_size} must equal "
                f"text.hidden_size={self.text.hidden_size} so merged vision tokens "
                f"and deepstack features land in the LM embedding space."
            )

    @property
    def image_token_index(self) -> int:
        """Alias kept for parity with HF naming."""
        return self.image_token_id


__all__ = [
    "Qwen3VLVisionConfig",
    "Qwen3VLTextConfig",
    "Qwen3VLConfig",
]
