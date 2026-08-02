"""Configs for pi0.

The shape mirrors :mod:`phyai.models.pi05.configuration_pi05`, but the
top-level composition follows pi0:

* :class:`SiglipVisionConfig` -- PaliGemma vision tower.
* :class:`PaliGemmaTextConfig` -- gemma_2b text side.
* :class:`GemmaExpertConfig` -- gemma_300m action expert without AdaRMS.
* :class:`PI0Config` -- full model geometry plus flow-matching knobs.

Unlike pi0.5, pi0 keeps robot state as a numeric expert-side token:
``state_proj`` maps ``max_state_dim`` to the expert hidden width. The
suffix block is therefore ``[state_token, action_tokens...]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class SiglipVisionConfig(PretrainedConfig):
    """Static config for the PaliGemma/SigLIP vision tower used by pi0."""

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    projection_dim: int = 2048

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} not divisible by "
                f"patch_size={self.patch_size}."
            )
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} not divisible by "
                f"num_attention_heads={self.num_attention_heads}."
            )

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True)
class PaliGemmaTextConfig(PretrainedConfig):
    """Config for the PaliGemma language model -- gemma_2b text side."""

    hidden_size: int = 2048
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 16384
    vocab_size: int = 257152
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads}."
            )
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be a positive even int.")

    @property
    def joint_attention_dim(self) -> int:
        """Total dim of the joint attention space (heads * head_dim)."""

        return self.num_attention_heads * self.head_dim


@dataclass(frozen=True)
class GemmaExpertConfig(PretrainedConfig):
    """Config for the pi0 gemma_300m action expert.

    pi0 does not use AdaRMS conditioning in the expert. Timestep is
    fused into each action token by ``action_time_mlp_in/out`` before
    entering the expert stack.
    """

    hidden_size: int = 1024
    num_hidden_layers: int = 18
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    use_adarms: bool = False
    adarms_cond_dim: int | None = None

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads}."
            )
        if self.head_dim <= 0 or self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be a positive even int.")
        if self.use_adarms and self.adarms_cond_dim != self.hidden_size:
            raise ValueError(
                f"adarms_cond_dim={self.adarms_cond_dim} must equal "
                f"hidden_size={self.hidden_size} when use_adarms=True."
            )

    @property
    def joint_attention_dim(self) -> int:
        return self.num_attention_heads * self.head_dim


@dataclass(frozen=True)
class PI0Config(PretrainedConfig):
    """Top-level pi0 config: vision + text + expert + flow matching.

    Defaults follow the LeRobot/OpenPI pi0 PyTorch implementation:
    PaliGemma gemma_2b for image/language, gemma_300m for the action
    expert, a 32-D padded robot state vector, and 50-step action chunks.
    """

    vision: SiglipVisionConfig = field(default_factory=SiglipVisionConfig)
    text: PaliGemmaTextConfig = field(default_factory=PaliGemmaTextConfig)
    expert: GemmaExpertConfig = field(default_factory=GemmaExpertConfig)

    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0
    tokenizer_max_length: int = 48
    empty_cameras: int = 0

    def __post_init__(self) -> None:
        if self.vision.projection_dim != self.text.hidden_size:
            raise ValueError(
                f"vision.projection_dim={self.vision.projection_dim} must equal "
                f"text.hidden_size={self.text.hidden_size} so the "
                f"multi_modal_projector output lands in the LM embedding space."
            )
        if self.text.joint_attention_dim != self.expert.joint_attention_dim:
            raise ValueError(
                f"text.joint_attention_dim={self.text.joint_attention_dim} must "
                f"equal expert.joint_attention_dim={self.expert.joint_attention_dim}; "
                f"both streams write into the same joint attention output space."
            )
        if self.text.head_dim != self.expert.head_dim:
            raise ValueError(
                f"text.head_dim={self.text.head_dim} must equal "
                f"expert.head_dim={self.expert.head_dim} (joint attention)."
            )
        if self.text.num_key_value_heads != self.expert.num_key_value_heads:
            raise ValueError(
                f"text.num_key_value_heads={self.text.num_key_value_heads} must "
                f"equal expert.num_key_value_heads="
                f"{self.expert.num_key_value_heads} (joint K/V layout)."
            )
        if self.text.num_hidden_layers != self.expert.num_hidden_layers:
            raise ValueError(
                f"text.num_hidden_layers={self.text.num_hidden_layers} must equal "
                f"expert.num_hidden_layers={self.expert.num_hidden_layers}; "
                f"joint decoder pairs one text layer with one expert layer."
            )
        if self.expert.use_adarms:
            raise ValueError("pi0 expert must use use_adarms=False.")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")
        if self.max_state_dim <= 0:
            raise ValueError(
                f"max_state_dim must be positive, got {self.max_state_dim}."
            )
        if self.max_action_dim <= 0:
            raise ValueError(
                f"max_action_dim must be positive, got {self.max_action_dim}."
            )
        if self.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {self.num_inference_steps}."
            )
        if self.tokenizer_max_length <= 0:
            raise ValueError(
                f"tokenizer_max_length must be positive, got "
                f"{self.tokenizer_max_length}."
            )
        if self.empty_cameras not in (0, 1):
            raise ValueError(
                f"empty_cameras must be 0 or 1 for pi0, got {self.empty_cameras}."
            )

    @property
    def num_layers(self) -> int:
        """Layer count for the joint stack -- text and expert share it."""

        return self.text.num_hidden_layers

    @property
    def suffix_len(self) -> int:
        """Number of expert-side suffix tokens: state token + action chunk."""

        return 1 + self.chunk_size

    @property
    def num_images(self) -> int:
        """Number of real camera streams consumed by this pi0 checkpoint."""

        return 3 - self.empty_cameras


__all__ = [
    "GemmaExpertConfig",
    "PaliGemmaTextConfig",
    "PI0Config",
    "SiglipVisionConfig",
]
