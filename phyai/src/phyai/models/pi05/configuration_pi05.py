"""Configs for pi0.5.

Three layered dataclasses, all :class:`~phyai.models.configuration.PretrainedConfig`
subclasses (frozen, JSON-loadable, mapping-style read access):

* :class:`SiglipVisionConfig` — vision tower (SigLIP-So400m).
* :class:`PaliGemmaTextConfig` — language model (gemma_2b text side).
* :class:`GemmaExpertConfig` — action expert (gemma_300m, AdaRMS).
* :class:`PI05Config` — top-level composition + flow-matching knobs.

Defaults across all four match the public ``pi05_base`` checkpoint at
https://huggingface.co/lerobot/pi05_base.

Loaded from a ``config.json`` via :meth:`PretrainedConfig.from_json`;
unknown keys are silently dropped. Nested sub-configs are NOT auto-built
from a flat top-level policy JSON — the caller constructs them
explicitly when overriding from defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class SiglipVisionConfig(PretrainedConfig):
    """Static config for the SigLIP-So400m vision tower used by pi0.5.

    All defaults match the ``vision_tower.vision_model`` keys of the
    public pi0.5 / paligemma-3b checkpoints. Callers building a
    different SigLIP variant (e.g. So400m@384, base@224) override the
    geometry knobs and leave the rest at defaults.
    """

    hidden_size: int = 1152
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    intermediate_size: int = 4304
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    # Output dim of the multi_modal_projector; equals the text-side
    # hidden size (gemma_2b: 2048, gemma_300m: 1024).
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
    """Config for the PaliGemma language model — gemma_2b text side.

    Defaults match the ``language_model.*`` keys of the pi0.5 base
    checkpoint. PaliGemma's text tower is decoder-only Gemma with MQA
    (``num_key_value_heads=1``); attention is **non-causal** (prefix-LM)
    in pi0.5's usage but the config does not encode that — it lives in
    the modeling code.

    The action expert uses :class:`GemmaExpertConfig` (gemma_300m) and
    must share ``num_attention_heads * head_dim`` with this config so
    the joint attention space (8 * 256 = 2048) lines up.
    """

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
        """Total dim of the joint attention space (heads * head_dim).

        Both the text tower and the action expert project Q to this dim
        and project K/V to ``num_key_value_heads * head_dim``; the two
        streams must agree on this number for joint attention.
        """
        return self.num_attention_heads * self.head_dim


@dataclass(frozen=True)
class GemmaExpertConfig(PretrainedConfig):
    """Config for the gemma_300m action expert with AdaRMS conditioning.

    Defaults match the ``gemma_expert.*`` keys of the pi0.5 base
    checkpoint. AdaRMS is on by default (``use_adarms=True``); the
    timestep embedding (``time_mlp`` output, dim 1024) feeds AdaRMS as
    the conditioning signal, so ``adarms_cond_dim == hidden_size``.

    The expert has no ``embed_tokens`` (the upstream pi0.5 sets it to
    ``None``); the input to the expert is the action chunk projected by
    ``action_in_proj`` (1024-D), not vocab tokens.
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
    use_adarms: bool = True
    adarms_cond_dim: int = 1024

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
                f"hidden_size={self.hidden_size} when use_adarms=True "
                f"(time_mlp produces hidden_size-dim output that feeds AdaRMS)."
            )

    @property
    def joint_attention_dim(self) -> int:
        return self.num_attention_heads * self.head_dim


@dataclass(frozen=True)
class PI05RecommendedEngineConfig(PretrainedConfig):
    """pi0.5's recommended engine runtime tunables for *this* repo's kernels.

    A model ships the engine settings it was tuned against instead of
    baking them into the shared backends. :meth:`PI05Entry.setup` reads
    this and installs it on the :class:`EngineConfig` singleton **before**
    building the model — but only where the user hasn't already pinned a
    value, so an explicit ``EngineConfig`` / ``PHYAI_*`` override always
    wins. Field names mirror
    :class:`~phyai.engine_config.RuntimeConfig` so the apply step is a
    straight key copy (no engine_config import here — this stays a config
    leaf).

    Fields
    ------
    flashinfer_prefill_backend:
        ``"fa2"``: the action-expert joint attention is short-query
        (chunk 50) against a long cached prefix at head_dim 256; FA2 is
        ~2.5x faster than the ``"auto"``-selected FA3 there under CUDA
        graph replay (numerically equal to bf16 ulp level). ``None`` would
        defer to flashinfer's auto heuristic.
    flashinfer_workspace_bytes:
        FA2's split-tmp scratch for this shape needs ~132 MiB, just over
        the engine default 128 MiB; 256 MiB covers it. Only applied (as a
        floor) when the effective prefill backend is FA2.
    """

    flashinfer_prefill_backend: str | None = "fa2"
    flashinfer_workspace_bytes: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class PI05Config(PretrainedConfig):
    """Top-level pi0.5 inference config: vision + text + expert + flow-matching.

    Defaults match ``pi05_base`` end-to-end. ``vision.projection_dim``
    must equal ``text.hidden_size`` (the multi_modal_projector lifts
    SigLIP's tokens into the LM embedding space), and ``text`` and
    ``expert`` must share ``joint_attention_dim`` (the joint attention
    output space the two streams write into).

    Flow-matching knobs:

    * ``chunk_size``: number of action tokens in one suffix block (50).
    * ``max_action_dim``: action vector width before action_in_proj (32).
    * ``num_inference_steps``: Euler steps in :func:`sample_actions` (10).
    * ``min_period`` / ``max_period``: frequency span of the sinusoidal
      time embedding (4e-3 / 4.0).
    * ``tokenizer_max_length``: language token padding budget (200).
    """

    vision: SiglipVisionConfig = field(default_factory=SiglipVisionConfig)
    text: PaliGemmaTextConfig = field(default_factory=PaliGemmaTextConfig)
    expert: GemmaExpertConfig = field(default_factory=GemmaExpertConfig)

    chunk_size: int = 50
    max_action_dim: int = 32
    num_inference_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0
    tokenizer_max_length: int = 200

    # RL critic. ``value_after_vlm`` selects the prefix or suffix residual
    # stream. ``action_chunk`` only affects suffix pooling when
    # ``chunk_critic_input=True``; ``None`` means the full action horizon.
    config_name: str = "pi05_base"
    action_chunk: int | None = None
    detach_critic_input: bool = False
    chunk_critic_input: bool = False
    add_value_head: bool = False
    value_after_vlm: bool = False
    value_vlm_mode: Literal["mean_token"] = "mean_token"

    # Engine runtime knobs pi0.5 was tuned against; applied by
    # PI05Entry.setup (user/env overrides still win). Not part of the
    # upstream checkpoint config.json — defaults are used when absent.
    recommended_engine: PI05RecommendedEngineConfig = field(
        default_factory=PI05RecommendedEngineConfig
    )

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
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}.")
        if self.num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {self.num_inference_steps}."
            )
        if (
            self.action_chunk is not None
            and not 1 <= self.action_chunk <= self.chunk_size
        ):
            raise ValueError(
                f"action_chunk must be in [1, {self.chunk_size}], got "
                f"{self.action_chunk}."
            )
        if self.value_vlm_mode != "mean_token":
            raise ValueError(
                f"value_vlm_mode must be 'mean_token', got {self.value_vlm_mode!r}."
            )

    @property
    def num_layers(self) -> int:
        """Layer count for the joint stack — text and expert share it."""
        return self.text.num_hidden_layers

    @property
    def critic_action_chunk(self) -> int:
        """Number of suffix tokens used by chunk-restricted critic pooling."""
        return self.action_chunk if self.action_chunk is not None else self.chunk_size

    @property
    def value_head_hidden_sizes(self) -> tuple[int, int, int]:
        """RLinf-compatible hidden widths for this pi0.5 configuration."""
        if self.config_name in {
            "pi05_maniskill",
            "pi05_libero",
            "pi05_droid_polaris",
        }:
            return (1024, 512, 256)
        return (512, 256, 128)

    @property
    def value_head_input_dim(self) -> int:
        """Residual-stream width consumed by the configured critic."""
        if self.value_after_vlm:
            return self.text.hidden_size
        return self.expert.hidden_size


__all__ = [
    "SiglipVisionConfig",
    "PaliGemmaTextConfig",
    "GemmaExpertConfig",
    "PI05RecommendedEngineConfig",
    "PI05Config",
]
