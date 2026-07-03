"""Configs for GR00T-N1.7.

The public Isaac-GR00T N1.7 checkpoint stores a mostly flat
``config.json``. PhyAI keeps the same defaults but groups them by the
runtime boundary we implement:

* processor: observation/action interpretation and preprocessing knobs;
* backbone: Cosmos-Reason2 / Qwen3-VL feature extractor knobs;
* action head: state/action encoders, optional VL self-attention, and DiT.

Checkpoint-specific choices such as LIBERO vs DROID do not live in the
modeling classes. They are selected by ``embodiment_tag`` and checkpoint
processor metadata at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phyai.models.configuration import PretrainedConfig
from phyai.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)


def _default_gr00t_qwen3vl_config() -> Qwen3VLConfig:
    """Native Cosmos-Reason2/Qwen3-VL config used by GR00T-N1.7.

    These values mirror the ``nvidia/Cosmos-Reason2-2B`` config referenced by
    the public GR00T-N1.7 checkpoint, so the native backbone can be built
    without resolving the HuggingFace repo at model-load time.
    """
    return Qwen3VLConfig(
        vision=Qwen3VLVisionConfig(
            depth=24,
            hidden_size=1024,
            hidden_act="gelu_pytorch_tanh",
            intermediate_size=4096,
            num_heads=16,
            in_channels=3,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=2048,
            num_position_embeddings=2304,
            deepstack_visual_indexes=(5, 11, 17),
            initializer_range=0.02,
        ),
        text=Qwen3VLTextConfig(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=5000000.0,
            mrope_section=(24, 20, 20),
            max_position_embeddings=262144,
            attention_bias=False,
            tie_word_embeddings=True,
        ),
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=True,
    )


def _tuple2(v: object) -> tuple[int, int] | None:
    if v is None:
        return None
    if isinstance(v, tuple) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    if isinstance(v, list) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ValueError(f"expected a 2-item tuple/list or None, got {v!r}.")


def _tuple_ints(v: object) -> tuple[int, ...]:
    if isinstance(v, tuple):
        return tuple(int(x) for x in v)
    if isinstance(v, list):
        return tuple(int(x) for x in v)
    raise ValueError(f"expected a tuple/list of ints, got {v!r}.")


@dataclass(frozen=True)
class GR00TN17ProcessorConfig(PretrainedConfig):
    """Processor and state/action normalization knobs."""

    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)
    shortest_image_edge: int | None = None
    crop_fraction: float | None = None
    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    extra_augmentation_config: dict[str, Any] | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = False
    use_percentiles: bool = True
    use_relative_action: bool = False
    use_mean_std: bool = False
    state_dropout_prob: float = 0.8
    exclude_state: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_crop_size", _tuple2(self.image_crop_size))
        object.__setattr__(self, "image_target_size", _tuple2(self.image_target_size))
        if self.shortest_image_edge is not None and self.shortest_image_edge <= 0:
            raise ValueError("shortest_image_edge must be positive when set.")
        if self.crop_fraction is not None and self.crop_fraction <= 0:
            raise ValueError("crop_fraction must be positive when set.")
        if not 0 <= self.state_dropout_prob <= 1:
            raise ValueError("state_dropout_prob must be in [0, 1].")


@dataclass(frozen=True)
class GR00TN17BackboneConfig(PretrainedConfig):
    """Cosmos-Reason2 / Qwen3-VL backbone knobs."""

    model_name: str = "nvidia/Cosmos-Reason2-2B"
    qwen3vl: Qwen3VLConfig | None = field(default_factory=_default_gr00t_qwen3vl_config)
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    backbone_embedding_dim: int = 2048
    tune_top_llm_layers: int = 0
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False
    backbone_trainable_params_fp32: bool = True
    use_native_qwen3vl: bool = True
    # Native Qwen3-VL no-cache attention backend. "sdpa" preserves the
    # previous PyTorch SDPA numerics; "eager"/"flashinfer" are opt-in.
    attention_backend: str = "sdpa"
    # CUDA-graph sequence buckets for the native Qwen3-VL text side. The
    # official GR00T config caps VL/action sequence handling at max_seq_len=1024.
    # LIBERO's canonical two-camera prompt is 156 tokens, hence the 160 bucket;
    # larger buckets cover future embodiments without always paying the full
    # 1024-token dense LLM/action-head cost.
    graph_seq_len_buckets: tuple[int, ...] = (160, 192, 256, 384, 512, 768, 1024)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GR00TN17BackboneConfig":
        payload = {k: v for k, v in data.items() if k in cls.field_names()}
        if isinstance(payload.get("qwen3vl"), dict):
            payload["qwen3vl"] = Qwen3VLConfig.from_dict(payload["qwen3vl"])
        return cls(**payload)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graph_seq_len_buckets",
            _tuple_ints(self.graph_seq_len_buckets),
        )
        if not self.model_name:
            raise ValueError("model_name must be non-empty.")
        if self.backbone_embedding_dim <= 0:
            raise ValueError("backbone_embedding_dim must be positive.")
        if self.tune_top_llm_layers < 0:
            raise ValueError("tune_top_llm_layers must be non-negative.")
        if self.attention_backend not in {"eager", "sdpa", "flashinfer"}:
            raise ValueError(
                "backbone.attention_backend must be one of: eager, sdpa, flashinfer."
            )
        if not self.graph_seq_len_buckets:
            raise ValueError("backbone.graph_seq_len_buckets must not be empty.")
        if any(bucket <= 0 for bucket in self.graph_seq_len_buckets):
            raise ValueError("backbone.graph_seq_len_buckets must all be positive.")
        if tuple(sorted(set(self.graph_seq_len_buckets))) != self.graph_seq_len_buckets:
            raise ValueError(
                "backbone.graph_seq_len_buckets must be unique and sorted ascending."
            )


@dataclass(frozen=True)
class GR00TN17DiTConfig(PretrainedConfig):
    """Diffusion transformer config inside the GR00T action head."""

    positional_embeddings: str | None = None
    num_layers: int = 16
    num_attention_heads: int = 32
    attention_head_dim: int = 48
    norm_type: str = "ada_norm"
    norm_elementwise_affine: bool = False
    dropout: float = 0.2
    final_dropout: bool = True
    output_dim: int = 1024
    interleave_self_attention: bool = True
    # Dense no-cache attention backend used inside GR00T's action head.
    # Default "sdpa" (phyai's recommended F.scaled_dot_product_attention), using
    # PyTorch's normal SDPA dispatch (flash / mem-efficient). CUDA-graph
    # captureable. Parity vs the reference: plain sdpa ~0.0221. For tighter parity
    # set ``PHYAI_GR00T_DIT_SDPA_MATH=1`` to force the math backend (mirroring the
    # official DiT's ``enable_flash=False`` / ``GR00T_DIT_SDPA_MODE=math``) ->
    # ~0.0190. Neither is byte-exact: "eager" does the softmax in fp32 and is the
    # only 0.0166 path (selected explicitly for parity validation). "flashinfer"
    # is supported for unmasked self-attention; masked cross-attn falls back to
    # the sdpa path (GR00T uses key-only masks).
    attention_backend: str = "sdpa"

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive.")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.attention_backend not in {"eager", "sdpa", "flashinfer"}:
            raise ValueError(
                "attention_backend must be one of: eager, sdpa, flashinfer."
            )


@dataclass(frozen=True)
class GR00TN17VLSelfAttentionConfig(PretrainedConfig):
    """Optional VL self-attention stack before the action head."""

    num_layers: int = 0
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    dropout: float = 0.2
    final_dropout: bool = True
    positional_embeddings: str | None = None
    output_dim: int | None = None
    # Same backend policy as GR00TN17DiTConfig.attention_backend.
    attention_backend: str = "sdpa"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GR00TN17VLSelfAttentionConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.field_names()})

    def __post_init__(self) -> None:
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative.")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if self.attention_head_dim <= 0:
            raise ValueError("attention_head_dim must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.attention_backend not in {"eager", "sdpa", "flashinfer"}:
            raise ValueError(
                "attention_backend must be one of: eager, sdpa, flashinfer."
            )


@dataclass(frozen=True)
class GR00TN17ActionHeadConfig(PretrainedConfig):
    """GR00T flow-matching action head config."""

    max_state_dim: int = 132
    max_action_dim: int = 132
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536
    state_history_length: int = 1
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True
    attend_text_every_n_blocks: int = 2
    max_num_embodiments: int = 32
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    dit: GR00TN17DiTConfig = field(default_factory=GR00TN17DiTConfig)
    vl_self_attention: GR00TN17VLSelfAttentionConfig | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GR00TN17ActionHeadConfig":
        payload = {k: v for k, v in data.items() if k in cls.field_names()}
        if isinstance(payload.get("dit"), dict):
            payload["dit"] = GR00TN17DiTConfig.from_dict(payload["dit"])
        if isinstance(payload.get("vl_self_attention"), dict):
            payload["vl_self_attention"] = GR00TN17VLSelfAttentionConfig.from_dict(
                payload["vl_self_attention"]
            )
        return cls(**payload)

    def __post_init__(self) -> None:
        if self.max_state_dim <= 0:
            raise ValueError("max_state_dim must be positive.")
        if self.max_action_dim <= 0:
            raise ValueError("max_action_dim must be positive.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.input_embedding_dim <= 0:
            raise ValueError("input_embedding_dim must be positive.")
        if self.state_history_length <= 0:
            raise ValueError("state_history_length must be positive.")
        if not 0 <= self.attn_dropout < 1:
            raise ValueError("attn_dropout must be in [0, 1).")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.attend_text_every_n_blocks <= 0:
            raise ValueError("attend_text_every_n_blocks must be positive.")
        if self.max_num_embodiments <= 0:
            raise ValueError("max_num_embodiments must be positive.")
        if self.num_inference_timesteps <= 0:
            raise ValueError("num_inference_timesteps must be positive.")
        if self.num_timestep_buckets <= 0:
            raise ValueError("num_timestep_buckets must be positive.")


@dataclass(frozen=True)
class GR00TN17Config(PretrainedConfig):
    """Top-level GR00T-N1.7 inference config."""

    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"
    processor: GR00TN17ProcessorConfig = field(default_factory=GR00TN17ProcessorConfig)
    backbone: GR00TN17BackboneConfig = field(default_factory=GR00TN17BackboneConfig)
    action_head: GR00TN17ActionHeadConfig = field(
        default_factory=GR00TN17ActionHeadConfig
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GR00TN17Config":
        """Load either PhyAI nested configs or upstream flat GR00T JSON."""
        if any(k in data for k in ("processor", "backbone", "action_head")):
            processor = GR00TN17ProcessorConfig.from_dict(data.get("processor", {}))
            backbone = GR00TN17BackboneConfig.from_dict(data.get("backbone", {}))
            action_head = GR00TN17ActionHeadConfig.from_dict(
                data.get("action_head", {})
            )
            return cls(
                model_type=data.get("model_type", "Gr00tN1d7"),
                model_dtype=data.get("model_dtype", "bfloat16"),
                processor=processor,
                backbone=backbone,
                action_head=action_head,
            )

        processor = GR00TN17ProcessorConfig.from_dict(data)
        backbone = GR00TN17BackboneConfig.from_dict(data)
        dit = GR00TN17DiTConfig.from_dict(data.get("diffusion_model_cfg", {}))
        vl_self_attention = None
        if (raw := data.get("vl_self_attention_cfg")) is not None:
            vl_self_attention = GR00TN17VLSelfAttentionConfig.from_dict(raw)
        action_payload = {
            k: v for k, v in data.items() if k in GR00TN17ActionHeadConfig.field_names()
        }
        action_payload["dit"] = dit
        action_payload["vl_self_attention"] = vl_self_attention
        action_head = GR00TN17ActionHeadConfig.from_dict(action_payload)
        return cls(
            model_type=data.get("model_type", "Gr00tN1d7"),
            model_dtype=data.get("model_dtype", "bfloat16"),
            processor=processor,
            backbone=backbone,
            action_head=action_head,
        )

    def __post_init__(self) -> None:
        if self.model_type != "Gr00tN1d7":
            raise ValueError(
                f"expected model_type='Gr00tN1d7', got {self.model_type!r}."
            )
        if not self.model_dtype:
            raise ValueError("model_dtype must be non-empty.")
        if self.action_head.dit.output_dim != self.action_head.hidden_size:
            raise ValueError(
                "action_head.dit.output_dim must equal action_head.hidden_size."
            )


__all__ = [
    "GR00TN17ActionHeadConfig",
    "GR00TN17BackboneConfig",
    "GR00TN17Config",
    "GR00TN17DiTConfig",
    "GR00TN17ProcessorConfig",
    "GR00TN17VLSelfAttentionConfig",
]
