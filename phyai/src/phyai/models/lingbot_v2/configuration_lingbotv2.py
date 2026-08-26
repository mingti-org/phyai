"""Configs for LingBot-VLA 2.0.

The organization follows :mod:`phyai.models.pi0.configuration_pi0`: leaf
configs describe one weight-bearing subsystem, while the top-level config
composes them and validates every interface where tensors cross subsystems.

* :class:`Qwen3VLVisionConfig` -- Qwen3-VL-4B vision tower.
* :class:`Qwen3VLTextConfig` -- Qwen3-VL-4B language decoder.
* :class:`LingBotV2ExpertConfig` -- Qwen2-style action stream.
* :class:`LingBotV2MoEConfig` -- sparse FFN replacement in the action stream.
* :class:`LingBotV2DualQueryConfig` -- learned current/future prefix queries.
* :class:`LingBotV2FlowMatchingConfig` -- action geometry and ODE schedule.
* :class:`LingBotVLA2Config` -- complete inference-time composition.

Defaults describe the public LingBot-VLA 2.0 RoboTwin checkpoint.  Training
only paths, teacher losses, optimizer settings, and runtime kernel choices do
not belong to this model-geometry config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class Qwen3VLVisionConfig(PretrainedConfig):
    """Qwen3-VL-4B-Instruct vision tower used by LingBot-VLA 2.0."""

    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 2560
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (5, 11, 17)
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.deepstack_visual_indexes, tuple):
            object.__setattr__(
                self,
                "deepstack_visual_indexes",
                tuple(self.deepstack_visual_indexes),
            )

        positive_int_fields = {
            "depth": self.depth,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_heads": self.num_heads,
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "spatial_merge_size": self.spatial_merge_size,
            "temporal_patch_size": self.temporal_patch_size,
            "out_hidden_size": self.out_hidden_size,
            "num_position_embeddings": self.num_position_embeddings,
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

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
                "perfect square."
            )

        if tuple(sorted(set(self.deepstack_visual_indexes))) != (
            self.deepstack_visual_indexes
        ):
            raise ValueError(
                "deepstack_visual_indexes must be unique and strictly increasing, "
                f"got {self.deepstack_visual_indexes}."
            )
        if any(i < 0 or i >= self.depth for i in self.deepstack_visual_indexes):
            raise ValueError(
                f"deepstack_visual_indexes={self.deepstack_visual_indexes} must all "
                f"be in [0, depth={self.depth})."
            )

        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, got {self.initializer_range}."
            )
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}.")

    @property
    def head_dim(self) -> int:
        """Per-head width of vision self-attention."""

        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        """Side length of the learned square position-embedding grid."""

        return int(self.num_position_embeddings**0.5)

    @property
    def spatial_merge_unit(self) -> int:
        """Number of spatial patch tokens collapsed into one vision token."""

        return self.spatial_merge_size**2

    @property
    def patch_vector_dim(self) -> int:
        """Flattened input width of one spatiotemporal patch."""

        return (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )


@dataclass(frozen=True)
class Qwen3VLTextConfig(PretrainedConfig):
    """Qwen3-VL-4B-Instruct text tower used by LingBot-VLA 2.0."""

    nested_sources = {
        "mrope_interleaved": (
            "rope_scaling.mrope_interleaved",
            "rope_parameters.mrope_interleaved",
        ),
        "mrope_section": (
            "rope_scaling.mrope_section",
            "rope_parameters.mrope_section",
        ),
        "rope_type": (
            "rope_scaling.rope_type",
            "rope_parameters.rope_type",
        ),
        "rope_theta": (
            "rope_scaling.rope_theta",
            "rope_parameters.rope_theta",
        ),
    }

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    mrope_interleaved: bool = True
    mrope_section: tuple[int, ...] = (24, 20, 20)
    rope_type: str = "default"
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mrope_section, tuple):
            object.__setattr__(self, "mrope_section", tuple(self.mrope_section))

        positive_int_fields = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads}."
            )
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be even for MRoPE.")

        if len(self.mrope_section) != 3:
            raise ValueError(
                "mrope_section must contain temporal, height, and width sections, "
                f"got {self.mrope_section}."
            )
        if any(section <= 0 for section in self.mrope_section):
            raise ValueError(
                f"mrope_section values must be positive, got {self.mrope_section}."
            )
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(self.mrope_section)} must equal "
                f"head_dim//2={self.head_dim // 2}."
            )
        if self.rope_type != "default":
            raise ValueError(
                f"LingBot-VLA 2.0 expects rope_type='default', got {self.rope_type!r}."
            )

        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}.")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}.")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(
                "attention_dropout must be in [0, 1), " f"got {self.attention_dropout}."
            )
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, got {self.initializer_range}."
            )

        token_ids = {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        for name, token_id in token_ids.items():
            if token_id is not None and not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"{name}={token_id} must be in [0, vocab_size={self.vocab_size})."
                )

    @property
    def num_key_value_groups(self) -> int:
        """Number of query-head groups sharing each key/value head."""

        return self.num_attention_heads // self.num_key_value_heads

    @property
    def joint_attention_dim(self) -> int:
        """Query/output width used by LingBot joint attention."""

        return self.num_attention_heads * self.head_dim

    @property
    def key_value_attention_dim(self) -> int:
        """Key/value projection width before GQA head replication."""

        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True)
class LingBotV2ExpertConfig(PretrainedConfig):
    """Qwen2-style action stream used by LingBot-VLA 2.0 joint attention.

    The upstream implementation constructs a Qwen2 config as a convenient
    module factory, then bypasses the expert's standalone embedding, LM head,
    RoPE, cache, and sliding-window path.  Consequently this config keeps only
    fields that affect the retained decoder weights or forward computation.
    Joint Q/K positions use :class:`Qwen3VLTextConfig` MRoPE.
    """

    nested_sources = {
        "hidden_size": "expert_hidden_size",
        "intermediate_size": "expert_intermediate_size",
        "num_attention_heads": "action_num_attention_heads",
        "num_key_value_heads": "action_num_key_value_heads",
        "head_dim": "action_head_dim",
        "use_adarms": "adanorm_time",
        "final_norm_adarms": "final_norm_adanorm",
    }

    hidden_size: int = 768
    intermediate_size: int = 2752
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    qkv_bias: bool = True
    output_bias: bool = False
    mlp_bias: bool = False
    use_adarms: bool = True
    adarms_cond_dim: int = 768
    final_norm_adarms: bool = False

    def __post_init__(self) -> None:
        positive_int_fields = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "adarms_cond_dim": self.adarms_cond_dim,
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} not divisible by "
                f"num_key_value_heads={self.num_key_value_heads}."
            )
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE.")

        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}.")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(
                "attention_dropout must be in [0, 1), " f"got {self.attention_dropout}."
            )
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, got {self.initializer_range}."
            )

        if self.use_adarms and self.adarms_cond_dim != self.hidden_size:
            raise ValueError(
                f"adarms_cond_dim={self.adarms_cond_dim} must equal "
                f"hidden_size={self.hidden_size} when use_adarms=True."
            )
        if self.final_norm_adarms and not self.use_adarms:
            raise ValueError("final_norm_adarms=True requires use_adarms=True.")

    @property
    def num_key_value_groups(self) -> int:
        """Number of query-head groups sharing each key/value head."""

        return self.num_attention_heads // self.num_key_value_heads

    @property
    def joint_attention_dim(self) -> int:
        """Query/output width shared with the Qwen3-VL text stream."""

        return self.num_attention_heads * self.head_dim

    @property
    def key_value_attention_dim(self) -> int:
        """Key/value projection width before GQA head replication."""

        return self.num_key_value_heads * self.head_dim


@dataclass(frozen=True)
class LingBotV2MoEConfig(PretrainedConfig):
    """Sparse token-level MoE inside every released V2 action-expert layer."""

    nested_sources = {
        "layer_indices": "token_moe_layers",
        "num_experts": "token_num_experts",
        "top_k": "token_top_k",
        "moe_intermediate_size": "token_moe_intermediate_size",
        "shared_expert_intermediate_size": "token_shared_intermediate_size",
        "shared_expert_gate": "use_shared_expert_gate",
    }

    layer_indices: tuple[int, ...] = tuple(range(36))
    num_experts: int = 32
    top_k: int = 4
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 704
    router_activation: str = "sigmoid"
    normalize_topk_prob: bool = True
    routed_scaling_factor: float = 4.0
    router_bias: bool = False
    use_router_correction_bias: bool = True
    shared_expert_gate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.layer_indices, tuple):
            object.__setattr__(self, "layer_indices", tuple(self.layer_indices))

        positive_int_fields = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "moe_intermediate_size": self.moe_intermediate_size,
            "shared_expert_intermediate_size": (self.shared_expert_intermediate_size),
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if not self.layer_indices:
            raise ValueError("layer_indices must contain at least one MoE layer.")
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError(
                "layer_indices must be unique and strictly increasing, "
                f"got {self.layer_indices}."
            )
        if any(index < 0 for index in self.layer_indices):
            raise ValueError(
                f"layer_indices must be non-negative, got {self.layer_indices}."
            )
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k={self.top_k} must not exceed num_experts={self.num_experts}."
            )
        if self.router_activation not in {"sigmoid", "softmax"}:
            raise ValueError(
                "router_activation must be 'sigmoid' or 'softmax', "
                f"got {self.router_activation!r}."
            )
        if self.routed_scaling_factor <= 0:
            raise ValueError(
                "routed_scaling_factor must be positive, "
                f"got {self.routed_scaling_factor}."
            )

    @property
    def num_moe_layers(self) -> int:
        """Number of decoder MLPs replaced by sparse MoE blocks."""

        return len(self.layer_indices)

    @property
    def active_intermediate_size(self) -> int:
        """Dense-equivalent FFN width evaluated for one token."""

        return (
            self.top_k * self.moe_intermediate_size
            + self.shared_expert_intermediate_size
        )

    @property
    def total_intermediate_capacity(self) -> int:
        """Total routed plus shared FFN width stored in one MoE layer."""

        return (
            self.num_experts * self.moe_intermediate_size
            + self.shared_expert_intermediate_size
        )


@dataclass(frozen=True)
class LingBotV2DualQueryConfig(PretrainedConfig):
    """Inference-time current/future perceptual queries retained after distillation.

    Teacher checkpoint paths and auxiliary loss coefficients are deliberately
    excluded.  The learned seed tables and shared projections remain part of
    the released inference graph because their eight-token current/future
    summaries are appended to the Qwen3-VL prefix.  The released RoboTwin
    training config leaves both future-query suffix masks disabled, so the
    action stream can attend to the shared future query tokens.
    """

    nested_sources = {
        "num_query_seeds": "depth.num_backbone_tokens",
        "query_hidden_size": "llm.dim_out",
        "image_token_size": "llm.image_token_size",
        "image_input_size": "llm.image_input_size",
        "use_future_depth": "depth.use_future_depth",
        "block_future_depth_to_action": ("depth.block_future_depth_to_action"),
        "use_future_video_patch": "video.use_patch_loss",
        "use_current_video_patch": "video.use_current_patch_loss",
        "share_future_depth_query": "video.share_future_depth_query",
        "use_shared_future_task_proj": ("video.use_shared_future_task_proj"),
        "use_current_shared_task_proj": ("video.use_current_shared_task_proj"),
        "use_future_video_cls": "video.use_cls_loss",
        "block_suffix_to_future_video": ("video.block_suffix_to_future_video"),
    }

    mode: str = "query"
    num_task_tokens: int = 8
    num_query_seeds: int = 256
    query_hidden_size: int = 2560
    image_token_size: int = 8
    image_input_size: int = 224
    use_future_depth: bool = True
    use_future_video: bool = True
    use_future_video_patch: bool = True
    use_current_video_patch: bool = True
    share_future_depth_query: bool = True
    use_shared_future_task_proj: bool = True
    use_current_shared_task_proj: bool = True
    use_future_video_cls: bool = False
    block_future_depth_to_action: bool = False
    block_suffix_to_future_video: bool = False
    fusion_bias: bool = True

    def __post_init__(self) -> None:
        positive_int_fields = {
            "num_task_tokens": self.num_task_tokens,
            "num_query_seeds": self.num_query_seeds,
            "query_hidden_size": self.query_hidden_size,
            "image_token_size": self.image_token_size,
            "image_input_size": self.image_input_size,
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.mode != "query":
            raise ValueError(
                f"LingBot-VLA 2.0 expects mode='query', got {self.mode!r}."
            )
        if self.num_query_seeds % self.num_task_tokens != 0:
            raise ValueError(
                f"num_query_seeds={self.num_query_seeds} must be divisible by "
                f"num_task_tokens={self.num_task_tokens}."
            )
        if not self.use_future_video:
            video_only_flags = {
                "use_future_video_patch": self.use_future_video_patch,
                "use_current_video_patch": self.use_current_video_patch,
                "share_future_depth_query": self.share_future_depth_query,
                "use_shared_future_task_proj": self.use_shared_future_task_proj,
                "use_current_shared_task_proj": self.use_current_shared_task_proj,
                "use_future_video_cls": self.use_future_video_cls,
                "block_suffix_to_future_video": (self.block_suffix_to_future_video),
            }
            enabled = [name for name, value in video_only_flags.items() if value]
            if enabled:
                raise ValueError(
                    "use_future_video=False is incompatible with enabled video "
                    f"options: {enabled}."
                )
        if self.share_future_depth_query and not (
            self.use_future_depth and self.use_future_video_patch
        ):
            raise ValueError(
                "share_future_depth_query=True requires both future-depth and "
                "future-video patch queries."
            )
        if self.use_shared_future_task_proj and not (
            self.share_future_depth_query and self.use_future_video_patch
        ):
            raise ValueError(
                "use_shared_future_task_proj=True requires a shared future "
                "depth/video patch query."
            )
        if self.use_current_shared_task_proj and not self.use_current_video_patch:
            raise ValueError(
                "use_current_shared_task_proj=True requires "
                "use_current_video_patch=True."
            )
        if self.block_future_depth_to_action and not self.use_future_depth:
            raise ValueError(
                "block_future_depth_to_action=True requires future-depth queries."
            )

    @property
    def seeds_per_task_token(self) -> int:
        """Seed rows averaged into each effective prefix query token."""

        return self.num_query_seeds // self.num_task_tokens

    @property
    def current_query_token_count(self) -> int:
        """Effective current-perception tokens appended to the prefix."""

        return self.num_task_tokens

    @property
    def future_query_token_count(self) -> int:
        """Effective future-perception tokens appended to the prefix."""

        count = self.num_task_tokens if self.use_future_depth else 0
        if self.use_future_video_cls:
            count += 1
        if self.use_future_video_patch and not self.share_future_depth_query:
            count += self.num_task_tokens
        return count

    @property
    def prefix_query_token_count(self) -> int:
        """Total non-language dual-query tokens in the VLM prefix."""

        return self.current_query_token_count + self.future_query_token_count

    @property
    def num_query_seed_tables(self) -> int:
        """Learned seed tables retained by the released inference graph."""

        count = 1
        if self.use_future_depth:
            count += 1
        if self.use_current_video_patch:
            count += 1
        if self.use_future_video_patch and (
            not self.share_future_depth_query or self.use_shared_future_task_proj
        ):
            count += 1
        return count


@dataclass(frozen=True)
class LingBotV2FlowMatchingConfig(PretrainedConfig):
    """Canonical action geometry and inference-time flow-matching schedule."""

    nested_sources = {
        "num_inference_steps": "num_steps",
        "num_observation_steps": "n_obs_steps",
    }

    action_dim: int = 55
    max_action_dim: int = 55
    max_state_dim: int = 55
    chunk_size: int = 50
    num_inference_steps: int = 10
    num_observation_steps: int = 1
    time_embedding_min_period: float = 4e-3
    time_embedding_max_period: float = 4.0
    time_embedding_activation: str = "silu"

    def __post_init__(self) -> None:
        positive_int_fields = {
            "action_dim": self.action_dim,
            "max_action_dim": self.max_action_dim,
            "max_state_dim": self.max_state_dim,
            "chunk_size": self.chunk_size,
            "num_inference_steps": self.num_inference_steps,
            "num_observation_steps": self.num_observation_steps,
        }
        for name, value in positive_int_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

        if self.action_dim > self.max_action_dim:
            raise ValueError(
                f"action_dim={self.action_dim} must not exceed "
                f"max_action_dim={self.max_action_dim}."
            )
        if not 0 < self.time_embedding_min_period < self.time_embedding_max_period:
            raise ValueError(
                "time embedding periods must satisfy 0 < min < max, got "
                f"{self.time_embedding_min_period} and "
                f"{self.time_embedding_max_period}."
            )
        if self.time_embedding_activation != "silu":
            raise ValueError(
                "LingBot-VLA 2.0 expects time_embedding_activation='silu', "
                f"got {self.time_embedding_activation!r}."
            )

    @property
    def suffix_length(self) -> int:
        """One state token followed by one token per action-chunk step."""

        return 1 + self.chunk_size


@dataclass(frozen=True)
class LingBotVLA2Config(PretrainedConfig):
    """Complete inference architecture for the released LingBot-VLA 2.0 6B.

    As in :class:`phyai.models.pi0.configuration_pi0.PI0Config`, this class is
    the integration boundary.  Leaf configs validate themselves; this class
    validates projection widths, paired joint-attention geometry, MoE layer
    replacement, and Dual Query placement in the Qwen prefix.

    Flow-matching values stay in a leaf config because LingBot has more
    independently sourced policy settings than pi0.  Read-only aliases below
    preserve pi0's convenient top-level interface for modeling code.
    """

    nested_sources = {
        "vision": "vision_config",
        "text": "text_config",
        "expert": ("expert_config", "qwen_expert_config"),
        "moe": "moe_config",
        "dual_query": ("dual_query_config", "align_params"),
        "flow_matching": ("flow_matching_config", "action_config"),
    }

    vision: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    text: Qwen3VLTextConfig = field(default_factory=Qwen3VLTextConfig)
    expert: LingBotV2ExpertConfig = field(default_factory=LingBotV2ExpertConfig)
    moe: LingBotV2MoEConfig = field(default_factory=LingBotV2MoEConfig)
    dual_query: LingBotV2DualQueryConfig = field(
        default_factory=LingBotV2DualQueryConfig
    )
    flow_matching: LingBotV2FlowMatchingConfig = field(
        default_factory=LingBotV2FlowMatchingConfig
    )

    vlm_family: str = "qwen3_vl"
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tokenizer_max_length: int = 72
    vlm_causal: bool = True
    use_vision_boundaries: bool = True
    use_lm_head: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LingBotVLA2Config:
        """Also accept the official flattened training/deployment dictionary."""

        normalized = dict(data)
        flat = dict(data)

        expert_keys = {
            "expert_hidden_size",
            "expert_intermediate_size",
            "action_num_attention_heads",
            "action_num_key_value_heads",
            "action_head_dim",
            "adanorm_time",
            "final_norm_adanorm",
        }
        if "expert" not in normalized and expert_keys.intersection(flat):
            normalized["expert"] = flat

        moe_keys = {
            "token_moe_layers",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "router_activation",
            "routed_scaling_factor",
            "use_shared_expert_gate",
        }
        if "moe" not in normalized and moe_keys.intersection(flat):
            normalized["moe"] = flat

        if "dual_query" not in normalized and isinstance(
            flat.get("align_params"), dict
        ):
            normalized["dual_query"] = flat["align_params"]

        flow_keys = {
            "action_dim",
            "max_action_dim",
            "max_state_dim",
            "chunk_size",
            "num_steps",
            "n_obs_steps",
        }
        if "flow_matching" not in normalized and flow_keys.intersection(flat):
            normalized["flow_matching"] = flat

        return super().from_dict(normalized)

    def __post_init__(self) -> None:
        if self.vlm_family != "qwen3_vl":
            raise ValueError(
                f"LingBot-VLA 2.0 expects vlm_family='qwen3_vl', "
                f"got {self.vlm_family!r}."
            )
        if self.vision.out_hidden_size != self.text.hidden_size:
            raise ValueError(
                f"vision.out_hidden_size={self.vision.out_hidden_size} must equal "
                f"text.hidden_size={self.text.hidden_size}."
            )

        joint_pairs = {
            "num_hidden_layers": (
                self.text.num_hidden_layers,
                self.expert.num_hidden_layers,
            ),
            "num_attention_heads": (
                self.text.num_attention_heads,
                self.expert.num_attention_heads,
            ),
            "num_key_value_heads": (
                self.text.num_key_value_heads,
                self.expert.num_key_value_heads,
            ),
            "head_dim": (self.text.head_dim, self.expert.head_dim),
            "joint_attention_dim": (
                self.text.joint_attention_dim,
                self.expert.joint_attention_dim,
            ),
            "key_value_attention_dim": (
                self.text.key_value_attention_dim,
                self.expert.key_value_attention_dim,
            ),
        }
        for name, (text_value, expert_value) in joint_pairs.items():
            if text_value != expert_value:
                raise ValueError(
                    f"text.{name}={text_value} must equal "
                    f"expert.{name}={expert_value} for joint attention."
                )

        if max(self.moe.layer_indices) >= self.expert.num_hidden_layers:
            raise ValueError(
                f"MoE layer index {max(self.moe.layer_indices)} is outside "
                f"expert layers [0, {self.expert.num_hidden_layers})."
            )
        if self.moe.active_intermediate_size != self.expert.intermediate_size:
            raise ValueError(
                f"moe.active_intermediate_size={self.moe.active_intermediate_size} "
                f"must equal expert.intermediate_size={self.expert.intermediate_size}."
            )
        if self.dual_query.query_hidden_size != self.text.hidden_size:
            raise ValueError(
                f"dual_query.query_hidden_size={self.dual_query.query_hidden_size} "
                f"must equal text.hidden_size={self.text.hidden_size}."
            )

        if self.tokenizer_max_length <= 0:
            raise ValueError(
                f"tokenizer_max_length must be positive, "
                f"got {self.tokenizer_max_length}."
            )
        token_ids = {
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
        }
        for name, token_id in token_ids.items():
            if not 0 <= token_id < self.text.vocab_size:
                raise ValueError(
                    f"{name}={token_id} must be in "
                    f"[0, vocab_size={self.text.vocab_size})."
                )
        if len(set(token_ids.values())) != len(token_ids):
            raise ValueError(f"multimodal token IDs must be unique, got {token_ids}.")

    @property
    def num_layers(self) -> int:
        """Number of paired text/action joint-attention layers."""

        return self.text.num_hidden_layers

    @property
    def joint_attention_dim(self) -> int:
        """Shared query/output width of text and action attention streams."""

        return self.text.joint_attention_dim

    @property
    def proj_width(self) -> int:
        """Width used by state/action input and action output projections."""

        return self.expert.hidden_size

    @property
    def action_dim(self) -> int:
        return self.flow_matching.action_dim

    @property
    def max_action_dim(self) -> int:
        return self.flow_matching.max_action_dim

    @property
    def max_state_dim(self) -> int:
        return self.flow_matching.max_state_dim

    @property
    def chunk_size(self) -> int:
        return self.flow_matching.chunk_size

    @property
    def n_action_steps(self) -> int:
        """Official LingBot name for the generated action-chunk length."""

        return self.flow_matching.chunk_size

    @property
    def num_inference_steps(self) -> int:
        return self.flow_matching.num_inference_steps

    @property
    def num_steps(self) -> int:
        """Official LingBot alias for the flow-matching Euler step count."""

        return self.flow_matching.num_inference_steps

    @property
    def n_obs_steps(self) -> int:
        return self.flow_matching.num_observation_steps

    @property
    def suffix_len(self) -> int:
        """State token followed by one token for each action-chunk step."""

        return self.flow_matching.suffix_length

    @property
    def image_token_index(self) -> int:
        """Alias used by model implementations that follow HF naming."""

        return self.image_token_id


__all__ = [
    "LingBotV2DualQueryConfig",
    "LingBotV2ExpertConfig",
    "LingBotV2FlowMatchingConfig",
    "LingBotV2MoEConfig",
    "LingBotVLA2Config",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
]
