"""Checkpoint geometry for MolmoAct2 vision, language, and action modules."""

from typing import ClassVar
from dataclasses import field, replace, dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class MolmoAct2TextConfig(PretrainedConfig):
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 154624
    additional_vocab_size: int = 128
    max_position_embeddings: int = 16384
    layer_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    rope_type: str = "default"
    rope_scaling: dict | None = None
    rope_scaling_layers: tuple[int, ...] | None = None
    qkv_bias: bool = False
    use_qk_norm: bool = True
    qk_norm_type: str = "qwen3"
    norm_after: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    attention_dropout: float = 0.0
    embedding_dropout: float = 0.0
    residual_dropout: float = 0.0

    nested_sources: ClassVar[dict[str, str]] = {
        "rope_type": "rope_parameters.rope_type",
        "rope_theta": "rope_parameters.rope_theta",
    }

    def __post_init__(self) -> None:
        if (
            min(
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.vocab_size,
                self.max_position_embeddings,
            )
            <= 0
        ):
            raise ValueError("Text dimensions must be positive.")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError(
                "Text GQA heads must divide evenly and head_dim must be even."
            )
        if (
            self.rope_type != "default"
            or self.rope_scaling_layers is not None
            or self.rope_scaling is not None
        ):
            raise ValueError("MolmoAct2 currently supports default, unscaled RoPE.")
        if self.tie_word_embeddings:
            raise ValueError("MolmoAct2 requires an untied language output head.")
        if self.qk_norm_type not in ("qwen3", "olmo"):
            raise ValueError("Unsupported MolmoAct2 qk_norm_type.")
        if self.hidden_act != "silu":
            raise ValueError("MolmoAct2 text MLP requires silu.")


@dataclass(frozen=True)
class MolmoAct2ViTConfig(PretrainedConfig):
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    image_default_input_size: tuple[int, int] = (378, 378)
    image_patch_size: int = 14
    image_num_pos: int = 729
    layer_norm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    float32_attention: bool = True
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "image_default_input_size", tuple(self.image_default_input_size)
        )
        if (
            min(
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.image_patch_size,
                self.image_num_pos,
            )
            <= 0
        ):
            raise ValueError("Vision dimensions must be positive.")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "Vision hidden_size must equal num_attention_heads * head_dim."
            )
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError(
                "MolmoAct2 vision requires equal query and KV head counts."
            )
        if self.hidden_act != "gelu_pytorch_tanh":
            raise ValueError("MolmoAct2 vision requires tanh GELU.")


@dataclass(frozen=True)
class MolmoAct2AdapterConfig(PretrainedConfig):
    hidden_size: int = 1152
    text_hidden_size: int = 2560
    intermediate_size: int = 9728
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 72
    vit_layers: tuple[int, ...] = (-3, -9)
    hidden_act: str = "silu"
    float32_attention: bool = True
    pooling_attention_mask: bool = True
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    image_feature_dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "vit_layers", tuple(self.vit_layers))
        if (
            not self.vit_layers
            or min(
                self.hidden_size,
                self.text_hidden_size,
                self.intermediate_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
            )
            <= 0
        ):
            raise ValueError(
                "Adapter dimensions and selected vision layers must be nonempty."
            )
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("Adapter attention heads must span hidden_size.")
        if (
            self.num_attention_heads != self.num_key_value_heads
            or self.hidden_act != "silu"
        ):
            raise ValueError("MolmoAct2 adapter requires MHA and silu.")


@dataclass(frozen=True)
class MolmoAct2ActionExpertConfig(PretrainedConfig):
    hidden_size: int = 768
    num_heads: int = 8
    num_layers: int = 36
    mlp_ratio: float = 4.0
    ffn_multiple_of: int = 256
    timestep_embed_dim: int = 256
    max_action_dim: int = 32
    max_action_horizon: int = 30
    context_layer_norm: bool = True
    qk_norm: bool = True
    qk_norm_eps: float = 1e-6
    rope: bool = True
    causal_attn: bool = False
    attn_dropout: float = 0.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if (
            min(
                self.hidden_size,
                self.num_heads,
                self.num_layers,
                self.mlp_ratio,
                self.ffn_multiple_of,
                self.timestep_embed_dim,
                self.max_action_dim,
                self.max_action_horizon,
                self.qk_norm_eps,
            )
            <= 0
        ):
            raise ValueError(
                "Action expert dimensions and normalization epsilon must be positive."
            )
        if (
            self.hidden_size % self.num_heads
            or (self.hidden_size // self.num_heads) % 2
        ):
            raise ValueError(
                "Action heads must divide hidden_size into even head dimensions."
            )
        if self.timestep_embed_dim % 2:
            raise ValueError("timestep_embed_dim must be even.")


@dataclass(frozen=True)
class MolmoAct2Config(PretrainedConfig):
    text_config: MolmoAct2TextConfig = field(default_factory=MolmoAct2TextConfig)
    vit_config: MolmoAct2ViTConfig = field(default_factory=MolmoAct2ViTConfig)
    adapter_config: MolmoAct2AdapterConfig = field(
        default_factory=MolmoAct2AdapterConfig
    )
    action_expert_config: MolmoAct2ActionExpertConfig = field(
        default_factory=MolmoAct2ActionExpertConfig
    )
    add_action_expert: bool = True
    action_mode: str = "both"
    state_format: str = "discrete"
    tie_word_embeddings: bool = False
    max_action_dim: int = 32
    max_action_horizon: int = 30
    flow_matching_num_steps: int = 10
    mask_action_dim_padding: bool = True
    enable_depth_reasoning: bool = False
    action_expert_depth_gate: bool = False
    image_patch_id: int = 154626
    image_end_token_id: int = 154625
    action_start_token_id: int = 151932
    action_end_token_id: int = 151933
    action_token_start_id: int = 151934
    num_action_tokens: int = 2048
    eos_token_id: int = 151645
    n_obs_steps: int = 1

    def __post_init__(self) -> None:
        if self.action_mode not in ("continuous", "discrete", "both"):
            raise ValueError("Unsupported action_mode.")
        if self.state_format != "discrete":
            raise ValueError("MolmoAct2 requires discrete state prompting.")
        if self.tie_word_embeddings:
            raise ValueError("MolmoAct2 requires untied word embeddings.")
        if self.enable_depth_reasoning or self.action_expert_depth_gate:
            raise ValueError(
                "Depth-reasoning checkpoints are not supported by this MolmoAct2 port."
            )
        if (
            min(
                self.max_action_dim,
                self.max_action_horizon,
                self.flow_matching_num_steps,
            )
            <= 0
        ):
            raise ValueError("Action dimensions and flow steps must be positive.")
        if self.adapter_config.text_hidden_size != self.text_config.hidden_size:
            raise ValueError("Adapter output must match the text hidden size.")
        if self.adapter_config.hidden_size != self.vit_config.hidden_size:
            raise ValueError("Adapter hidden size must match the vision hidden size.")
        if (
            self.add_action_expert
            and self.action_expert_config.num_layers
            != self.text_config.num_hidden_layers
        ):
            raise ValueError("Action expert requires one block per text layer.")
        object.__setattr__(
            self,
            "action_expert_config",
            replace(
                self.action_expert_config,
                max_action_dim=self.max_action_dim,
                max_action_horizon=self.max_action_horizon,
            ),
        )
