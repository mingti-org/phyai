"""Qwen3.5 text configuration for the MOSS prompt encoder."""

from __future__ import annotations

from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class MOSSImageTextConfig(PretrainedConfig):
    """Config for the Qwen3.5 hybrid text decoder."""

    nested_sources = {
        "vocab_size": "text_config.vocab_size",
        "hidden_size": "text_config.hidden_size",
        "intermediate_size": "text_config.intermediate_size",
        "num_hidden_layers": "text_config.num_hidden_layers",
        "num_attention_heads": "text_config.num_attention_heads",
        "num_key_value_heads": "text_config.num_key_value_heads",
        "head_dim": "text_config.head_dim",
        "hidden_act": "text_config.hidden_act",
        "rms_norm_eps": "text_config.rms_norm_eps",
        "rope_theta": (
            "text_config.rope_parameters.rope_theta",
            "rope_parameters.rope_theta",
            "text_config.rope_theta",
        ),
        "rope_type": (
            "text_config.rope_parameters.rope_type",
            "rope_parameters.rope_type",
            "text_config.rope_type",
        ),
        "partial_rotary_factor": (
            "text_config.rope_parameters.partial_rotary_factor",
            "rope_parameters.partial_rotary_factor",
            "text_config.partial_rotary_factor",
        ),
        "mrope_section": (
            "text_config.rope_parameters.mrope_section",
            "rope_parameters.mrope_section",
            "text_config.mrope_section",
        ),
        "max_position_embeddings": "text_config.max_position_embeddings",
        "attention_bias": "text_config.attention_bias",
        "attention_dropout": "text_config.attention_dropout",
        "attn_output_gate": "text_config.attn_output_gate",
        "tie_word_embeddings": "text_config.tie_word_embeddings",
        "linear_conv_kernel_dim": "text_config.linear_conv_kernel_dim",
        "linear_key_head_dim": "text_config.linear_key_head_dim",
        "linear_value_head_dim": "text_config.linear_value_head_dim",
        "linear_num_key_heads": "text_config.linear_num_key_heads",
        "linear_num_value_heads": "text_config.linear_num_value_heads",
        "layer_types": "text_config.layer_types",
    }

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
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
    tie_word_embeddings: bool = False
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    layer_types: tuple[str, ...] = (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ) * 8

    def __post_init__(self) -> None:
        dimensions = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.linear_conv_kernel_dim,
            self.linear_key_head_dim,
            self.linear_value_head_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Text encoder dimensions must be positive integers")
        if self.hidden_act != "silu":
            raise ValueError("MOSS-Image text encoding requires hidden_act='silu'")
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
