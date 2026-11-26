"""Configuration for the Xiaomi Robotics U0 autoregressive model."""

from __future__ import annotations

from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class UNISConfig(PretrainedConfig):
    """Architecture configuration with defaults matching U0-4B."""

    model_type: str = "UNIS"
    vocab_size: int = 282926
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: None = None
    max_position_embeddings: int = 32768
    attention_dropout: float = 0.1
    tie_word_embeddings: bool = False
    pretraining_tp: int = 1

    pad_token_id: int = 151643
    bos_token_id: int = 151849
    eos_token_id: int = 151850
    img_token_id: int = 151851
    boi_token_id: int = 151852
    eoi_token_id: int = 151853
    eol_token_id: int = 151846
    eof_token_id: int = 151847

    def __post_init__(self) -> None:
        """Reject invalid dimensions, token IDs and unsupported architecture options."""
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Query heads must be divisible by KV heads.")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE.")
        if not self.rms_norm_eps > 0 or not self.rope_theta > 0:
            raise ValueError("rms_norm_eps and rope_theta must be positive.")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1).")
        if self.model_type != "UNIS":
            raise ValueError("Expected model_type='UNIS'.")
        if self.hidden_act != "silu":
            raise ValueError("The initial U0 implementation supports SiLU only.")
        if self.rope_scaling is not None:
            raise ValueError("Scaled RoPE is not supported by the initial port.")
        if self.pretraining_tp != 1:
            raise ValueError("The initial port requires pretraining_tp=1.")
        if self.tie_word_embeddings:
            raise ValueError("The initial port requires separate embedding and head.")

        for name in (
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "img_token_id",
            "boi_token_id",
            "eoi_token_id",
            "eol_token_id",
            "eof_token_id",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} must be an integer within the vocabulary.")

    @property
    def num_key_value_groups(self) -> int:
        """Return the number of query heads sharing each key/value head."""
        return self.num_attention_heads // self.num_key_value_heads


__all__ = ["UNISConfig"]
