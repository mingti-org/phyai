"""Configuration of the released dense MOSS-Image transformer."""

from dataclasses import dataclass
import math

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class MOSSImageConfig(PretrainedConfig):
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        dimensions = (
            self.dim,
            self.ffn_dim,
            self.num_heads,
            self.num_layers,
            self.text_dim,
            self.freq_dim,
            self.latent_channels,
        )
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("MOSS-Image dimensions must be positive integers")
        if self.dim % self.num_heads or self.head_dim % 8:
            raise ValueError("dim must divide into heads with head_dim divisible by 8")
        if self.freq_dim % 2:
            raise ValueError("freq_dim must be even")
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


__all__ = ["MOSSImageConfig"]
