"""Checkpoint configuration for the Qwen-Image 2.1 denoiser."""

from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class QwenImage21Config(PretrainedConfig):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = 64
    num_layers: int = 32
    attention_head_dim: int = 128
    num_attention_heads: int = 32
    context_in_dim: int = 4096
    mlp_ratio: int = 3
    axes_dims_rope: tuple[int, int, int] = (16, 56, 56)
    eps: float = 1e-6
    causal_condition: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes_dims_rope", tuple(self.axes_dims_rope))
        if self.out_channels is None:
            object.__setattr__(self, "out_channels", self.in_channels)
        if self.patch_size != 1:
            raise ValueError("Qwen-Image 2.1 consumes unpatched latent tokens")
        if any(
            value <= 0
            for value in (
                self.in_channels,
                self.out_channels,
                self.num_layers,
                self.attention_head_dim,
                self.num_attention_heads,
                self.context_in_dim,
                self.mlp_ratio,
                self.eps,
            )
        ):
            raise ValueError("model dimensions and eps must be positive")
        if (
            len(self.axes_dims_rope) != 3
            or any(dim <= 0 or dim % 2 for dim in self.axes_dims_rope)
            or sum(self.axes_dims_rope) != self.attention_head_dim
        ):
            raise ValueError("three even rotary axes must partition attention_head_dim")

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim
