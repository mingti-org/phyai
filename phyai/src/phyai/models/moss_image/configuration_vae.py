"""Configuration of the FLUX.2 autoencoder released with MOSS-Image."""

from __future__ import annotations

from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class MOSSImageVAEConfig(PretrainedConfig):
    in_channels: int = 3
    out_channels: int = 3
    down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",) * 4
    up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",) * 4
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    decoder_block_out_channels: tuple[int, ...] | None = (96, 192, 384, 384)
    layers_per_block: int = 2
    act_fn: str = "silu"
    latent_channels: int = 32
    norm_num_groups: int = 32
    sample_size: int = 1024
    force_upcast: bool = True
    use_quant_conv: bool = True
    use_post_quant_conv: bool = True
    mid_block_add_attention: bool = True
    batch_norm_eps: float = 1e-4
    batch_norm_momentum: float = 0.1
    patch_size: tuple[int, int] = (2, 2)

    def __post_init__(self) -> None:
        for name in (
            "down_block_types",
            "up_block_types",
            "block_out_channels",
            "decoder_block_out_channels",
            "patch_size",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        if (
            min(
                self.in_channels,
                self.out_channels,
                self.latent_channels,
                self.norm_num_groups,
                self.layers_per_block,
                self.sample_size,
            )
            <= 0
        ):
            raise ValueError("VAE dimensions and layer counts must be positive")
        decoder_channels = self.decoder_block_out_channels or self.block_out_channels
        for channels, blocks, expected in (
            (self.block_out_channels, self.down_block_types, "DownEncoderBlock2D"),
            (decoder_channels, self.up_block_types, "UpDecoderBlock2D"),
        ):
            if not channels or len(channels) != len(blocks):
                raise ValueError("VAE block types must match channel widths")
            if any(width <= 0 or width % self.norm_num_groups for width in channels):
                raise ValueError(
                    "VAE channel widths must be divisible by norm_num_groups"
                )
            if any(block != expected for block in blocks):
                raise ValueError(f"MOSS-Image VAE only supports {expected}")
        if len(decoder_channels) != len(self.block_out_channels):
            raise ValueError("VAE encoder and decoder must use the same spatial scale")
        if self.act_fn != "silu":
            raise ValueError("MOSS-Image VAE requires SiLU activation")
        if len(self.patch_size) != 2 or min(self.patch_size) <= 0:
            raise ValueError("patch_size must contain two positive dimensions")
        if self.batch_norm_eps <= 0 or not 0 <= self.batch_norm_momentum <= 1:
            raise ValueError("Invalid VAE batch normalization parameters")

    @property
    def scale_factor_spatial(self) -> int:
        return 2 ** (len(self.block_out_channels) - 1)

    @property
    def packed_latent_channels(self) -> int:
        return self.latent_channels * self.patch_size[0] * self.patch_size[1]


__all__ = ["MOSSImageVAEConfig"]
