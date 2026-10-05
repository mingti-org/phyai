"""Configuration of the Qwen-Image-2.1 RGBA VAE."""

from __future__ import annotations

from dataclasses import dataclass

from phyai.models.configuration import PretrainedConfig


@dataclass(frozen=True)
class QwenImage21VAEConfig(PretrainedConfig):
    base_dim: int = 96
    decoder_base_dim: int = 144
    z_dim: int = 64
    dim_mult: tuple[int, ...] = (1, 2, 4, 8, 8)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True, True)
    dropout: float = 0.0
    in_channels: int = 4
    out_channels: int = 4
    is_residual: bool = True
    patch_size: int | None = None
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 8
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for name in (
            "dim_mult",
            "attn_scales",
            "temperal_downsample",
            "latents_mean",
            "latents_std",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        if (
            min(
                self.base_dim,
                self.decoder_base_dim,
                self.z_dim,
                self.in_channels,
                self.out_channels,
            )
            <= 0
        ):
            raise ValueError("VAE dimensions must be positive")
        if len(self.dim_mult) < 2 or any(value <= 0 for value in self.dim_mult):
            raise ValueError("dim_mult must contain at least two positive entries")
        if len(self.temperal_downsample) != len(self.dim_mult) - 1:
            raise ValueError(
                "temperal_downsample must match the number of downsample stages"
            )
        if self.num_res_blocks < 1 or not 0 <= self.dropout < 1:
            raise ValueError(
                "num_res_blocks must be positive and dropout must lie in [0, 1)"
            )
        if self.patch_size is not None and self.patch_size < 1:
            raise ValueError("patch_size must be positive when specified")
        patch_size = self.patch_size or 1
        if self.scale_factor_spatial != patch_size * 2 ** (len(self.dim_mult) - 1):
            raise ValueError(
                "scale_factor_spatial disagrees with patching and downsampling"
            )
        if self.scale_factor_temporal != 2 ** sum(self.temperal_downsample):
            raise ValueError(
                "scale_factor_temporal disagrees with temporal downsampling"
            )
        for name in ("latents_mean", "latents_std"):
            value = getattr(self, name)
            if value is not None and len(value) != self.z_dim:
                raise ValueError(f"{name} must contain z_dim entries")
        if self.latents_std is not None and any(
            value <= 0 for value in self.latents_std
        ):
            raise ValueError("latents_std must be positive")


__all__ = ["QwenImage21VAEConfig"]
