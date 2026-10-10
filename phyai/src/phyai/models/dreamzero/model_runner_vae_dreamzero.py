"""DreamZero VAE runner."""

from __future__ import annotations

import torch

from phyai.models.dreamzero.vae_wan import DreamZeroWanVAE
from phyai.runtime.model_runner import ModelRunner


class DreamZeroVAERunner(ModelRunner):
    """Wraps the DreamZero Wan2.1 VAE encoder and decoder."""

    def __init__(
        self,
        vae: DreamZeroWanVAE,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype

    def setup(self) -> None:
        return None

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
        use_autocast: bool = True,
    ) -> torch.Tensor:
        pixels = pixels.to(self.device, self.dtype)
        if use_autocast and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                return self.vae.encode(
                    pixels,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
        return self.vae.encode(
            pixels,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        return self.vae.decode(
            latents.to(self.device, self.dtype),
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)


__all__ = ["DreamZeroVAERunner"]
