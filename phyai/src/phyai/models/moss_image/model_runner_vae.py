"""MOSS-Image VAE runtime and FLUX.2 latent normalization."""

from __future__ import annotations

import torch

from phyai.models.moss_image.vae import MOSSImageVAE
from phyai.runtime.model_runner import ModelRunner


class MOSSImageVAERunner(ModelRunner):
    def __init__(self, vae: MOSSImageVAE, *, use_slicing: bool = False) -> None:
        self.vae = vae
        self.config = vae.config
        self.use_slicing = use_slicing

    def setup(self) -> None:
        self.vae.eval()

    def reset(self) -> None:
        return None

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = latents.shape
        patch_h, patch_w = self.config.patch_size
        if (
            channels != self.config.latent_channels
            or height % patch_h
            or width % patch_w
        ):
            raise ValueError(
                "Raw VAE latents must match latent_channels and patch_size"
            )
        latents = latents.reshape(
            batch, channels, height // patch_h, patch_h, width // patch_w, patch_w
        )
        latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(
            batch, channels * patch_h * patch_w, height // patch_h, width // patch_w
        )
        mean = self.vae.bn.running_mean.to(latents).reshape(1, -1, 1, 1)
        std = (
            (self.vae.bn.running_var + self.config.batch_norm_eps)
            .sqrt()
            .to(latents)
            .reshape(1, -1, 1, 1)
        )
        return (latents - mean) / std

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4 or latents.shape[1] != self.config.packed_latent_channels:
            raise ValueError(
                "Normalized VAE latents must have shape [batch, packed_latent_channels, height, width]"
            )
        mean = self.vae.bn.running_mean.to(latents).reshape(1, -1, 1, 1)
        std = (
            (self.vae.bn.running_var + self.config.batch_norm_eps)
            .sqrt()
            .to(latents)
            .reshape(1, -1, 1, 1)
        )
        latents = latents * std + mean
        batch, _, height, width = latents.shape
        patch_h, patch_w = self.config.patch_size
        latents = latents.reshape(
            batch, self.config.latent_channels, patch_h, patch_w, height, width
        )
        return latents.permute(0, 1, 4, 2, 5, 3).reshape(
            batch, self.config.latent_channels, height * patch_h, width * patch_w
        )

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        normalize: bool = True,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        pixels = pixels.to(self.vae.encoder.conv_in.weight)
        scale = self.config.scale_factor_spatial
        patch_h, patch_w = self.config.patch_size if normalize else (1, 1)
        if pixels.shape[-2] % (scale * patch_h) or pixels.shape[-1] % (scale * patch_w):
            raise ValueError(
                "VAE image dimensions must be divisible by the latent spatial scale"
            )
        moments = (
            torch.cat([self.vae.encode_moments(item) for item in pixels.split(1)])
            if self.use_slicing and pixels.shape[0] > 1
            else self.vae.encode_moments(pixels)
        )
        mean, logvar = moments.chunk(2, dim=1)
        if sample:
            noise = torch.randn(
                mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
            )
            mean = mean + (logvar.clamp(-30, 20) * 0.5).exp() * noise
        return self.normalize_latents(mean) if normalize else mean

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
        latents = latents.to(self.vae.decoder.conv_in.weight)
        if normalize:
            latents = self.denormalize_latents(latents)
        if self.use_slicing and latents.shape[0] > 1:
            return torch.cat([self.vae.decode(item) for item in latents.split(1)])
        return self.vae.decode(latents)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)


__all__ = ["MOSSImageVAERunner"]
