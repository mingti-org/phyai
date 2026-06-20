"""Cosmos3 VAE model runners — wrap the WAN video VAE and the AVAE sound decoder."""

from __future__ import annotations

import torch

from phyai.models.cosmos3.avae_sound import Cosmos3AVAESoundDecoder
from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE
from phyai.runtime.model_runner import ModelRunner


class Cosmos3VAERunner(ModelRunner):
    """Wraps :class:`Cosmos3WanVAE`; owns its device/dtype and routes decode/encode."""

    def __init__(
        self,
        vae: Cosmos3WanVAE,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype

    def setup(self) -> None:
        """No-op: the WAN VAE has no warmup / graph capture."""
        return None

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents ``[B, z, t, h, w]`` -> pixels ``[B, 3, T, H, W]`` in ``[-1, 1]``."""
        return self.vae.decode(latents.to(self.device, self.dtype))

    @torch.no_grad()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Pixels ``[B, 3, T, H, W]`` in ``[-1, 1]`` -> normalized latent ``[B, z, t, h, w]``."""
        return self.vae.encode(
            pixels.to(self.device, self.dtype), sample=sample, generator=generator
        )

    @torch.no_grad()
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode is the canonical hot path; ``forward`` aliases :meth:`decode`."""
        return self.decode(latents)


class Cosmos3SoundVAERunner(ModelRunner):
    """Wraps :class:`Cosmos3AVAESoundDecoder`; owns its device/dtype and routes decode."""

    def __init__(
        self,
        avae: Cosmos3AVAESoundDecoder,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.avae = avae
        self.device = torch.device(device)
        self.dtype = dtype

    def setup(self) -> None:
        """No-op: the AVAE sound decoder has no warmup / graph capture."""
        return None

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Sound latent ``[B, latent_ch, T]`` -> waveform ``[B, ch, T*hop]`` in ``[-1, 1]``.

        The cast is load-bearing: unlike the WAN VAE, ``Cosmos3AVAESoundDecoder.decode``
        does not cast its input internally.
        """
        return self.avae.decode(latent.to(self.device, self.dtype))

    @torch.no_grad()
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode is the canonical hot path; ``forward`` aliases :meth:`decode`."""
        return self.decode(latent)

    @property
    def sample_rate(self) -> int:
        """Output waveform sample rate (Hz) of the wrapped AVAE."""
        return self.avae.sample_rate


__all__ = ["Cosmos3VAERunner", "Cosmos3SoundVAERunner"]
