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
        """Latents ``[B, z, t, h, w]`` -> pixels ``[B, 3, T, H, W]`` in ``[-1, 1]``.

        Under tensor / CFG parallelism (``cfg_size * tp_size > 1``) the WAN VAE is
        replicated on every rank and the latent is identical across ranks, so the
        decode payload is split spatially across all of them
        (:meth:`Cosmos3WanVAE.decode_parallel`) and the blended frame is identical
        on every rank. At combined parallel size 1 this is the byte-identical
        single-card :meth:`Cosmos3WanVAE.decode`.
        """
        x = latents.to(self.device, self.dtype)
        if self._decode_parallel_size() > 1:
            return self.vae.decode_parallel(x)
        return self.vae.decode(x)

    @staticmethod
    def _decode_parallel_size() -> int:
        """Product of the axes the decode splits across (``cfg_size * tp_size``).

        Each axis defaults to 1 when absent / no mesh, so this is 1 on a single
        card and the runner routes to the plain :meth:`Cosmos3WanVAE.decode`.
        """
        import phyai.parallel as P

        def _size(name: str) -> int:
            try:
                return P.default_mesh().axis_size(name)
            except Exception:
                return 1

        return _size("cfg") * _size("tp")

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
