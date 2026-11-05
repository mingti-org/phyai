"""Independent-frame observation encoding with HumanWAM-Goal latent normalization."""

import torch

from phyai.models.cosmos3.vae_wan import Cosmos3WanVAE, patchify
from phyai.models.cosmos3.configuration_cosmos3 import Cosmos3WanVAEConfig


class HumanWAMGoalVAE(Cosmos3WanVAE):
    def __init__(
        self, config: Cosmos3WanVAEConfig, *, params_dtype: torch.dtype
    ) -> None:
        super().__init__(config)
        self.register_buffer(
            "latents_inv_std", 1.0 / self.latents_std.to(params_dtype), persistent=False
        )

    def encode(
        self,
        pixels: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if sample or generator is not None:
            raise ValueError("HumanWAM-Goal conditions use the VAE posterior mean")
        if pixels.ndim != 5 or pixels.shape[0:3] != (1, 3, 1):
            raise ValueError(
                "Encode each HumanWAM-Goal observation as one [1,3,1,H,W] frame"
            )
        # Capture and eager encoding use the same deterministic convolution
        # choices, without changing the caller's cuDNN settings permanently.
        with torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False
        ):
            features = self.encoder(
                patchify(pixels, self.patch_size),
                feat_cache=[None] * self._enc_conv_num,
                feat_idx=[0],
            )
            mean, _ = self.quant_conv(features).chunk(2, dim=1)
        # The reciprocal is rounded before multiplication in the source VAE.
        return (mean - self.latents_mean.to(mean)) * self.latents_inv_std.to(mean)
