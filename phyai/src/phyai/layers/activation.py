"""Activation layers"""

from __future__ import annotations

import torch
import torch.nn as nn

from phyai.engine_config import get_engine_config
from phyai.weights.shards import replicated


class Snake1d(nn.Module):
    """Learnable 1-D Snake activation: ``x + (exp(beta)+eps)^-1 * sin(exp(alpha)·x)^2``."""

    def __init__(
        self,
        hidden_dim: int,
        logscale: bool = True,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if device is None:
            device = get_engine_config().device.target
        self.logscale = logscale
        self.alpha = nn.Parameter(
            torch.zeros(1, hidden_dim, 1, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.beta = nn.Parameter(
            torch.zeros(1, hidden_dim, 1, dtype=dtype, device=device),
            requires_grad=False,
        )
        if prefix:
            self.alpha.hf_keys = [(f"{prefix}.alpha", None)]
            self.alpha.weight_loader = replicated()
            self.beta.hf_keys = [(f"{prefix}.beta", None)]
            self.beta.weight_loader = replicated()

        # Inference-time per-channel constants exp(alpha) and 1/(exp(beta)+eps),
        # baked once by post_load() (called by load_pretrained). None until then;
        # forward falls back to computing them from the parameters when unset.
        self.register_buffer("_alpha_eff", None, persistent=False)
        self.register_buffer("_beta_recip", None, persistent=False)

    def post_load(self) -> None:
        """Bake the per-channel constants once the parameters are loaded."""
        alpha = self.alpha.exp() if self.logscale else self.alpha.clone()
        beta = self.beta.exp() if self.logscale else self.beta.clone()
        self._alpha_eff = alpha.detach()
        self._beta_recip = (beta + 1e-9).reciprocal().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_eff, beta_recip = self._alpha_eff, self._beta_recip
        if alpha_eff is None:
            alpha_eff = torch.exp(self.alpha) if self.logscale else self.alpha
            beta = torch.exp(self.beta) if self.logscale else self.beta
            beta_recip = (beta + 1e-9).reciprocal()
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + beta_recip * torch.sin(alpha_eff * x).pow(2)
        return x.reshape(shape)


__all__ = ["Snake1d"]
