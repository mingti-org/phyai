"""Normalization layers for LingBot V2."""

import torch
from torch import nn

from phyai.weights.shards import replicated
from phyai_kernel.triton.lingbot_norm import lingbot_layernorm, lingbot_rmsnorm


class LingBotV2RMSNorm(nn.Module):
    """Round normalized activations before multiplying the RMSNorm weight."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=False
        )
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return lingbot_rmsnorm(x, self.weight, self.variance_epsilon)


class LingBotV2LayerNorm(nn.Module):
    """Use two-pass statistics and separate FP32 affine operations."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_size, dtype=dtype, device=device), requires_grad=False
        )
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()
            self.bias.hf_keys = [(f"{prefix}.bias", None)]
            self.bias.weight_loader = replicated()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return lingbot_layernorm(x, self.weight, self.bias, self.variance_epsilon)
