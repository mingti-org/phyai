"""Channel L2 normalization with an affine scale for image and video features."""

from __future__ import annotations

import torch
import torch.nn as nn

from phyai.engine_config import get_engine_config
from phyai.kernel.call import CallSite, backend_preference
from phyai.weights.shards import replicated


class ChannelL2Norm(nn.Module):
    """Compute ``normalize(x) * sqrt(channels) * gamma + bias``.

    ``eps`` clamps the L2 norm before division. ``compute_dtype`` controls the
    reduction precision; normalized values return to the input dtype before
    applying the scale and affine parameters.
    """

    def __init__(
        self,
        channels: int,
        *,
        spatial_dims: int = 2,
        channel_first: bool = True,
        eps: float = 1e-12,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        compute_dtype: torch.dtype | None = torch.float32,
        prefix: str = "",
        backend: str | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0 or spatial_dims < 0 or eps <= 0:
            raise ValueError(
                "channels and eps must be positive; spatial_dims must be nonnegative"
            )
        self.channels = channels
        self.spatial_dims = spatial_dims
        self.channel_first = channel_first
        self.eps = eps
        self.scale = channels**0.5
        self.compute_dtype = compute_dtype
        self._call = CallSite(
            "channel_l2_norm",
            role="norm.channel_l2",
            prefer=backend_preference("channel_l2_norm", backend),
            dims={"hidden": channels},
            attrs={"channel_first": channel_first, "bias": bias},
        )
        if device is None:
            device = get_engine_config().device.target
        shape = (channels,) + (1,) * spatial_dims if channel_first else (channels,)
        self.gamma = nn.Parameter(
            torch.ones(shape, dtype=dtype, device=device), requires_grad=False
        )
        self.bias = (
            nn.Parameter(
                torch.zeros(shape, dtype=dtype, device=device), requires_grad=False
            )
            if bias
            else None
        )
        if prefix:
            self.gamma.hf_keys = [(f"{prefix}.gamma", None)]
            self.gamma.weight_loader = replicated()
            if self.bias is not None:
                self.bias.hf_keys = [(f"{prefix}.bias", None)]
                self.bias.weight_loader = replicated()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_compatible = (
            x.ndim >= 2
            and x.shape[1 if self.channel_first else -1] == self.channels
            and (not self.channel_first or x.ndim == self.spatial_dims + 2)
            and x.is_contiguous()
            and self.gamma.is_contiguous()
            and x.dtype == self.gamma.dtype
            and self.compute_dtype in (None, torch.float32, x.dtype)
            and (
                self.bias is None
                or (self.bias.is_contiguous() and self.bias.dtype == x.dtype)
            )
        )
        handle = self._call.select(
            device=x.device,
            dtype={
                "input": x.dtype,
                "weight": self.gamma.dtype,
                "compute": self.compute_dtype or x.dtype,
                "bias": None if self.bias is None else self.bias.dtype,
            },
            dims={"tokens": x.numel() // self.channels},
            attrs={"fused_compatible": fused_compatible},
        )
        return handle.execute(
            x,
            self.gamma,
            self.bias,
            self.eps,
            channel_first=self.channel_first,
            compute_dtype=self.compute_dtype,
        )


__all__ = ["ChannelL2Norm"]
