"""Runner-owned selection for a separately rounded residual addition."""

from __future__ import annotations
import torch
from phyai.kernel.call import CallSite, backend_preference, token_shape


class Cosmos3ResidualNorm:
    def __init__(self):
        self.call = CallSite("rmsnorm_rounded_add")
        self.plain_calls = {}

    def __call__(self, x, residual, norm):
        # Fuse only the plain FlashInfer norm whose reduction the kernel preserves.
        key = norm.backend
        if key not in self.plain_calls:
            self.plain_calls[key] = CallSite(
                "rmsnorm", prefer=backend_preference("rmsnorm", key)
            )
        plain = self.plain_calls[key].select(
            device=x.device,
            dtype={"input": x.dtype, "weight": norm.weight.dtype, "output": x.dtype},
            dims=token_shape(x, hidden=x.shape[-1]),
            attrs={"variant": norm.variant, "rank": 2},
        )
        if plain.kernel_id != "flashinfer.rmsnorm" or norm.cast_before_affine:
            summed = residual + x
            return norm(summed), summed
        selected = self.call.select(
            device=x.device,
            dtype={
                "input": x.dtype,
                "weight": norm.weight.dtype,
                "residual": residual.dtype,
            },
            dims=token_shape(x, hidden=x.shape[-1]),
            attrs={"variant": norm.variant},
        )
        return selected.execute(x, residual, norm)
