"""HummingKernel — dense GEMM for humming-backed quant specs."""

from __future__ import annotations

import torch

from phyai.layers.linear.backend import KernelProbe
from phyai.layers.linear.registry import register_linear_kernel
from phyai.utils.humming import humming_supports_sm

try:  # pragma: no cover - depends on optional install + CUDA toolchain
    import humming  # noqa: F401
    from humming.layer import HummingMethod

    _HAS_HUMMING = True
except Exception:  # pragma: no cover - the common CPU/dev path
    HummingMethod = None  # type: ignore[assignment]
    _HAS_HUMMING = False


# SM requirement is set by the ACTIVATION (compute) dtype, mirroring humming's
# kernel/humming.py check_dtype (``assert sm >= dtype_map[a_dtype]``). The weight
# dtype only has bit constraints (num_bits<=8 and <=activation bits), no SM gate —
# a low-bit weight is dequantized to the activation dtype for the mma. So e.g. an
# fp8/mxfp4 *weight* with bf16 activation runs on sm80, while sm120 is only needed
# when the *activation* itself is fp4.
_ACT_SM_GATE = {
    "int8": 75,
    "float16": 75,
    "int4": 80,
    "bfloat16": 80,
    "float8e4m3": 89,
    "float8e5m2": 89,
    "float4e2m1": 120,
}


def _parse_dtypes(spec_id: str) -> tuple[str, str]:
    """Pull (weight_dtype, activation_dtype) out of a ``humming_w<w>_a<a>_...`` id.

    humming dtype strings contain no underscore (``int4``, ``float8e4m3``,
    ``bfloat16``), so splitting on ``_`` is unambiguous.
    """
    parts = spec_id.split("_")
    w = parts[1][1:] if len(parts) > 1 and parts[1].startswith("w") else ""
    a = parts[2][1:] if len(parts) > 2 and parts[2].startswith("a") else ""
    return w, a


@register_linear_kernel()
class HummingKernel:
    """Dense GEMM via humming's ``HummingMethod.forward_layer``."""

    name = "humming"

    def supports_capture(self) -> bool:
        # humming JIT-compiles a cubin per (config, tuning) on first call; the
        # captured shape is warmed on the main thread before capture (same as
        # flashinfer), so the compile happens outside the capture region.
        return True

    def can_handle(self, probe: KernelProbe) -> bool:
        if not _HAS_HUMMING:
            return False
        if not probe.spec_id.startswith("humming_"):
            return False
        # note(chenghua): humming KeyErrors on SMs absent from its heuristics_map
        # (e.g. Thor sm_110), so decline — a pre-packed humming checkpoint then
        # fails with a clean "no kernel" error instead of a KeyError mid-forward.
        if not humming_supports_sm(probe.sm):
            return False
        _w_dtype, a_dtype = _parse_dtypes(probe.spec_id)
        # Gate on the activation dtype only (see _ACT_SM_GATE). Unknown/unsupported
        # activation dtype -> not handled.
        need = _ACT_SM_GATE.get(a_dtype)
        if need is None:
            return False
        return probe.sm >= need

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        y = HummingMethod.forward_layer(
            layer,
            inputs=x_2d,
            compute_config=layer._humming_spec._compute_config,
        )
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], -1)


__all__ = ["HummingKernel"]
