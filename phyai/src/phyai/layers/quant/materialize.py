"""Lower a semantic :class:`QuantScheme` to a physical ``WeightSpec``."""

from __future__ import annotations

from phyai.layers.quant.bf16 import Bf16Spec
from phyai.layers.quant.fp8 import Fp8Spec
from phyai.layers.quant.nvfp4 import Nvfp4Spec
from phyai.layers.quant.scheme import QDType, QuantScheme
from phyai.kernel.device import probe_device
from phyai.kernel.types import DeviceProfile, arch_at_least

#: NVFP4 scale layouts genuinely differ by architecture: Blackwell's tensor
#: cores read a 128x4 swizzled block of scale factors, earlier ones read them
#: linearly.
#:
#: Compared as an *architecture*, not as digits scraped out of a name. The old
#: form stripped non-digits, so ``"h100"`` became 100 and a Hopper card was
#: handed the Blackwell layout -- weights materialized with the wrong scale
#: layout, silently, while every kernel gate read the same string as an unknown
#: device. ``"gfx942"`` became 942 and was held off only by a
#: ``vendor == "nvidia"`` conjunct beside it. Product names are now refused where
#: they enter, so the two readings cannot diverge again.
#:
#: That conjunct is gone on purpose: the ``sm`` series *is* the vendor test.
#: Bare-number architectures are rejected at ``DeviceProfile`` construction,
#: so no digits-only value can reach this comparison at all.
_NVFP4_128X4_MIN_ARCH = "sm100"


def materialize(
    scheme: QuantScheme,
    device: DeviceProfile | str | None = None,
) -> object:
    """Lower semantic quantization into a physical storage specification.

    The device affects only storage layouts that genuinely differ (NVFP4
    linear versus 128x4); execution backend selection is left to the kernel
    resolver.
    """
    w = scheme.weight
    if scheme.online and w.dtype is not QDType.BF16:
        from phyai.layers.quant.humming import HummingSpec

        return HummingSpec(scheme)
    if w.dtype is QDType.BF16:
        return Bf16Spec()
    if w.dtype is QDType.FP8_E4M3:
        return Fp8Spec(granularity=w.granularity, block_shape=w.block_shape)
    if w.dtype is QDType.NVFP4:
        if isinstance(device, DeviceProfile):
            profile = device
        else:
            # Normal torch device strings (notably ``cuda:0``) are probed;
            # the compact ``vendor:arch`` form covers synthetic targets.
            profile = probe_device(device)
        layout = "128x4" if arch_at_least(profile, _NVFP4_128X4_MIN_ARCH) else "linear"
        return Nvfp4Spec(scale_layout=layout)
    raise NotImplementedError(f"materialize: unsupported weight dtype {w.dtype!r}")


__all__ = ["materialize"]
