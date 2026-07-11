"""Lower a semantic :class:`QuantScheme` to a physical ``WeightSpec``."""

from __future__ import annotations

from phyai.layers.quant.bf16 import Bf16Spec
from phyai.layers.quant.fp8 import Fp8Spec
from phyai.layers.quant.granularity import Granularity
from phyai.layers.quant.humming import HummingWeightSpec
from phyai.layers.quant.nvfp4 import Nvfp4Spec
from phyai.layers.quant.scheme import QDType, QuantScheme
from phyai.utils.humming import (
    has_humming,
    humming_supports_sm,
    require_humming_supports_sm,
)

_NVFP4_128X4_MIN_SM = 100

# QDType -> humming canonical dtype string (humming/dtypes.py to_str form).
_HUMMING_WDTYPE = {
    QDType.INT8: "int8",
    QDType.INT6: "int6",
    QDType.INT4: "int4",
    QDType.INT3: "int3",
    QDType.INT2: "int2",
    QDType.FP8_E4M3: "float8e4m3",
    QDType.FP8_E5M2: "float8e5m2",
    QDType.MXFP4: "float4e2m1",
    QDType.FP6_E2M3: "float6e2m3",
    QDType.FP6_E3M2: "float6e3m2",
}
# Weight dtypes humming owns exclusively (flashinfer/torch have no path).
_HUMMING_ONLY = (
    QDType.FP8_E5M2,
    QDType.INT8,
    QDType.INT6,
    QDType.INT4,
    QDType.INT3,
    QDType.INT2,
    QDType.MXFP4,
    QDType.FP6_E2M3,
    QDType.FP6_E3M2,
)
_HUMMING_ADTYPE = {
    QDType.BF16: "bfloat16",
    QDType.INT8: "int8",
    QDType.INT4: "int4",
    QDType.FP8_E4M3: "float8e4m3",
    QDType.FP8_E5M2: "float8e5m2",
    QDType.MXFP4: "float4e2m1",
}
_GRAN_TO_SCALE = {
    Granularity.PER_TENSOR: "tensor",
    Granularity.PER_CHANNEL: "channel",
    Granularity.BLOCK: "block",
}


def _humming_spec(scheme: QuantScheme) -> HummingWeightSpec:
    """Build a :class:`HummingWeightSpec` from a semantic scheme.

    Scale strategy is derived from ``granularity`` + ``group_size``:
    a positive ``group_size`` (AWQ/GPTQ, llm-compressor ``group``) becomes a
    ``group`` scale; ``BLOCK`` becomes a 2-D ``block`` scale; ``PER_TENSOR`` a
    ``tensor`` scale; otherwise ``channel``.

    MXFP4 is the OCP microscaling format: E2M1 weight with an **e8m0 (ue8m0)**
    block scale. That e8m0 scale is what distinguishes it from NVFP4 (E2M1 with
    an e4m3 scale), which is routed separately to ``Nvfp4Spec`` / flashinfer and
    never reaches this function.
    """
    w = scheme.weight
    w_dtype = _HUMMING_WDTYPE[w.dtype]
    a_dtype = (
        _HUMMING_ADTYPE.get(scheme.input.dtype, "bfloat16")
        if scheme.input is not None
        else "bfloat16"
    )
    group_size = int(w.group_size)
    group_size_n = 0
    if w.granularity is Granularity.BLOCK and w.block_shape is not None:
        scale_type = "block"
        group_size_n, group_size = int(w.block_shape[0]), int(w.block_shape[1])
    elif group_size > 0:
        scale_type = "group"
    else:
        scale_type = _GRAN_TO_SCALE[w.granularity]
    # MXFP4's defining microscale is e8m0; this is what separates it from
    # NVFP4 (e4m3 scale, flashinfer-only). Other dtypes leave humming to pick
    # the scale dtype from the schema default.
    scale_dtype = "float8e8m0" if w.dtype is QDType.MXFP4 else None
    return HummingWeightSpec(
        w_dtype=w_dtype,
        a_dtype=a_dtype,
        scale_type=scale_type,
        group_size=group_size,
        group_size_n=group_size_n,
        has_zero_point=not w.symmetric,
        scale_dtype=scale_dtype,
    )


def _quant_backend() -> str:
    """Backend preference for quantized Linear (``PHYAI_LINEAR_QUANT_BACKEND``).

    ``auto`` (default) leans humming for every format humming can run.
    """
    from phyai.env import envs

    backend = (envs.PHYAI_LINEAR_QUANT_BACKEND.get() or "auto").lower()
    if backend not in ("auto", "humming", "flashinfer", "torch"):
        raise ValueError(
            f"PHYAI_LINEAR_QUANT_BACKEND must be auto/humming/flashinfer/torch, got "
            f"{backend!r}"
        )
    return backend


def materialize(scheme: QuantScheme, sm: int) -> object:
    """Return the physical ``WeightSpec`` for ``scheme`` on an SM-``sm`` device.

    humming is a kernel *and* a weight layout; because that layout is fixed at
    load time, the backend choice is made here (not per-forward). The semantic
    ``scheme`` stays backend-agnostic; ``PHYAI_LINEAR_QUANT_BACKEND`` decides which
    physical spec (and thus kernel) serves formats more than one backend can do
    (fp8). Formats only humming can run always go to humming; NVFP4 (e4m3 scale,
    128x4) is flashinfer-only; bf16 is unquantized.
    """
    w = scheme.weight
    if w.dtype is QDType.BF16:
        return Bf16Spec()
    if w.dtype is QDType.NVFP4:
        layout = "128x4" if sm >= _NVFP4_128X4_MIN_SM else "linear"
        return Nvfp4Spec(scale_layout=layout)
    if w.dtype is QDType.FP8_E4M3:
        # Both humming and flashinfer/torch serve fp8; pick per preference.
        # flashinfer/torch -> the standard Fp8Spec layout (dispatcher then routes
        # block->flashinfer, per-tensor/channel->torch). auto/humming -> humming.
        backend = _quant_backend()
        if backend in ("flashinfer", "torch"):
            return Fp8Spec(granularity=w.granularity, block_shape=w.block_shape)
        if backend == "humming":
            require_humming_supports_sm(sm)
        elif has_humming() and not humming_supports_sm(sm):
            # note(chenghua): humming installed but has no kernels for this SM (e.g.
            # Thor sm_110); auto falls back to the flashinfer/torch fp8 layout.
            return Fp8Spec(granularity=w.granularity, block_shape=w.block_shape)
        return _humming_spec(scheme)
    if w.dtype in _HUMMING_ONLY:
        # note(chenghua): no flashinfer/torch path for these dtypes, so an SM humming
        # can't serve has no fallback — fail loudly instead of KeyError'ing in humming.
        require_humming_supports_sm(sm)
        return _humming_spec(scheme)
    raise NotImplementedError(f"materialize: unsupported weight dtype {w.dtype!r}")


__all__ = ["materialize"]
