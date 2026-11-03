"""FlashInfer GEMMs — bf16 (cublasLt/cuDNN/TGV), block-fp8 groupwise, NVFP4.

flashinfer's ``mm_fp8`` ≠ generic fp8 GEMM: it targets ``trtllm_low_latency``
with pre-processed weights and a single alpha scalar. Per-tensor and
per-channel fp8 therefore stay on the torch backend for now; this module
covers the paths where flashinfer is unambiguously the best choice:

* bf16 GEMM on sm≥89 (cuBLASLt / cuDNN / TGV, autoselected by flashinfer);
* block-FP8 (DeepSeek-V3 style) on sm≥100 via ``gemm_fp8_nt_groupwise``;
* NVFP4 on sm≥100 via ``mm_fp4`` with 128x4 scale-factor layout.

If flashinfer is not installed, ``lib.flashinfer`` is false and every row
gated on it is ineligible, so the torch backend picks up the work.

Weight layouts these functions assume:

* **block-fp8**, DeepSeek-V3 style — ``layer.weight`` is ``(N, K)`` fp8_e4m3fn,
  ``layer.weight_scale`` is ``(N // bn, K // bk)`` fp32, and ``x`` is
  rowwise-quantised to fp8 with a ``(M, K // bk)`` scale tensor by
  ``spec.quantize_activation``.
* **NVFP4** — ``layer.weight`` is packed ``(N, K // 2)`` uint8,
  ``layer.weight_scale`` uses FlashInfer's 128x4 layout, and
  ``layer.weight_global_scale`` is the per-tensor descale factor.
"""

from __future__ import annotations

import torch


try:
    import flashinfer  # noqa: F401
    import flashinfer.gemm as _fi_gemm
    from flashinfer.quantization import SfLayout as _FiSfLayout
    from flashinfer.quantization import nvfp4_quantize as _fi_nvfp4_quantize

    _HAS_FLASHINFER = True
except Exception:  # pragma: no cover — depends on install
    _fi_gemm = None  # type: ignore[assignment]
    _FiSfLayout = None  # type: ignore[assignment]
    _fi_nvfp4_quantize = None  # type: ignore[assignment]
    _HAS_FLASHINFER = False


# --------------------------------------------------------------------------- #
# GEMM entry points
# --------------------------------------------------------------------------- #
#
# Module-level functions rather than methods on a namespace class: they never
# touched ``self``, so the class only grouped them -- at the cost of a
# stringly-typed ``importlib`` + two ``getattr`` calls wherever they were bound,
# and of tests instantiating a stateless object to reach a private method.
#
# Which of these runs, and on what hardware, is declared in
# :mod:`phyai.kernel.ops.gemm`: one catalog row per storage format, each naming
# its own eligibility conditions.


def _flatten_activations(x: torch.Tensor, K: int) -> torch.Tensor:
    """Flatten activations to a dense row-major ``(M, K)``.

    The FlashInfer GEMM and quantization kernels assume a dense row-major A
    and silently read the wrong values on any other layout. ``reshape``
    returns a same-shape *view* for 2-D inputs, so an expanded or otherwise
    strided activation (e.g. ``cond.expand(M, -1)`` feeding an AdaRMS dense)
    would reach the kernels un-copied without this.
    """
    x_2d = x.reshape(-1, K)
    return x_2d if x_2d.is_contiguous() else x_2d.contiguous()


def gemm_bf16(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_dtype: torch.dtype | None = None,
    backend: str = "cudnn",
) -> torch.Tensor:
    assert _fi_gemm is not None
    K = x.shape[-1]
    x_2d = _flatten_activations(x, K)
    out_dtype = out_dtype or x.dtype
    if x_2d.shape[0] == 0:
        return x.new_empty((*x.shape[:-1], layer.weight.shape[0]), dtype=out_dtype)
    fused_bias = (
        bias
        if out_dtype == x.dtype and backend in {"cudnn", "tgv", "tinygemm", "auto"}
        else None
    )
    # weight is (N, K) row-major; ``.t()`` is the (K, N) column-major view.
    y = _fi_gemm.mm_bf16(
        x_2d,
        layer.weight.t(),
        bias=fused_bias,
        out_dtype=out_dtype,
        backend=backend,
    )
    if bias is not None and fused_bias is None:
        y = y + bias
    return y.reshape(*x.shape[:-1], layer.weight.shape[0])


# ------------------------------------------------------------------
# block-fp8: gemm_fp8_nt_groupwise
# ------------------------------------------------------------------


def gemm_block_fp8(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    assert _fi_gemm is not None
    spec = layer.spec
    assert spec.block_shape is not None
    bn, bk = spec.block_shape
    K = x.shape[-1]
    x_2d = _flatten_activations(x, K)
    # Per-token rowwise fp8 activation; spec handles the scale shape.
    act = spec.quantize_activation(x_2d, layer)
    # groupwise GEMM: a (m, k) row-major, b (n, k) col-major.
    y = _fi_gemm.gemm_fp8_nt_groupwise(
        act.x,
        layer.weight,
        a_scale=act.x_scale.reshape(-1),
        b_scale=layer.weight_scale,
        scale_granularity_mnk=(1, bn, bk),
        out_dtype=x.dtype,
    )
    if bias is not None:
        y = y + bias
    return y.reshape(*x.shape[:-1], -1)


# ------------------------------------------------------------------
# nvfp4: mm_fp4 with 128x4 scale-factor layout
# ------------------------------------------------------------------


def gemm_nvfp4(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    assert _fi_gemm is not None
    assert _fi_nvfp4_quantize is not None
    assert _FiSfLayout is not None
    K = x.shape[-1]
    x_2d = _flatten_activations(x, K)
    x_global_scale = (448.0 * 6.0) / x_2d.float().abs().nan_to_num().max().clamp_min(
        1e-12
    )
    x_global_scale = x_global_scale.reshape(1).to(torch.float32)
    act_x, act_scale = _fi_nvfp4_quantize(
        x_2d,
        x_global_scale,
        sfLayout=_FiSfLayout.layout_128x4,
        do_shuffle=False,
        enable_pdl=False,
    )
    alpha = (layer.weight_global_scale / x_global_scale).to(torch.float32)
    y = _fi_gemm.mm_fp4(
        act_x,
        layer.weight.t(),
        act_scale,
        layer.weight_scale.t().view(torch.uint8),
        alpha,
        x.dtype,
        None,
        block_size=16,
        use_nvfp4=True,
        backend="cudnn",
    )
    if bias is not None:
        y = y + bias
    return y.reshape(*x.shape[:-1], -1)


__all__ = ["gemm_bf16", "gemm_block_fp8", "gemm_nvfp4"]
