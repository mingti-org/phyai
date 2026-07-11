"""FlashInferKernel — bf16 (cublasLt/cuDNN/TGV) + block-fp8 groupwise GEMM + NVFP4 GEMM.

flashinfer's ``mm_fp8`` ≠ generic fp8 GEMM: it targets ``trtllm_low_latency``
with pre-processed weights and a single alpha scalar. Per-tensor and
per-channel fp8 therefore stay on :class:`TorchKernel` for now; this kernel
covers the paths where flashinfer is unambiguously the best choice:

* bf16 GEMM on sm≥89 (cuBLASLt / cuDNN / TGV, autoselected by flashinfer);
* block-FP8 (DeepSeek-V3 style) on sm≥100 via ``gemm_fp8_nt_groupwise``;
* NVFP4 on sm≥100 via ``mm_fp4`` with 128x4 scale-factor layout.

If flashinfer is not installed, :meth:`can_handle` returns ``False`` for
every probe and the fallback :class:`TorchKernel` picks up the work.
"""

from __future__ import annotations

from threading import Lock

import torch

from phyai.engine_config import get_engine_config
from phyai.layers.linear.backend import KernelProbe
from phyai.layers.linear.registry import register_linear_kernel
from phyai.parallel.state import Mode, current_mode


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

if _HAS_FLASHINFER:
    _fi_autotune = getattr(flashinfer, "autotune", None)
    try:
        from flashinfer.utils import _get_cache_buf as _fi_get_cache_buf
    except (AttributeError, ImportError):  # pragma: no cover - version dependent
        _fi_get_cache_buf = None  # type: ignore[assignment]
else:
    _fi_autotune = None  # type: ignore[assignment]
    _fi_get_cache_buf = None  # type: ignore[assignment]


# TinyGEMM is a latency kernel for tiny row counts (FlashInfer documents an
# ideal range of 1-8). Letting ``auto`` profile it at vision/LLM row counts can
# take minutes for one shape. Large-M problems still tune exact cuDNN tactics.
_AUTO_TINYGEMM_MAX_M = 8
_BF16_AUTOTUNE_WORKSPACE_BYTES = 64 * 1024 * 1024


@register_linear_kernel(
    prefer_for={
        ("bf16", "prefill"),
        ("fp8_block_128_128", "prefill"),
        ("fp8_block_128_128", "decode"),
        ("nvfp4_block_16_128x4", "prefill"),
        ("nvfp4_block_16_128x4", "decode"),
    },
)
class FlashInferKernel:
    """bf16 + block-fp8 + NVFP4 via flashinfer.gemm.

    For block-fp8 we assume DeepSeek-V3 style weight layout:
    ``layer.weight`` is ``(N, K)`` fp8_e4m3fn, ``layer.weight_scale`` is
    ``(N // bn, K // bk)`` fp32, and ``x`` gets rowwise-quantised to fp8
    with a ``(M, K // bk)`` scale tensor by :meth:`spec.quantize_activation`.

    For NVFP4, ``layer.weight`` is packed ``(N, K // 2)`` uint8,
    ``layer.weight_scale`` uses FlashInfer's 128x4 layout, and
    ``layer.weight_global_scale`` is the per-tensor descale factor.

    ``prefer_for`` is attached at decoration time and consulted by
    :class:`phyai.layers.linear.registry.LinearKernelRegistry` —
    everything else falls through to registration order.
    """

    name = "flashinfer"

    def __init__(self) -> None:
        self._bf16_tuned_shapes: set[tuple[object, ...]] = set()
        self._bf16_tune_lock = Lock()

    def supports_capture(self) -> bool:
        # First-call concerns (JIT, cudnn handle init, backend heuristic)
        # all happen during ``CudaGraph.capture``'s side-stream warmup
        # iterations and are gone by the time we enter the capture region.
        # The captured kernel is a single cuDNN / cuBLAS / cutlass / tgv
        # matmul launch — the Python wrapper's per-call overhead also
        # disappears inside the graph (it only runs at capture time).
        # On sm_100+ this is the only way to land on the cutlass / tgv
        # paths from inside captured runners; on sm_90 it's neutral.
        return True

    def can_handle(self, probe: KernelProbe) -> bool:
        if not _HAS_FLASHINFER:
            return False
        if probe.spec_id == "bf16":
            # cuBLASLt/cuDNN paths cover sm80+; flashinfer's own heuristic
            # picks the right backend at call time.
            return probe.sm >= 80 and probe.in_dtype == torch.bfloat16
        if probe.spec_id.startswith("fp8_block_"):
            # gemm_fp8_nt_groupwise is sm100+ only today.
            return probe.sm >= 100
        if probe.spec_id == "nvfp4_block_16_128x4":
            return probe.sm >= 100 and probe.K % 16 == 0
        return False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        spec = layer.spec
        if spec.spec_id == "bf16":
            return self._bf16(layer, x, bias)
        if spec.spec_id.startswith("fp8_block_"):
            return self._block_fp8(layer, x, bias)
        if spec.spec_id == "nvfp4_block_16_128x4":
            return self._nvfp4(layer, x, bias)
        raise RuntimeError(f"FlashInferKernel got unhandled spec_id={spec.spec_id!r}")

    # ------------------------------------------------------------------
    # bf16: mm_bf16(a (M,K) row, b (K,N) col, bias (N,))
    # ------------------------------------------------------------------

    def _bf16(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        assert _fi_gemm is not None
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        runtime = get_engine_config().runtime
        if runtime.flashinfer_autotune and x_2d.is_cuda:
            if _fi_get_cache_buf is None:
                raise RuntimeError(
                    "flashinfer_autotune requires FlashInfer's GEMM "
                    "workspace cache API."
                )
            # cuDNN tactics may grow FlashInfer's default 32 MiB buffer by a
            # few bytes.  Reserve once before any graph is captured so the
            # workspace shape stays in the autotuner key and its storage
            # pointer remains stable across all PI0.5 graph captures.
            _fi_get_cache_buf(
                "mm_bf16_workspace",
                _BF16_AUTOTUNE_WORKSPACE_BYTES,
                x_2d.device,
            )
        configured_backend = runtime.flashinfer_bf16_backend
        backend = (
            "cudnn"
            if configured_backend == "auto" and x_2d.shape[0] > _AUTO_TINYGEMM_MAX_M
            else configured_backend
        )
        tune_key = (
            backend,
            x_2d.device.type,
            x_2d.device.index,
            tuple(x_2d.shape),
            tuple(x_2d.stride()),
            tuple(layer.weight.shape),
            tuple(layer.weight.stride()),
            x_2d.dtype,
            bias is not None,
        )

        def run() -> torch.Tensor:
            return _fi_gemm.mm_bf16(
                x_2d,
                layer.weight.t(),
                bias=bias,
                out_dtype=x.dtype,
                backend=backend,
            )

        def run_tuned(*, tune_mode: bool) -> torch.Tensor:
            assert _fi_autotune is not None
            with _fi_autotune(
                tune_mode,
                tuning_buckets=(int(x_2d.shape[0]),),
            ):
                return run()

        if not runtime.flashinfer_autotune:
            y = run()
        elif current_mode() is Mode.GRAPH_CAPTURING:
            if tune_key not in self._bf16_tuned_shapes:
                raise RuntimeError(
                    "FlashInfer BF16 GEMM reached CUDA-graph capture before its "
                    f"shape was tuned: M={x_2d.shape[0]}, N={layer.weight.shape[0]}, "
                    f"K={x_2d.shape[1]}, backend={backend!r}. Run at least one "
                    "eager warmup iteration before capture."
                )
            y = run_tuned(tune_mode=False)
        elif tune_key in self._bf16_tuned_shapes:
            y = run_tuned(tune_mode=False)
        else:
            if _fi_autotune is None:
                raise RuntimeError(
                    "flashinfer_autotune requires a FlashInfer release "
                    "that exports flashinfer.autotune."
                )
            with self._bf16_tune_lock:
                if tune_key in self._bf16_tuned_shapes:
                    y = run_tuned(tune_mode=False)
                else:
                    y = run_tuned(tune_mode=True)
                    self._bf16_tuned_shapes.add(tune_key)
        return y.reshape(*x.shape[:-1], -1)

    # ------------------------------------------------------------------
    # block-fp8: gemm_fp8_nt_groupwise
    # ------------------------------------------------------------------

    def _block_fp8(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        assert _fi_gemm is not None
        spec = layer.spec
        assert spec.block_shape is not None
        bn, bk = spec.block_shape
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
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

    def _nvfp4(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        assert _fi_gemm is not None
        assert _fi_nvfp4_quantize is not None
        assert _FiSfLayout is not None
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        x_global_scale = (
            448.0 * 6.0
        ) / x_2d.float().abs().nan_to_num().max().clamp_min(1e-12)
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
