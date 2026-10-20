"""Numerical-equivalence tests for the Triton FP32 Router GEMM."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import phyai_kernel
import phyai_kernel.triton.router_gemm as triton_router_gemm


if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for phyai-kernel Triton tests", allow_module_level=True
    )


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    # fp32_router_gemm intentionally avoids TF32.  Compare it with an IEEE
    # FP32 PyTorch reference even when the surrounding application (including
    # LingBot) selected torch's "high" matmul mode.
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        return F.linear(
            x.to(torch.float32),
            weight.to(torch.float32),
            bias,
        )
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def test_module_exposes_fp32_router_gemm() -> None:
    assert phyai_kernel.fp32_router_gemm is triton_router_gemm.fp32_router_gemm


@pytest.mark.parametrize(
    ("input_shape", "num_experts"),
    [
        ((1, 768), 32),
        ((7, 768), 32),
        ((33, 768), 32),
        ((2, 5, 768), 32),
        ((11, 127), 17),
        ((4, 1024), 65),
    ],
)
@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.bfloat16, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_fp32_router_gemm_matches_reference(
    input_shape: tuple[int, ...],
    num_experts: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    with_bias: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(0xC0DE)
    hidden_size = input_shape[-1]
    x = torch.randn(
        input_shape,
        dtype=input_dtype,
        device="cuda",
        generator=generator,
    )
    weight = torch.randn(
        (num_experts, hidden_size),
        dtype=weight_dtype,
        device="cuda",
        generator=generator,
    )
    bias = (
        torch.randn(
            num_experts,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        if with_bias
        else None
    )

    actual = phyai_kernel.fp32_router_gemm(x, weight, bias)
    expected = _reference(x, weight, bias)

    assert actual.shape == input_shape[:-1] + (num_experts,)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)


def test_fp32_router_gemm_out_argument() -> None:
    x = torch.randn(4, 768, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda")
    out = torch.empty(4, 32, dtype=torch.float32, device="cuda")

    returned = phyai_kernel.fp32_router_gemm(x, weight, out=out)

    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        returned,
        _reference(x, weight, None),
        atol=5e-4,
        rtol=2e-4,
    )


def test_fp32_weight_values_are_not_rounded_to_bfloat16() -> None:
    x = torch.ones(1, 768, dtype=torch.bfloat16, device="cuda")
    weight = torch.full(
        (32, 768),
        1.001,
        dtype=torch.float32,
        device="cuda",
    )

    actual = phyai_kernel.fp32_router_gemm(x, weight)
    expected = _reference(x, weight, None)
    bf16_rounded = _reference(x, weight.to(torch.bfloat16), None)

    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)
    assert torch.max(torch.abs(actual - bf16_rounded)).item() > 0.5


def test_fp32_router_gemm_accepts_non_contiguous_input() -> None:
    full = torch.randn(8, 2, 768, dtype=torch.bfloat16, device="cuda")
    x = full[:, 0, :]
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda")
    assert not x.is_contiguous()

    actual = phyai_kernel.fp32_router_gemm(x, weight)
    expected = _reference(x, weight, None)

    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=2e-4)


def test_fp32_router_gemm_empty_input() -> None:
    x = torch.empty(0, 768, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda")

    out = phyai_kernel.fp32_router_gemm(x, weight)

    assert out.shape == (0, 32)
    assert out.dtype == torch.float32


def test_fp32_router_gemm_rejects_non_cuda() -> None:
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    weight = torch.randn(4, 16, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="must live on CUDA"):
        phyai_kernel.fp32_router_gemm(x, weight)


def test_fp32_router_gemm_rejects_shape_mismatch() -> None:
    x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 8, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="must match weight"):
        phyai_kernel.fp32_router_gemm(x, weight)


def test_fp32_router_gemm_rejects_fp16() -> None:
    x = torch.randn(2, 16, dtype=torch.float16, device="cuda")
    weight = torch.randn(4, 16, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="bfloat16 or float32"):
        phyai_kernel.fp32_router_gemm(x, weight)


def test_fp32_router_gemm_rejects_non_fp32_bias() -> None:
    x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4, 16, dtype=torch.float32, device="cuda")
    bias = torch.randn(4, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="bias must be float32"):
        phyai_kernel.fp32_router_gemm(x, weight, bias)


# --------------------------------------------------------------------------- #
# Storage aliasing (review: `out` may be aliased onto an input)
#
# The contract is enforced by validation, so it is asserted there directly.
# That also keeps this evidence independent of kernel compilation: this host's
# bundled ptxas rejects the device's `sm_110a` target, which makes every
# kernel-launch test in this file fail for environmental reasons.
# --------------------------------------------------------------------------- #


def _validate(x, weight, bias=None, out=None) -> None:
    triton_router_gemm._check_inputs(x, weight, bias, out)


def test_check_inputs_rejects_out_sharing_weight_storage() -> None:
    """Square fp32 weights let ``out is weight`` pass every other check.

    The kernel stores ``out`` while other programs are still loading
    ``weight``, so accepting this silently corrupts both the logits and the
    weights.  Program ordering does not help: one program's store can land
    before another program's load.
    """

    size = 4
    x = torch.randn(size, size, dtype=torch.float32, device="cuda")
    weight = torch.randn(size, size, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="share storage with `weight`"):
        _validate(x, weight, None, weight)


def test_check_inputs_rejects_out_sharing_input_storage() -> None:
    x = torch.randn(1, 4, dtype=torch.float32, device="cuda")
    weight = torch.randn(4, 4, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="must not share storage"):
        _validate(x, weight, None, x)


def test_check_inputs_rejects_partially_overlapping_out() -> None:
    """A view into ``weight`` overlaps without being the same tensor."""

    x = torch.randn(1, 8, dtype=torch.float32, device="cuda")
    weight = torch.randn(8, 8, dtype=torch.float32, device="cuda")
    overlapping = weight.reshape(-1)[2:10].reshape(1, 8)

    assert overlapping.data_ptr() != weight.data_ptr()
    assert (
        overlapping.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
    )

    with pytest.raises(RuntimeError, match="must not share storage"):
        _validate(x, weight, None, overlapping)


def test_check_inputs_rejects_out_sharing_bias_storage() -> None:
    x = torch.randn(4, dtype=torch.float32, device="cuda")
    weight = torch.randn(4, 4, dtype=torch.float32, device="cuda")
    # `bias` and `out` share one allocation that touches neither input, so
    # the `bias` entry in the guard is the one that has to fire.  Aliasing
    # `bias` onto `weight` would let the earlier `weight` entry raise first
    # and leave this branch untested.
    bias_pool = torch.empty(8, dtype=torch.float32, device="cuda")
    bias = bias_pool[:4]
    out = bias_pool[4:]

    with pytest.raises(RuntimeError, match="share storage with `bias`"):
        _validate(x, weight, bias, out)


def test_check_inputs_conservatively_rejects_disjoint_same_allocation_out() -> None:
    """Documented cost of the conservative rule.

    ``out`` here is provably disjoint from ``weight``, but both live in one
    allocation and are therefore refused.  Proving disjointness instead means
    reasoning about the element ranges each input actually touches, which
    strided inputs make unreliable; callers should pass a separate buffer.
    """

    combined = torch.empty(128, dtype=torch.float32, device="cuda")
    weight = combined[:64].reshape(8, 8)
    disjoint_in_same_allocation = combined[64:72].reshape(1, 8)
    x = torch.randn(1, 8, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="must not share storage"):
        _validate(x, weight, None, disjoint_in_same_allocation)


def test_check_inputs_accepts_a_separate_out_buffer() -> None:
    """The ordinary case must keep working."""

    x = torch.randn(4, 768, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda")
    out = torch.empty(4, 32, dtype=torch.float32, device="cuda")

    _validate(x, weight, None, out)


def test_fp32_router_gemm_rejects_aliased_out_before_launching() -> None:
    """The public entry point must refuse before any kernel work."""

    size = 4
    x = torch.randn(size, size, dtype=torch.float32, device="cuda")
    weight = torch.randn(size, size, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="must not share storage"):
        phyai_kernel.fp32_router_gemm(x, weight, out=weight)


# --------------------------------------------------------------------------- #
# Launch device context (review: must launch on the input's device)
# --------------------------------------------------------------------------- #


def test_fp32_router_gemm_launches_inside_a_device_context(monkeypatch) -> None:
    """The launch must be issued inside ``x.device``.

    The Triton kernel is stubbed, so this asserts the context the launch is
    issued in rather than the kernel itself.  True multi-device behaviour --
    a caller on ``cuda:0`` feeding tensors on ``cuda:1`` -- needs two GPUs and
    is NOT VERIFIED here.
    """

    entered: list = []
    real_device_cls = torch.cuda.device

    # Subclass rather than replace: torch calls isinstance(..., device)
    # internally, which a plain function would break.
    class _RecordingDevice(real_device_cls):
        def __init__(self, target=None):
            entered.append(target)
            super().__init__(target)

    launched: dict = {}

    class _StubKernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                launched["device"] = torch.cuda.current_device()

            return _launch

    monkeypatch.setattr(torch.cuda, "device", _RecordingDevice)
    monkeypatch.setattr(triton_router_gemm, "_fp32_router_gemm_kernel", _StubKernel())

    x = torch.randn(4, 768, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda")
    device_before = torch.cuda.current_device()

    triton_router_gemm.fp32_router_gemm(x, weight)

    assert launched, "the kernel was never launched"
    cuda_targets = [
        target for target in entered if getattr(target, "type", None) == "cuda"
    ]
    assert cuda_targets, f"no CUDA device context was entered: {entered}"
    assert launched["device"] == x.device.index
    assert torch.cuda.current_device() == device_before


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason=(
        "needs two CUDA devices; NOT VERIFIED on this machine (1 GPU). "
        "This test has never executed."
    ),
)
def test_fp32_router_gemm_launches_on_the_input_device() -> None:
    """With two devices the launch must target ``x.device``, not the caller's."""

    torch.cuda.set_device(0)
    device_before = torch.cuda.current_device()

    x = torch.randn(4, 768, dtype=torch.bfloat16, device="cuda:1")
    weight = torch.randn(32, 768, dtype=torch.float32, device="cuda:1")

    actual = phyai_kernel.fp32_router_gemm(x, weight)
    torch.cuda.synchronize(1)

    assert torch.cuda.current_device() == device_before
    assert actual.device == x.device
    torch.testing.assert_close(
        actual, _reference(x, weight, None), atol=5e-4, rtol=2e-4
    )
