"""Paired full-projection normalization and rotary precision/layout contracts."""

import pytest
import torch

from phyai_kernel import qk_rmsnorm_rope, rmsnorm_cast_affine
from phyai_kernel.triton.rotary_embedding import apply_rope_precomputed


def reference(q, k, qw, kw, cos, sin, head_dim, interleave, start, heads):
    def norm(x, weight):
        value = x.float()
        normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        return (
            (normalized.to(x.dtype) * weight)
            .unflatten(-1, (-1, head_dim))
            .narrow(-2, start, heads)
        )

    return apply_rope_precomputed(norm(q, qw), norm(k, kw), cos, sin, interleave)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize(
    "width,head_dim", [(120, 6), (512, 32), (5120, 128), (8192, 256)]
)
@pytest.mark.parametrize("rank", [2, 3])
def test_full_width_tp_slice_and_strides(dtype, interleave, width, head_dim, rank):
    torch.manual_seed(517)
    storage = torch.randn(2, 11, width * 6, device="cuda", dtype=dtype)
    q = storage[:, :7, : width * 2 : 2]
    k = storage[:, :7, width * 2 : width * 4 : 2]
    if rank == 2:
        q, k = q[0], k[0]
    qw, kw = [torch.randn(width, device=q.device, dtype=dtype) for _ in range(2)]
    phase = torch.randn(7, head_dim * 2, device=q.device)
    cos, sin = phase[:, ::2].cos(), phase[:, ::2].sin()
    start, heads = width // head_dim // 2, width // head_dim // 2
    original = storage.clone()
    actual = qk_rmsnorm_rope(
        q, k, qw, kw, cos, sin, 1e-6, head_dim, interleave, start, heads
    )
    expected = reference(q, k, qw, kw, cos, sin, head_dim, interleave, start, heads)
    tolerance = {torch.float16: 0.005, torch.bfloat16: 0.04, torch.float32: 3e-6}[dtype]
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=tolerance, rtol=tolerance)
        assert result.is_contiguous()
        assert result.shape == (*q.shape[:-1], heads, head_dim)
    torch.testing.assert_close(storage, original, atol=0, rtol=0)


@pytest.mark.parametrize("interleave", [False, True])
def test_broadcast_phases_and_empty_inputs(interleave):
    q = torch.randn(1, 9, 512, device="cuda", dtype=torch.bfloat16).expand(2, -1, -1)
    k = torch.randn_like(q)
    weight = torch.ones(512, device=q.device, dtype=q.dtype)
    phases = torch.randn(2, 1, 64, device=q.device)[..., ::2]
    cos, sin = phases, phases.flip(-1)
    actual = qk_rmsnorm_rope(q, k, weight, weight, cos, sin, 1e-6, 32, interleave)
    expected = reference(q, k, weight, weight, cos, sin, 32, interleave, 0, 16)
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=0.032, rtol=0.032)
    for shape in ((2, 0, 512), (0, 9, 512), (0, 512)):
        empty = torch.empty(shape, device=q.device, dtype=q.dtype)
        out = qk_rmsnorm_rope(
            empty,
            empty,
            weight,
            weight,
            torch.ones(32, device=q.device),
            torch.zeros(32, device=q.device),
            1e-6,
            32,
        )
        assert out[0].shape == (*shape[:-1], 16, 32)


def test_low_precision_rounding_is_not_elided():
    torch.manual_seed(19)
    q = torch.randn(1, 32, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(5120, device=q.device, dtype=q.dtype)
    cos, sin = torch.ones(128, device=q.device), torch.zeros(128, device=q.device)
    actual = qk_rmsnorm_rope(q, q, w, w, cos, sin, 1e-6, 128)[0].flatten(-2)
    rounded = rmsnorm_cast_affine(q, w)
    torch.testing.assert_close(actual, rounded, atol=0, rtol=0)
    unrounded = (
        q.float()
        * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-6)
        * w.float()
    ).to(q.dtype)
    assert not torch.equal(actual, unrounded)


def test_invalid_kernel_contracts():
    q = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(128, device=q.device, dtype=q.dtype)
    c = torch.ones(32, device=q.device)
    with pytest.raises(ValueError, match="matching rank"):
        qk_rmsnorm_rope(q, q[:1], w, w, c, c, 1e-6, 32)
    with pytest.raises(ValueError, match="positive even"):
        qk_rmsnorm_rope(q, q, w, w, c, c, 1e-6, 31)
    with pytest.raises(ValueError, match="head slice"):
        qk_rmsnorm_rope(q, q, w, w, c, c, 1e-6, 32, head_start=4)
    with pytest.raises(ValueError, match="finite and positive"):
        qk_rmsnorm_rope(q, q, w, w, c, c, float("nan"), 32)
    with pytest.raises(ValueError, match="matching projection"):
        qk_rmsnorm_rope(q, q, w.float(), w, c, c, 1e-6, 32)
    with pytest.raises(ValueError, match="num_warps"):
        qk_rmsnorm_rope(q, q, w, w, c, c, 1e-6, 32, num_warps=2)


@pytest.mark.parametrize("num_warps", [4, 8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_configurable_reduction_layout(num_warps, dtype):
    torch.manual_seed(739)
    storage = torch.randn(2, 13, 15360, device="cuda", dtype=dtype)
    q, k = storage[:, :11, :5120], storage[:, :11, 5120:10240]
    qw, kw = [torch.randn(5120, device=q.device, dtype=dtype) for _ in range(2)]
    cos, sin = [torch.randn(11, 128, device=q.device) for _ in range(2)]
    actual = qk_rmsnorm_rope(
        q,
        k,
        qw,
        kw,
        cos,
        sin,
        1e-6,
        128,
        head_start=20,
        local_heads=20,
        num_warps=num_warps,
    )
    expected = reference(q, k, qw, kw, cos, sin, 128, True, 20, 20)
    tolerance = 0.04 if dtype == torch.bfloat16 else 3e-6
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=tolerance, rtol=tolerance)
