"""Residual and dtype-boundary coverage for fused LayerNorm modulation."""

import pytest
import torch
import torch.nn.functional as F

from phyai_kernel import layernorm_modulate


def reference(x, shift, scale, eps=1e-6, residual=None):
    updated = x if residual is None else x + residual
    normalized = F.layer_norm(updated.float(), (x.shape[-1],), None, None, eps).to(
        x.dtype
    )
    return normalized * (1 + scale[:, None]) + shift[:, None], updated


def assert_result(actual, expected, *, x, residual):
    out, updated = actual
    assert out.is_contiguous()
    if residual is None:
        assert updated is x
    else:
        assert updated.is_contiguous()
    torch.testing.assert_close(updated, expected[1], atol=0, rtol=0, equal_nan=True)
    tolerance = {
        torch.float32: 2e-5,
        torch.float16: 3e-3,
        torch.bfloat16: 2e-2,
    }[x.dtype]
    torch.testing.assert_close(
        out, expected[0], atol=tolerance, rtol=tolerance, equal_nan=True
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden", [1, 513, 1024, 5120, 8192, 8193])
@pytest.mark.parametrize("add_residual", [False, True])
def test_layernorm_modulate_reference(device, dtype, hidden, add_residual):
    torch.manual_seed(314)
    x = torch.randn(2, 7, hidden, device=device, dtype=dtype)
    residual = torch.randn_like(x) if add_residual else None
    parameters = torch.randn(2, 6, hidden, device=device, dtype=dtype)
    shift, scale = parameters.unbind(1)[:2]
    assert_result(
        layernorm_modulate(x, shift, scale, residual=residual),
        reference(x, shift, scale, residual=residual),
        x=x,
        residual=residual,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["sliced", "permuted", "expanded"])
@pytest.mark.parametrize("add_residual", [False, True])
def test_layernorm_modulate_strided_inputs(dtype, layout, add_residual):
    torch.manual_seed(123)
    batch, sequence, hidden = 3, 5, 129
    if layout == "sliced":
        x = torch.randn(7, 11, 259, device="cuda", dtype=dtype)[1::2, 1::2, 1::2]
    elif layout == "permuted":
        x = torch.randn(hidden, batch, sequence, device="cuda", dtype=dtype).permute(
            1, 2, 0
        )
    else:
        x = torch.randn(1, sequence, hidden, device="cuda", dtype=dtype).expand(
            batch, sequence, hidden
        )
    residual = (
        torch.randn(batch, sequence, hidden * 2, device="cuda", dtype=dtype)[..., ::2]
        if add_residual
        else None
    )
    parameters = torch.randn(batch, 6, hidden, device="cuda", dtype=dtype)
    shift, scale = parameters.unbind(1)[:2]
    inputs = (x, shift, scale) + (() if residual is None else (residual,))
    snapshots = [value.clone() for value in inputs]
    assert_result(
        layernorm_modulate(x, shift, scale, residual=residual),
        reference(x, shift, scale, residual=residual),
        x=x,
        residual=residual,
    )
    for actual, snapshot in zip(inputs, snapshots, strict=True):
        assert torch.equal(actual, snapshot)


@pytest.mark.parametrize(
    "shift_shape,scale_shape",
    [((1, 33), (3, 1)), ((3, 1), (1, 33)), ((1, 1), (1, 1)), ((3, 33), (3, 33))],
)
def test_layernorm_modulate_independent_broadcasts(shift_shape, scale_shape):
    x = torch.randn(3, 5, 33, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(shift_shape, device=x.device, dtype=x.dtype)
    scale = torch.randn(scale_shape, device=x.device, dtype=x.dtype)
    assert_result(
        layernorm_modulate(x, shift, scale),
        reference(x, shift, scale),
        x=x,
        residual=None,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layernorm_modulate_rounds_normalized_values_and_product(dtype):
    x = torch.tensor([[[-1, 1]]], device="cuda", dtype=dtype)
    normalized = F.layer_norm(x.float(), (2,), eps=0.5).to(dtype)
    # The two cancellations separately detect missing normalization and
    # multiplication casts even when ordinary random errors look small.
    for scale_value in (0.0, 0.5):
        scale = torch.full((1, 2), scale_value, device=x.device, dtype=dtype)
        shift = -(normalized * (1 + scale[:, None])).squeeze(1)
        actual, updated = layernorm_modulate(x, shift, scale, eps=0.5)
        assert updated is x
        assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("dtype,bits", [(torch.float16, 10), (torch.bfloat16, 7)])
def test_layernorm_modulate_rounds_residual_and_scale(dtype, bits):
    x = torch.tensor([[[2.0**bits, 0]]], device="cuda", dtype=dtype)
    residual = torch.tensor([[[0.5, 1]]], device=x.device, dtype=dtype)
    scale = torch.full((1, 2), 2.0 ** (-bits - 1), device=x.device, dtype=dtype)
    normalized = F.layer_norm((x + residual).float(), (2,), eps=1e-6).to(dtype)
    shift = -normalized.squeeze(1)
    actual, updated = layernorm_modulate(x, shift, scale, residual=residual)
    assert torch.equal(updated, x + residual)
    assert torch.equal(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(0, 3, 5), (2, 0, 5), (2, 3, 0)])
def test_layernorm_modulate_empty(device, shape):
    x = torch.empty(shape, device=device)
    condition = torch.zeros(1, 1, device=device)
    for residual in (None, torch.empty_like(x)):
        actual, updated = layernorm_modulate(x, condition, condition, residual=residual)
        assert actual.shape == updated.shape == x.shape
        assert actual.numel() == updated.numel() == 0
        if residual is None:
            assert updated is x


def test_layernorm_modulate_nonfinite_rows():
    x = torch.tensor(
        [[[float("inf"), 1, 2], [float("nan"), -1, 0], [0, 0, 0]]], device="cuda"
    )
    condition = torch.zeros(1, 3, device=x.device)
    assert_result(
        layernorm_modulate(x, condition, condition),
        reference(x, condition, condition),
        x=x,
        residual=None,
    )


def test_layernorm_modulate_cuda_graph_reuses_outputs():
    x = torch.randn(2, 7, 513, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    shift = torch.randn(2, 513, device=x.device, dtype=x.dtype)
    scale = torch.randn_like(shift)
    for _ in range(3):
        layernorm_modulate(x, shift, scale, residual=residual)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = layernorm_modulate(x, shift, scale, residual=residual)
    pointers = tuple(value.data_ptr() for value in outputs)
    for _ in range(2):
        x.mul_(0.75)
        residual.add_(0.125)
        shift.add_(0.25)
        scale.mul_(0.5)
        graph.replay()
        assert tuple(value.data_ptr() for value in outputs) == pointers
        assert_result(
            outputs,
            reference(x, shift, scale, residual=residual),
            x=x,
            residual=residual,
        )


def test_layernorm_modulate_validates_contract():
    x = torch.randn(2, 3, 8)
    condition = torch.randn(2, 8)
    for eps in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="eps"):
            layernorm_modulate(x, condition, condition, eps=eps)
    with pytest.raises(ValueError, match="3-D"):
        layernorm_modulate(x[0], condition, condition)
    with pytest.raises(ValueError, match="dtype"):
        layernorm_modulate(x, condition.half(), condition)
    with pytest.raises(ValueError, match="residual"):
        layernorm_modulate(x, condition, condition, residual=x[0])
    with pytest.raises(ValueError, match="broadcast"):
        layernorm_modulate(x, torch.randn(3, 8), condition)
    with pytest.raises(ValueError, match="float16, bfloat16, and float32"):
        layernorm_modulate(x.double(), condition.double(), condition.double())
