"""Precision and layout contracts for channel L2 normalization."""

import pytest
import torch
import torch.nn.functional as F

from phyai.layers.channel_l2_norm import ChannelL2Norm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_channel_l2_norm_matches_reference(dtype, spatial_dims):
    channels = 96
    value = torch.randn((2, channels) + (3,) * spatial_dims, dtype=dtype, device="cuda")
    layer = ChannelL2Norm(channels, spatial_dims=spatial_dims, bias=True, dtype=dtype)
    layer.gamma.copy_(torch.randn_like(layer.gamma))
    layer.bias.copy_(torch.randn_like(layer.bias))
    expected = (
        F.normalize(value.float(), dim=1).to(dtype) * channels**0.5 * layer.gamma
        + layer.bias
    )
    torch.testing.assert_close(layer(value), expected, atol=0, rtol=0)


def test_channel_l2_norm_zero_and_tiny_vectors():
    value = torch.tensor(
        [[0.0, 0.0], [3e-14, 4e-14]], dtype=torch.float32, device="cuda"
    )
    layer = ChannelL2Norm(2, channel_first=False, dtype=torch.float32)
    expected = torch.tensor([[0.0, 0.0], [0.03, 0.04]], device="cuda") * 2**0.5
    torch.testing.assert_close(layer(value), expected)
    assert torch.isfinite(layer(value)).all()


def test_channel_l2_norm_channel_last_and_native_reduction():
    value = torch.randn(2, 7, 96, device="cuda", dtype=torch.bfloat16)
    layer = ChannelL2Norm(96, channel_first=False, compute_dtype=None)
    expected = F.normalize(value, dim=-1) * 96**0.5
    torch.testing.assert_close(layer(value), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("compute_dtype", [None, torch.float32])
@pytest.mark.parametrize("channel_first", [False, True])
@pytest.mark.parametrize("channels", [144, 1152])
def test_fused_channel_l2_norm_preserves_rounding(
    dtype, compute_dtype, channel_first, channels
):
    shape = (2, channels, 3, 5) if channel_first else (2, 3, 5, channels)
    value = torch.randn(shape, device="cuda", dtype=dtype)
    layer = ChannelL2Norm(
        channels,
        channel_first=channel_first,
        compute_dtype=compute_dtype,
        dtype=dtype,
        bias=True,
        backend="phyai-kernel",
    )
    layer.gamma.copy_(torch.randn_like(layer.gamma))
    layer.bias.copy_(torch.randn_like(layer.bias))
    promoted = value if compute_dtype is None else value.to(compute_dtype)
    expected = (
        F.normalize(promoted, dim=1 if channel_first else -1).to(dtype)
        * channels**0.5
        * layer.gamma
        + layer.bias
    )
    actual = layer(value)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert {choice.kernel_id for choice in layer._call._memo.values()} == {
        "phyai_kernel.channel_l2_norm"
    }


@pytest.mark.parametrize("case", ["cpu", "noncontiguous", "mixed_affine"])
def test_channel_l2_norm_fallback_retains_reference_behavior(case):
    device = "cpu" if case == "cpu" else "cuda"
    layer = ChannelL2Norm(
        96,
        device=device,
        dtype=torch.float32 if case == "mixed_affine" else torch.bfloat16,
        bias=True,
        prefix="norm",
    )
    value = torch.randn(2, 96, 3, 5, dtype=torch.bfloat16, device=device)
    if case == "noncontiguous":
        value = value.transpose(-1, -2)
    expected = F.normalize(value.float(), dim=1).to(value.dtype) * 96**0.5
    expected = expected * layer.gamma + layer.bias
    torch.testing.assert_close(layer(value), expected, atol=0, rtol=0)
    assert {choice.kernel_id for choice in layer._call._memo.values()} == {
        "torch.channel_l2_norm"
    }
    assert set(layer.state_dict()) == {"gamma", "bias"}
    assert layer.gamma.hf_keys == [("norm.gamma", None)]


@pytest.mark.parametrize("compute_dtype", [None, torch.float32])
def test_fused_channel_l2_norm_cuda_graph_replays_new_inputs(compute_dtype):
    value = torch.randn(2, 96, 3, 5, device="cuda", dtype=torch.bfloat16)
    layer = ChannelL2Norm(96, compute_dtype=compute_dtype)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(value)
    value.normal_()
    graph.replay()
    promoted = value if compute_dtype is None else value.float()
    expected = F.normalize(promoted, dim=1).to(value.dtype) * 96**0.5
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_dtype", [None, torch.float32])
def test_fused_channel_l2_norm_compile_preserves_reduction(dtype, compute_dtype):
    from phyai_kernel import channel_l2_norm

    value = torch.randn(2, 144, 13, 17, device="cuda", dtype=dtype)
    weight = torch.randn(144, 1, 1, device="cuda", dtype=dtype)
    bias = torch.randn_like(weight)
    compiled = torch.compile(channel_l2_norm, fullgraph=True)
    promoted = value if compute_dtype is None else value.float()
    expected = F.normalize(promoted, dim=1).to(dtype) * 144**0.5 * weight + bias
    actual = compiled(value, weight, bias, compute_dtype=compute_dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("compute_dtype", [None, torch.float32])
def test_fused_channel_l2_norm_clamp_and_nonfinite_values(compute_dtype):
    value = torch.tensor(
        [[0.0, 0.0], [3e-14, 4e-14], [float("nan"), 1.0], [float("inf"), 1.0]],
        dtype=torch.bfloat16,
        device="cuda",
    )
    layer = ChannelL2Norm(2, channel_first=False, compute_dtype=compute_dtype)
    promoted = value if compute_dtype is None else value.float()
    expected = F.normalize(promoted, dim=-1).to(value.dtype) * 2**0.5
    torch.testing.assert_close(layer(value), expected, atol=0, rtol=0, equal_nan=True)
