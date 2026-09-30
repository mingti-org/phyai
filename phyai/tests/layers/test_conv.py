"""Precision contracts for convolution layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phyai.layers.conv import Conv2d


def test_conv2d_compute_dtype_preserves_parameter_storage() -> None:
    layer = Conv2d(
        3,
        5,
        kernel_size=3,
        stride=2,
        padding=1,
        dtype=torch.bfloat16,
        compute_dtype=torch.float32,
        device="cpu",
    )
    layer.weight.copy_(torch.randn_like(layer.weight))
    assert layer.bias is not None
    layer.bias.copy_(torch.randn_like(layer.bias))
    layer.post_load()
    value = torch.randn(2, 3, 9, 11, dtype=torch.float32)

    actual = layer(value)
    expected = F.conv2d(
        value,
        layer.weight.float(),
        layer.bias.float(),
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        groups=layer.groups,
    )

    assert layer.weight.dtype == torch.bfloat16
    assert layer.bias.dtype == torch.bfloat16
    assert layer._compute_weight is not None
    assert layer._compute_bias is not None
    assert layer._compute_weight.dtype == torch.float32
    assert layer._compute_bias.dtype == torch.float32

    compute_weight_ptr = layer._compute_weight.data_ptr()
    compute_bias_ptr = layer._compute_bias.data_ptr()
    layer.weight.copy_(torch.randn_like(layer.weight))
    layer.bias.copy_(torch.randn_like(layer.bias))
    layer.post_load()
    assert layer._compute_weight.data_ptr() == compute_weight_ptr
    assert layer._compute_bias.data_ptr() == compute_bias_ptr
    torch.testing.assert_close(layer._compute_weight, layer.weight.float())
    torch.testing.assert_close(layer._compute_bias, layer.bias.float())

    assert "_compute_weight" not in layer.state_dict()
    assert "_compute_bias" not in layer.state_dict()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
