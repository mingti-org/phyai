"""Packed checkpoint partitions keep their own TP shard boundaries."""

import pytest
import torch
import torch.nn.functional as F

from phyai.layers.linear import ColumnParallelLinear
from phyai.weights import WeightLoadSession


@pytest.mark.parametrize("tp_size,rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("bias", [False, True])
def test_packed_column_checkpoint_and_cuda_forward(
    fake_mesh, torch_gemm, tp_size, rank, bias
):
    fake_mesh(tp_size=tp_size, rank=rank)
    sizes = [64, 32, 32]
    layer = ColumnParallelLinear(
        32,
        sum(sizes),
        output_sizes=sizes,
        params_dtype=torch.bfloat16,
        device="cuda",
        bias=bias,
        prefix="attn.qkv",
    )
    checkpoint = {
        "attn.qkv.weight": torch.randn(
            sum(sizes), 32, device="cuda", dtype=torch.bfloat16
        )
        * 0.02,
    }
    if bias:
        checkpoint["attn.qkv.bias"] = torch.randn(
            sum(sizes), device="cuda", dtype=torch.bfloat16
        )
    session = WeightLoadSession(layer)
    session.load(checkpoint)
    report = session.finish(strict=True)
    assert not report.missing and not report.unexpected

    expected = {
        key: torch.cat(
            [part.chunk(tp_size, dim=0)[rank] for part in value.split(sizes, dim=0)]
        )
        for key, value in checkpoint.items()
    }
    torch.testing.assert_close(
        layer.weight, expected["attn.qkv.weight"], rtol=0, atol=0
    )
    if bias:
        torch.testing.assert_close(
            layer.bias, expected["attn.qkv.bias"], rtol=0, atol=0
        )
    value = torch.randn(2, 5, 32, device="cuda", dtype=torch.bfloat16)
    output, _ = layer(value)
    reference = F.linear(
        value, expected["attn.qkv.weight"], expected.get("attn.qkv.bias")
    )
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_packed_column_rejects_wrong_checkpoint_shape(fake_mesh):
    fake_mesh(tp_size=2)
    layer = ColumnParallelLinear(32, 96, output_sizes=[32, 32, 32], prefix="qkv")
    with pytest.raises(ValueError, match="output partitions"):
        layer.weight.weight_loader(
            layer.weight,
            torch.zeros(64, 32, device="cuda", dtype=layer.weight.dtype),
            None,
        )
