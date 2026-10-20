"""DenseMLP SiTU gate ordering and optimized activation selection."""

import pytest
import torch
import torch.nn.functional as F

from phyai.kernel.call import CallSite, token_shape
from phyai.layers.mlp import DenseMLP


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_mlp_situ_uses_triton_and_preserves_gate_order(fake_mesh, dtype):
    fake_mesh()
    model = DenseMLP(
        32, 64, activation="situ", gated=True, bias=False, params_dtype=dtype
    )
    torch.manual_seed(42)
    for parameter in model.parameters():
        parameter.data.normal_(std=0.1)
    value = torch.randn(2, 7, 32, dtype=dtype, device="cuda")
    projected = F.linear(value, model.gate_up_proj.weight)
    gate, hidden = projected.chunk(2, dim=-1)
    activated = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )
    expected = F.linear(activated, model.down_proj.weight)
    actual = model(value)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    site = CallSite(
        "activation", role="mlp.activation", attrs={"activation": "situ", "gated": True}
    )
    selection = site.select(
        device=value.device,
        dtype={"input": dtype, "output": dtype},
        dims=token_shape(projected, hidden=projected.shape[-1]),
    )
    assert selection.kernel_id == "phyai_kernel.activation.situ_and_mul"


def test_situ_rejects_ungated_mlp(fake_mesh):
    fake_mesh()
    with pytest.raises(ValueError, match="Unsupported plain activation"):
        DenseMLP(32, 64, activation="situ", gated=False)


def test_situ_cpu_catalog_falls_back_to_torch():
    value = torch.randn(3, 128, dtype=torch.bfloat16)
    gate, hidden = value.chunk(2, dim=-1)
    expected = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )
    site = CallSite(
        "activation", role="mlp.activation", attrs={"activation": "situ", "gated": True}
    )
    selection = site.select(
        device=value.device,
        dtype={"input": value.dtype, "output": value.dtype},
        dims=token_shape(value, hidden=value.shape[-1]),
    )
    assert selection.kernel_id == "torch.activation.situ_and_mul"
    torch.testing.assert_close(selection.execute(value), expected, atol=0, rtol=0)
