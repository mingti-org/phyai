import pytest
import torch
from torch.nn import functional as F

from phyai.layers.mlp import DenseMLP


@pytest.mark.parametrize("activation", ["silu", "gelu", "gelu_tanh"])
def test_gated_mlp_matches_separate_bfloat16_operators(fake_mesh, activation):
    fake_mesh()
    torch.manual_seed(91)
    layer = DenseMLP(
        128,
        128,
        activation=activation,
        params_dtype=torch.bfloat16,
        cast_before_multiply=True,
        device="cuda",
    )
    layer.gate_up_proj.weight.copy_(torch.randn_like(layer.gate_up_proj.weight) / 8)
    layer.down_proj.weight.copy_(torch.eye(128, device="cuda", dtype=torch.bfloat16))
    value = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16)
    gate_weight, up_weight = layer.gate_up_proj.weight.chunk(2)
    gate = F.linear(value, gate_weight)
    up = F.linear(value, up_weight)
    if activation == "silu":
        activated = F.silu(gate)
    else:
        activated = F.gelu(
            gate, approximate="tanh" if activation == "gelu_tanh" else "none"
        )
    expected = F.linear(activated * up, layer.down_proj.weight)
    torch.testing.assert_close(layer(value), expected, atol=0, rtol=0)
