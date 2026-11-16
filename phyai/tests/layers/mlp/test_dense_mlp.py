"""DenseMLP: construction, HF-key contract, forward parity, fused-vs-split load."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.layers.mlp import DenseMLP


def test_activation_aliases_resolve_consistently(fake_mesh):
    fake_mesh()
    for alias in ("gelu_tanh", "gelu_pytorch_tanh", "gelu-tanh", "gelu_new"):
        m = DenseMLP(
            hidden_size=8,
            intermediate_size=16,
            activation=alias,
            gated=True,
            prefix="block.mlp",
        )
        assert m.activation == "gelu_tanh"


def test_gated_and_plain_topologies_allocate_their_projections(fake_mesh):
    fake_mesh()
    gated = DenseMLP(
        hidden_size=8,
        intermediate_size=16,
        activation="silu",
        gated=True,
        prefix="block.mlp",
    )
    assert gated.gated is True
    assert gated.gate_up_proj.weight.shape == (32, 8)  # gate + up = 2 * 16
    assert gated.down_proj.weight.shape == (8, 16)

    plain = DenseMLP(
        hidden_size=8,
        intermediate_size=24,
        activation="gelu_tanh",
        gated=False,
        bias=True,
        prefix="block.mlp",
    )
    assert plain.gated is False
    assert plain.fc1.weight.shape == (24, 8)
    assert plain.fc2.weight.shape == (8, 24)
    assert plain.fc1.bias is not None and plain.fc2.bias is not None


def test_unsupported_activation_combinations_are_rejected(fake_mesh):
    fake_mesh()
    with pytest.raises(ValueError, match="non-gated SiLU"):
        DenseMLP(hidden_size=8, intermediate_size=16, activation="silu", gated=False)
    with pytest.raises(ValueError, match="Unsupported"):
        DenseMLP(hidden_size=8, intermediate_size=16, activation="relu", gated=True)


def test_hf_keys_follow_the_prefix_and_leg_names(fake_mesh):
    fake_mesh()
    gated = DenseMLP(
        hidden_size=8,
        intermediate_size=16,
        activation="silu",
        gated=True,
        prefix="model.layers.3.mlp",
    )
    # gate_up_proj is a fused MergedColumn with two HF source legs.
    assert gated.gate_up_proj.weight.hf_keys == [
        ("model.layers.3.mlp.gate_proj.weight", 0),
        ("model.layers.3.mlp.up_proj.weight", 1),
    ]
    assert gated.down_proj.weight.hf_keys == [
        ("model.layers.3.mlp.down_proj.weight", None)
    ]

    custom = DenseMLP(
        hidden_size=8,
        intermediate_size=16,
        activation="silu",
        gated=True,
        gated_hf_legs=("w_gate", "w_up"),
        prefix="block.mlp",
    )
    assert custom.gate_up_proj.weight.hf_keys == [
        ("block.mlp.w_gate.weight", 0),
        ("block.mlp.w_up.weight", 1),
    ]

    plain = DenseMLP(
        hidden_size=8,
        intermediate_size=24,
        activation="gelu_tanh",
        gated=False,
        bias=True,
        prefix="vision.encoder.layers.0.mlp",
    )
    assert plain.fc1.weight.hf_keys == [
        ("vision.encoder.layers.0.mlp.fc1.weight", None)
    ]
    assert plain.fc1.bias.hf_keys == [("vision.encoder.layers.0.mlp.fc1.bias", None)]
    assert plain.fc2.weight.hf_keys == [
        ("vision.encoder.layers.0.mlp.fc2.weight", None)
    ]
    assert plain.fc2.bias.hf_keys == [("vision.encoder.layers.0.mlp.fc2.bias", None)]


# ---------------------------------------------------------------------------
# Forward parity (CUDA — flashinfer fused kernels)
# ---------------------------------------------------------------------------


def _load_gated(m: DenseMLP, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor):
    """Load three HF-style tensors via the param-attached weight_loader."""
    m.gate_up_proj.weight.weight_loader(m.gate_up_proj.weight, gate, 0)
    m.gate_up_proj.weight.weight_loader(m.gate_up_proj.weight, up, 1)
    m.down_proj.weight.weight_loader(m.down_proj.weight, down, None)


def _gated_weights(H: int, I: int, seed: int):
    torch.manual_seed(seed)
    gate = (torch.randn(I, H) * 0.02).to(torch.bfloat16).cuda()
    up = (torch.randn(I, H) * 0.02).to(torch.bfloat16).cuda()
    down = (torch.randn(H, I) * 0.02).to(torch.bfloat16).cuda()
    x = (torch.randn(8, H) * 0.1).to(torch.bfloat16).cuda()
    return gate, up, down, x


@pytest.mark.parametrize("activation", ("silu", "gelu_tanh"))
def test_forward_gated_matches_torch_reference(fake_mesh, activation):
    fake_mesh()
    H, I = 64, 256
    m = DenseMLP(
        hidden_size=H,
        intermediate_size=I,
        activation=activation,
        gated=True,
        params_dtype=torch.bfloat16,
        prefix="mlp",
    ).cuda()
    gate, up, down, x = _gated_weights(H, I, seed=0)
    _load_gated(m, gate, up, down)

    act = F.silu if activation == "silu" else (lambda t: F.gelu(t, approximate="tanh"))
    ref = F.linear(act(F.linear(x, gate)) * F.linear(x, up), down)
    torch.testing.assert_close(m(x), ref, atol=2e-2, rtol=2e-2)


def test_forward_plain_gelu_tanh_matches_torch_reference(fake_mesh, torch_gemm):
    fake_mesh()
    # Plain path uses F.gelu — the torch reference row, no flashinfer needed.
    H, I = 32, 96
    m = DenseMLP(
        hidden_size=H,
        intermediate_size=I,
        activation="gelu_tanh",
        gated=False,
        bias=True,
        params_dtype=torch.bfloat16,
        prefix="vit.mlp",
    )
    torch.manual_seed(2)
    for param in (m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias):
        nn.init.normal_(param, std=0.02)

    x = (torch.randn(4, H, device="cuda") * 0.1).to(torch.bfloat16)
    h = F.gelu(F.linear(x, m.fc1.weight, m.fc1.bias), approximate="tanh")
    ref = F.linear(h, m.fc2.weight, m.fc2.bias)
    torch.testing.assert_close(m(x), ref, atol=0, rtol=0)


def test_fused_vs_split_load_produces_identical_output(fake_mesh):
    """Loading via [gate, up] split == pre-concatenated fused weight."""
    fake_mesh()
    H, I = 32, 64
    gate, up, down, x = _gated_weights(H, I, seed=3)
    x = x[:4]

    def build() -> DenseMLP:
        return DenseMLP(
            hidden_size=H,
            intermediate_size=I,
            activation="silu",
            gated=True,
            params_dtype=torch.bfloat16,
            prefix="mlp",
        ).cuda()

    split = build()
    _load_gated(split, gate, up, down)

    fused = build()
    fused.gate_up_proj.weight.data.copy_(torch.cat([gate, up], dim=0))
    fused.down_proj.weight.data.copy_(down)

    torch.testing.assert_close(split(x), fused(x), atol=0, rtol=0)
