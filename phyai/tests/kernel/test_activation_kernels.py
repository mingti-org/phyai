"""Direct selection and numerical contracts for activation kernel rows."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


@pytest.mark.parametrize(
    ("activation", "gated", "kernel_id"),
    (
        ("silu", False, "torch.activation.silu"),
        ("gelu", False, "torch.activation.gelu"),
        ("gelu_tanh", False, "torch.activation.gelu_tanh"),
        ("silu", True, "torch.activation.silu_and_mul"),
        ("gelu", True, "torch.activation.gelu_and_mul"),
        ("gelu_tanh", True, "torch.activation.gelu_tanh_and_mul"),
    ),
)
def test_torch_activation_rows_are_directly_selectable(
    activation: str, gated: bool, kernel_id: str
) -> None:
    selector = Selector(build_catalog(), device="cpu")
    selection = selector.select(
        KernelQuery.build(
            "activation",
            dtype={"input": "fp32"},
            shape={"tokens": 2, "hidden": 8},
            attrs={"activation": activation, "gated": gated},
        )
    )
    assert selection.kernel_id == kernel_id


@pytest.mark.parametrize(
    ("activation", "approximate"),
    (("gelu", "none"), ("gelu_tanh", "tanh")),
)
def test_gelu_rows_preserve_exact_vs_tanh_semantics(
    activation: str, approximate: str
) -> None:
    selector = Selector(build_catalog(), device="cpu")
    selection = selector.select(
        KernelQuery.build(
            "activation",
            dtype={"input": "fp32"},
            shape={"tokens": 2, "hidden": 8},
            attrs={"activation": activation, "gated": True},
        )
    )
    value = torch.linspace(-4.0, 4.0, 16).view(2, 8)
    gate, up = value.chunk(2, dim=-1)
    expected = F.gelu(gate, approximate=approximate) * up
    torch.testing.assert_close(selection.execute(value), expected)


@pytest.mark.parametrize(
    ("dtype", "kernel_id"),
    (
        ("bf16", "phyai_kernel.activation.t5_gelu"),
        ("fp16", "phyai_kernel.activation.t5_gelu"),
        ("fp32", "torch.activation.t5_gelu"),
    ),
)
def test_t5_gelu_selection(dtype, kernel_id):
    selector = Selector(build_catalog(), device="cuda")
    selection = selector.select(
        KernelQuery.build(
            "activation",
            dtype={"input": dtype},
            shape={"tokens": 512, "hidden": 10240},
            attrs={"activation": "t5_gelu", "gated": False},
        )
    )
    assert selection.kernel_id == kernel_id
