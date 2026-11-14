"""Tests for the aiter bf16 GEMM row.

The numeric tests need a ROCm GPU with aiter installed and compare
``tgemm.mm`` with ``F.linear`` on shapes taken from π0.5, covering the
tuned CSV rows, the skinny path, bias and an empty batch.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("aiter")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("aiter GEMM needs a ROCm GPU", allow_module_level=True)

from phyai.kernel.ops.gemm import _aiter_bf16  # noqa: E402


def _gemm_eligible(device: str, output: str = "bf16"):
    from phyai.kernel.bootstrap import get_kernel_selector
    from phyai.kernel.types import KernelQuery, PhysicalSignature

    selector = get_kernel_selector()
    spec = selector.catalog.op("gemm")
    query = selector.normalize(
        KernelQuery.build(
            "gemm",
            device=device,
            dtype={"input": "bf16", "output": output},
            quant=PhysicalSignature(format="bf16"),
            shape={"M": 50, "N": 1024, "K": 2048},
        )
    )
    facts = selector.facts_for(query, spec)
    eligible, _ = selector.assess(spec, facts, query.mode)
    return {item.kernel_id for item in eligible}


def test_aiter_gemm_is_amd_bf16_only():
    assert "aiter.gemm.bf16" in _gemm_eligible("amd:gfx942")
    assert "aiter.gemm.bf16" not in _gemm_eligible("amd:gfx942", output="fp32")
    assert "aiter.gemm.bf16" not in _gemm_eligible("nvidia:SM90")


def test_tuned_shapes_apply_after_aiter_config_was_read():
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    if (get_gfx(), get_cu_num()) != ("gfx942", 304):
        pytest.skip("the shipped tuned shapes are for MI300X")
    shape = (50, 1024, 2048, False, str(torch.bfloat16), str(torch.bfloat16))
    get_GEMM_A16W16_config(*shape)
    _aiter_bf16(None, {})
    assert get_GEMM_A16W16_config(*shape)["libtype"] != "torch"


@pytest.mark.parametrize(
    "M,N,K,with_bias",
    [
        (50, 1024, 2048, False),
        (50, 2560, 1024, False),
        (880, 32768, 2048, False),
        (768, 1152, 1152, True),
        (10, 3072, 1024, True),
        (0, 1024, 2048, False),
    ],
)
def test_aiter_gemm_matches_torch(M, N, K, with_bias):
    torch.manual_seed(0)
    layer = torch.nn.Linear(K, N, bias=with_bias, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    gemm = _aiter_bf16(None, {})
    actual = gemm(layer, x, layer.bias)
    expected = F.linear(x, layer.weight, layer.bias)
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
