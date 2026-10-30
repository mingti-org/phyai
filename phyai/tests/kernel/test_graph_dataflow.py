"""CPU contracts for the opt-in dataflow operations; no CUDA preparation."""

from __future__ import annotations

import pytest
import torch

from phyai.kernel.ops.rope import strided_input_supported
from phyai.kernel.policy import policy_from_mapping
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


def query(op, *, device="cpu", dtype="bf16", **updates):
    kwargs = {
        "gated_residual_adarmsnorm": {
            "dtype": {"input": dtype, "modulation": dtype},
            "shape": {"tokens": 50, "hidden": 1024},
        },
        "kv_scatter": {
            "dtype": {"input": dtype},
            "shape": {"tokens": 50, "heads_k": 1, "head_dim": 256, "slots": 1019},
        },
        "rope_strided": {
            "dtype": {"input": dtype},
            "shape": {
                "tokens": 50,
                "heads_q": 8,
                "heads_k": 1,
                "head_dim": 256,
                "rotary_dim": 256,
            },
            "attrs": {"rope_type": "default", "interleave": False, "rank": 3},
        },
    }[op]
    for key, value in updates.items():
        kwargs[key].update(value)
    return KernelQuery.build(op, device=device, **kwargs)


@pytest.fixture
def selector(monkeypatch):
    # Simulated library facts exercise predicates without importing a GPU library.
    monkeypatch.setattr(
        "phyai.kernel.selector.library_facts",
        lambda names: {f"lib.{name}": True for name in names},
    )
    return Selector(build_catalog(), device="cpu")


@pytest.mark.parametrize(
    "op,prefix",
    [
        ("gated_residual_adarmsnorm", "phyai_kernel"),
        ("kv_scatter", "phyai_kernel"),
        ("rope_strided", "flashinfer"),
    ],
)
def test_architecture_and_precision_are_explicit(selector, op, prefix):
    target = f"{prefix}.{op}.sm110"
    assert selector.explain(query(op, device="nvidia:sm110")).selected == target
    for other in ("cpu", "nvidia:sm100", "nvidia:sm120", "amd:gfx942"):
        assert selector.explain(query(op, device=other)).selected != target
    assert (
        selector.explain(query(op, device="nvidia:sm110", dtype="fp16")).selected
        != target
    )


@pytest.mark.parametrize(
    "op,updates",
    [
        ("gated_residual_adarmsnorm", {"shape": {"hidden": 2048}}),
        ("gated_residual_adarmsnorm", {"shape": {"tokens": 4097}}),
        ("kv_scatter", {"shape": {"slots": 1020}}),
        ("kv_scatter", {"shape": {"heads_k": 2}}),
        ("kv_scatter", {"shape": {"head_dim": 128}}),
        ("rope_strided", {"attrs": {"interleave": True}}),
        ("rope_strided", {"attrs": {"rank": 4}}),
        ("rope_strided", {"shape": {"rotary_dim": 128}}),
        ("rope_strided", {"shape": {"heads_q": 4}}),
    ],
)
def test_unsupported_shapes_decline_before_preparation(selector, op, updates):
    trace = selector.explain(query(op, device="nvidia:sm110", **updates))
    target = next(c for c in trace.candidates if c.kernel_id.endswith(".sm110"))
    assert not target.eligible


def test_existing_rope_selection_does_not_opt_in(selector):
    q = query("rope_strided", device="nvidia:sm110")
    original = KernelQuery.build(
        "rope", device=q.device, dtype=dict(q.dtype), shape=q.shape, attrs=dict(q.attrs)
    )
    assert selector.explain(original).selected == "flashinfer.rope"


def test_explicit_policy_can_decline_specialization(selector):
    selector.policy = policy_from_mapping(
        {
            "rules": [
                {
                    "id": "reference",
                    "match": {"op": "gated_residual_adarmsnorm"},
                    "prefer": ["torch.gated_residual_adarmsnorm"],
                }
            ]
        },
        selector.catalog,
    )
    trace = selector.explain(query("gated_residual_adarmsnorm", device="nvidia:sm110"))
    assert trace.selected == "torch.gated_residual_adarmsnorm"


def test_missing_library_declines_specialization(selector, monkeypatch):
    monkeypatch.setattr(
        "phyai.kernel.selector.library_facts",
        lambda names: {f"lib.{name}": False for name in names},
    )
    trace = selector.explain(query("gated_residual_adarmsnorm", device="nvidia:sm110"))
    assert trace.selected == "torch.gated_residual_adarmsnorm"


@pytest.mark.parametrize("mod_rows", [1, 2, 4])
def test_reference_fusion_preserves_group_broadcast_and_bf16_residual(mod_rows):
    torch.manual_seed(23)
    a, b = (torch.randn(4, 1024, dtype=torch.bfloat16) for _ in range(2))
    gate = torch.randn(mod_rows, 1024, dtype=torch.bfloat16)
    modulation = torch.randn(mod_rows, 3072, dtype=torch.bfloat16)
    h, y = torch.empty_like(a), torch.empty_like(a)
    selection = Selector(build_catalog(), device="cpu").select(
        query("gated_residual_adarmsnorm")
    )
    returned = selection.execute(a, b, gate, modulation, h, y, 1e-6)
    expected_h = torch.addcmul(a, b, gate.repeat_interleave(4 // mod_rows, 0))
    p = expected_h.float()
    scale, shift, _ = (
        modulation.repeat_interleave(4 // mod_rows, 0).float().chunk(3, -1)
    )
    expected_y = (
        p * torch.rsqrt(p.square().mean(-1, keepdim=True) + 1e-6) * (1 + scale) + shift
    ).bfloat16()
    assert returned[0] is h and returned[1] is y
    torch.testing.assert_close(h, expected_h, rtol=0, atol=0)
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)
    assert a.data_ptr() != h.data_ptr() != y.data_ptr()


def test_reference_scatter_preserves_padding_and_every_other_slot():
    k, v = (torch.randn(4, 1, 256, dtype=torch.bfloat16) for _ in range(2))
    kc, vc = (torch.randn(1019, 1, 1, 256, dtype=torch.bfloat16) for _ in range(2))
    before_k, before_v = kc.clone(), vc.clone()
    slots = torch.tensor([1, 1018, 0, 0])
    selection = Selector(build_catalog(), device="cpu").select(query("kv_scatter"))
    selection.execute(k, v, kc, vc, slots)
    before_k[slots[:2], 0], before_v[slots[:2], 0] = k[:2], v[:2]
    torch.testing.assert_close(kc, before_k, rtol=0, atol=0)
    torch.testing.assert_close(vc, before_v, rtol=0, atol=0)


def test_rope_stride_guard_checks_actual_address_and_storage():
    fused = torch.empty((4, 2560), dtype=torch.bfloat16)
    q, k, _ = fused.split((2048, 256, 256), -1)
    assert strided_input_supported(q.view(4, 8, 256), 8)
    assert strided_input_supported(k.view(4, 1, 256), 1)
    # Both views are contiguous by PyTorch's definition; only one is aligned.
    offset = torch.empty(2049, dtype=torch.bfloat16)[1:].view(1, 8, 256)
    assert offset.is_contiguous() and not strided_input_supported(offset, 8)
    assert not strided_input_supported(q.float().view(4, 8, 256), 8)
    assert not strided_input_supported(q.view(4, 8, 256).transpose(1, 2), 8)
