"""MoE registration, capability gating, and the absence of a fallback.

The MoE operation ships exactly one implementation and declares no
reference path, so two properties carry the whole contract: the FlashInfer
row is discovered and chosen when its capability holds, and every
violation of that capability fails at selection time instead of degrading
into something else. A silent Torch stand-in would return plausible
numbers computed by a different kernel, which is worse than an error.
"""

from __future__ import annotations

import pytest
from phyai.kernel.ops import moe as moe_ops
from phyai.kernel.types import Arch, KernelQuery
from phyai.kernel.ops.moe import MOE, CUTLASS_MOE_ARCHES
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import Selector, NoKernelError

KERNEL_ID = "flashinfer.moe.cutlass_fused_moe"

#: Facts that satisfy the declared capability exactly.
SUPPORTED = {
    "dtype": {"input": "bf16", "weight": "bf16"},
    "shape": {"tokens": 8, "hidden": 64},
    "attrs": {"activation": "swiglu"},
}


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def moe_query(**overrides):
    facts = {key: dict(value) for key, value in SUPPORTED.items()}
    facts.update(overrides)
    return KernelQuery.build("moe", **facts)


def thor_selector(catalog):
    """Thor is the deployment target: sm110 under CUDA 13."""
    return Selector(catalog, device="nvidia:sm110")


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_moe_is_registered_and_discoverable(catalog):
    assert catalog.op("moe") is MOE
    assert [impl.kernel_id for impl in catalog.impls("moe")] == [KERNEL_ID]
    assert catalog.backends("moe") == ("flashinfer",)


def test_moe_declares_no_reference_path(catalog):
    assert catalog.op("moe").requires_reference is False


def test_moe_has_no_reference_row(catalog):
    assert all(not impl.reference for impl in catalog.impls("moe"))


def test_moe_registration_stands_alone():
    """The hook must work on a fresh catalog, not rely on import order."""
    fresh = Catalog()
    moe_ops.register(fresh)
    assert fresh.op("moe") is MOE
    assert fresh.get(KERNEL_ID).op == "moe"


def test_moe_is_not_reported_as_a_coverage_gap(catalog):
    """No reference is required, so its absence is not a gap."""
    assert "moe" not in catalog.coverage_gaps()


# --------------------------------------------------------------------------- #
# Capability: selection succeeds where the contract holds
# --------------------------------------------------------------------------- #


def test_selector_chooses_the_flashinfer_impl(catalog):
    assert thor_selector(catalog).select(moe_query()).kernel_id == KERNEL_ID


def test_thor_is_a_declared_architecture():
    assert "sm110" in CUTLASS_MOE_ARCHES


def test_declared_architectures_convert_to_the_kernel_dispatch_keys():
    """FlashInfer keys its module table on ``f"{major * 10 + minor}"``.

    A mistyped name would still parse and would register an entry no
    device can ever match, so the conversion is pinned to the accepted
    key set rather than trusted to the spelling.
    """
    keys = {
        f"{arch.major * 10 + arch.minor}"
        for arch in (Arch.parse(name) for name in CUTLASS_MOE_ARCHES)
    }
    assert keys == {"89", "90", "100", "103", "107", "110", "120", "121"}


@pytest.mark.parametrize("arch", sorted(CUTLASS_MOE_ARCHES))
def test_every_declared_architecture_is_selectable(catalog, arch):
    selector = Selector(catalog, device=f"nvidia:{arch}")
    assert selector.select(moe_query()).kernel_id == KERNEL_ID


# --------------------------------------------------------------------------- #
# Capability: every violation fails at selection, never deeper
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arch", ["sm80", "sm87", "sm101", "sm130"])
def test_unsupported_architecture_fails_at_selection(catalog, arch):
    selector = Selector(catalog, device=f"nvidia:{arch}")
    with pytest.raises(NoKernelError, match="moe"):
        selector.select(moe_query())


@pytest.mark.parametrize("activation_dtype", ["fp16", "fp32"])
def test_non_bf16_activation_fails_at_selection(catalog, activation_dtype):
    selector = thor_selector(catalog)
    with pytest.raises(NoKernelError, match="moe"):
        selector.select(moe_query(dtype={"input": activation_dtype, "weight": "bf16"}))


def test_non_bf16_expert_weights_fail_at_selection(catalog):
    selector = thor_selector(catalog)
    with pytest.raises(NoKernelError, match="moe"):
        selector.select(moe_query(dtype={"input": "bf16", "weight": "fp16"}))


def test_non_nvidia_device_fails_at_selection(catalog):
    selector = Selector(catalog, device="amd:gfx942")
    with pytest.raises(NoKernelError, match="moe"):
        selector.select(moe_query())


def test_unsupported_activation_fails_at_selection(catalog):
    selector = thor_selector(catalog)
    with pytest.raises(NoKernelError, match="moe"):
        selector.select(moe_query(attrs={"activation": "gelu"}))


def test_absent_flashinfer_fails_at_selection(catalog, monkeypatch):
    """There is no fallback, so a missing dependency is an explicit error."""
    import phyai.kernel.selector as selector_module

    monkeypatch.setattr(
        selector_module,
        "library_facts",
        lambda names: {name: False for name in names},
    )
    selector = thor_selector(catalog)
    with pytest.raises(NoKernelError, match="moe"):
        selector.select(moe_query())


def test_failure_reports_the_rejected_candidate(catalog):
    """The trace names the row that was considered and why it was refused."""
    selector = thor_selector(catalog)
    with pytest.raises(NoKernelError) as excinfo:
        selector.select(moe_query(dtype={"input": "fp32", "weight": "fp32"}))
    message = str(excinfo.value)
    assert "no kernel can handle" in message
    assert KERNEL_ID in message
