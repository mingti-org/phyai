"""Regression tests for the shipped example kernel policy.

``examples/configs/kernel_policy.example.yaml`` documents the policy
DSL, and nothing in the codebase loads it. That combination is a liability: a
refactor can break the one documented example while the whole suite stays
green. These tests close that gap.

The important assertion is not "the file parses" — it is "every rule actually
fires". Asserting each rule id against a representative call is what catches a
rule that has been narrowed into uselessness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phyai.kernel.facts import facts_from_query
from phyai.kernel.library import library_available
from phyai.kernel.policy import Policy, load_policy
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery

requires_flashinfer = pytest.mark.skipif(
    not library_available("flashinfer"),
    reason="expects a row gated on lib.flashinfer; flashinfer is not importable",
)


EXAMPLE_POLICY = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "configs"
    / "kernel_policy.example.yaml"
)


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    return build_catalog()


@pytest.fixture(scope="module")
def policy(catalog: Catalog) -> Policy:
    if not EXAMPLE_POLICY.exists():
        pytest.skip(f"example policy not present at {EXAMPLE_POLICY}")
    return load_policy(EXAMPLE_POLICY, catalog)


def facts_for(catalog: Catalog, op: str, **kwargs):
    return facts_from_query(
        KernelQuery.build(op, **kwargs),
        catalog.op(op),
        libraries={"lib.flashinfer": True, "lib.phyai_kernel": True},
    )


def test_example_policy_loads_and_declares_the_expected_shape(policy: Policy) -> None:
    assert policy.profile == "static"
    assert policy.fallback == "reference"
    assert [rule.rule_id for rule in policy.rules] == [
        "sm100-nvfp4-gemm",
        "sm90-fp8-gemm",
        "ragged-prefill",
        "pin-fa2-prefill",
        "fp32-affine-layernorm",
    ]
    assert [rule.rule_id for rule in policy.overrides] == [
        "force-reference-attention-for-debug",
        "force-torch-gemm",
    ]
    # ``load_policy`` raises on an unknown id, so reaching this point proves
    # every ``prefer`` / ``use`` / ``restrict_to`` entry still exists.
    referenced = {kernel_id for rule in policy.rules for kernel_id in rule.prefer}
    referenced |= {rule.use for rule in policy.overrides if rule.use}
    assert referenced and all(isinstance(item, str) for item in referenced)


def gemm_call(catalog: Catalog, **overrides):
    base = dict(
        role="mlp.down",
        device="nvidia:SM100",
        dtype={"input": "bf16", "output": "bf16"},
        quant={"format": "nvfp4", "layout": "128x4", "block_shape": (1, 16)},
        shape={"M": 8, "N": 4096, "K": 4096},
    )
    base.update(overrides)
    return facts_for(catalog, "gemm", **base)


CASES = {
    "sm100-nvfp4-gemm": lambda catalog: gemm_call(catalog),
    "sm90-fp8-gemm": lambda catalog: gemm_call(
        catalog,
        device="nvidia:SM90",
        quant={
            "format": "fp8_e4m3",
            "granularity": "block",
            "block_shape": (128, 128),
        },
    ),
    "ragged-prefill": lambda catalog: facts_for(
        catalog,
        "attention",
        role="attention",
        device="nvidia:SM90",
        dtype={"input": "bf16", "key": "bf16", "value": "bf16"},
        shape={"head_dim": 128, "tokens": 512},
        attrs={"layout": "ragged", "causal": True},
    ),
    # The action expert's paged op -- the site pi0.5 actually measured. The
    # rule used to target the no-cache `attention` op, which is the vision
    # tower, and was pinned only because the knob was global.
    "pin-fa2-prefill": lambda catalog: facts_for(
        catalog,
        "attention_paged",
        role="expert",
        device="nvidia:SM90",
        model={"family": "pi05"},
        dtype={"input": "bf16", "key": "bf16", "value": "bf16"},
        shape={"head_dim": 256, "tokens": 512},
        attrs={"layout": "paged", "causal": True},
    ),
    "fp32-affine-layernorm": lambda catalog: facts_for(
        catalog,
        "layernorm",
        role="layernorm",
        device="nvidia:SM90",
        dtype={"input": "bf16", "weight": "fp32", "bias": "fp32"},
        shape={"tokens": 512, "hidden": 4096},
        attrs={"bias": True},
    ),
}


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_each_example_rule_actually_fires(
    policy: Policy, catalog: Catalog, rule_id: str
) -> None:
    """A rule that matches nothing is indistinguishable from a typo."""

    decision = policy.decide(CASES[rule_id](catalog), catalog)
    assert decision.matched_rules == (
        rule_id,
    ), f"rule {rule_id!r} did not fire; matched {decision.matched_rules} instead"
    assert not decision.strict


def test_the_example_overrides_are_strict_and_narrow_as_documented(
    policy: Policy, catalog: Catalog
) -> None:
    # The ragged-prefill call, plus the debug tag the override selects on. Its
    # priority 1000 must beat the rule's 60.
    call = facts_for(
        catalog,
        "attention",
        role="attention",
        device="nvidia:SM90",
        model={"family": "qwen", "tags": frozenset({"debug"})},
        dtype={"input": "bf16", "key": "bf16", "value": "bf16"},
        shape={"head_dim": 128, "tokens": 512},
        attrs={"layout": "ragged", "causal": True},
    )
    decision = policy.decide(call, catalog)
    assert decision.matched_rules == ("force-reference-attention-for-debug",)
    assert decision.strict and decision.candidates == ("eager.attention",)

    narrowed = policy.decide(
        gemm_call(
            catalog,
            model={"family": "qwen", "tags": frozenset({"no-flashinfer"})},
            quant={"format": "bf16"},
        ),
        catalog,
    )
    assert narrowed.matched_rules == ("force-torch-gemm",)
    assert all(item.startswith("torch.gemm.") for item in narrowed.candidates)


def test_the_example_does_not_capture_unrelated_calls(
    policy: Policy, catalog: Catalog
) -> None:
    """Guard against a matcher loose enough to swallow everything."""

    cpu_bf16 = gemm_call(catalog, device="cpu", quant={"format": "bf16"})
    assert policy.decide(cpu_bf16, catalog).matched_rules == ()

    plain_rmsnorm = facts_for(
        catalog,
        "rmsnorm",
        role="norm",
        device="nvidia:SM90",
        dtype={"input": "bf16", "weight": "bf16"},
        shape={"tokens": 8, "hidden": 4096},
        attrs={"variant": "rms"},
    )
    assert policy.decide(plain_rmsnorm, catalog).matched_rules == ()


@requires_flashinfer
def test_the_example_policy_drives_a_real_selection(
    policy: Policy, catalog: Catalog
) -> None:
    """End to end: the documented file must actually select something."""

    selector = Selector(catalog, policy, device="nvidia:SM100")
    trace = selector.explain(
        KernelQuery.build(
            "gemm",
            role="mlp.down",
            dtype={"input": "bf16", "output": "bf16"},
            quant={"format": "nvfp4", "layout": "128x4", "block_shape": (1, 16)},
            shape={"M": 8, "N": 4096, "K": 4096},
        )
    )
    assert trace.matched_rules == ("sm100-nvfp4-gemm",)
    assert trace.selected == "flashinfer.gemm.nvfp4_128x4"
    # Each FlashInfer prefill kernel is a row, so pinning one is ``restrict_to``;
    # a row is preferred over ``params`` because a row is what autotune can measure.
    decision = policy.decide(CASES["pin-fa2-prefill"](catalog), catalog)
    assert decision.candidates == ("flashinfer.attention_paged.fa2",)
