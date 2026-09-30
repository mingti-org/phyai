"""FlashInfer prefill-kernel rows: gating, default order, and rule scoping.

Selection is pure fact evaluation, so a synthetic ``device=`` string is enough
to assert what a Hopper or Blackwell box would pick. That is the point of having
these as catalog rows instead of a config field: the choice becomes testable
without the hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phyai.kernel.ops.attention import FA3_HEAD_DIMS
from phyai.kernel.policy import Policy, PolicyError, load_policy, policy_from_mapping
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery, ModelContext


def _query(op: str, *, head_dim: int = 256, role: str = "", **attrs):
    """A realistic paged/ragged attention query, mirroring shared.py."""
    layout = "ragged" if op == "attention" else "paged"
    base_attrs = {"layout": layout, "causal": True}
    base_attrs.update(attrs)
    shape = {"tokens": 512, "heads": 8, "kv_heads": 8, "head_dim": head_dim}
    if op == "attention":
        shape["kv_tokens"] = 512
    return KernelQuery.build(
        op,
        role=role or op.removeprefix("attention_"),
        dtype={"input": "bf16", "key": "bf16", "value": "bf16", "output": "bf16"},
        shape=shape,
        attrs=base_attrs,
        mode="eager",
    )


def _selector(device: str, policy: Policy | None = None, **kw) -> Selector:
    if policy is None:
        return Selector(build_catalog(), device=device, **kw)
    return Selector(build_catalog(), policy, device=device, **kw)


def _eligible(op: str, *, device: str, **kw) -> set[str]:
    trace = _selector(device).explain(_query(op, **kw))
    return {item.kernel_id for item in trace.candidates if item.eligible}


def _rejection(op: str, kernel_id: str, *, device: str, **kw) -> str:
    trace = _selector(device).explain(_query(op, **kw))
    return next(item.reason for item in trace.candidates if item.kernel_id == kernel_id)


def _rule(**fields) -> Policy:
    return policy_from_mapping(
        {"schema": 1, "rules": [{"id": "r", "priority": 20, **fields}]},
        build_catalog(),
        source="test",
    )


# --------------------------------------------------------------------------- #
# registration shape
# --------------------------------------------------------------------------- #


def test_prefill_rows_are_registered_per_wrapper_and_record_their_backend():
    catalog = build_catalog()
    assert set(catalog.match_ids("flashinfer.attention_paged.*")) == {
        "flashinfer.attention_paged.fa2",
        "flashinfer.attention_paged.fa3",
        "flashinfer.attention_paged.cudnn",
        "flashinfer.attention_paged.trtllm-gen",
    }
    # The ragged wrapper accepts two kernels the paged one does not (they were
    # unreachable under the old global field, whose valid-name set was written
    # for the paged wrapper); the paged wrapper raises for cute-dsl.
    ragged = set(catalog.match_ids("flashinfer.attention.*"))
    assert {"flashinfer.attention.cutlass", "flashinfer.attention.cute-dsl"} <= ragged
    assert "flashinfer.attention.trtllm-gen" not in ragged
    assert "flashinfer.attention_paged.cute-dsl" not in catalog.kernel_ids()
    assert (
        catalog.get("flashinfer.attention_paged.fa2").metadata["prefill_backend"]
        == "fa2"
    )


@pytest.mark.parametrize("op", ("attention", "attention_paged"))
def test_auto_outranks_every_pinned_row_which_outranks_sdpa_and_is_the_default(op):
    """No rule, no autotune: the same row as before these variants existed."""
    catalog = build_catalog()
    auto = int(catalog.get(f"flashinfer.{op}").priority)
    pinned = catalog.match_ids(f"flashinfer.{op}.*")
    assert pinned
    sdpa = int(catalog.get("sdpa.attention").priority)
    for kernel_id in pinned:
        assert sdpa < int(catalog.get(kernel_id).priority) < auto
    assert _selector("nvidia:SM90").select(_query(op)).kernel_id == f"flashinfer.{op}"


# --------------------------------------------------------------------------- #
# gating is generation membership, not a floor
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", ("attention", "attention_paged"))
def test_fa3_is_eligible_on_hopper_only(op):
    """SM120 is the case a ``sm >= 90`` floor gets wrong; the GDN row carried
    the same lesson (a floor let sm120 pass selection and raise inside the
    backend, where a raise cannot drive a fallback)."""
    assert f"flashinfer.{op}.fa3" in _eligible(op, device="nvidia:SM90")
    for arch in ("SM89", "SM120"):
        assert f"flashinfer.{op}.fa3" not in _eligible(op, device=f"nvidia:{arch}")
        assert "arch" in _rejection(op, f"flashinfer.{op}.fa3", device=f"nvidia:{arch}")


@pytest.mark.parametrize("op", ("attention", "attention_paged"))
def test_fa3_needs_a_supported_head_dim(op):
    """Mirrors flashinfer's is_fa3_prefill_head_dim_supported: an unsupported
    head dim *deselects* FA3 rather than reaching the backend. A ragged 96-dim
    call falls through to the reference row; a paged one has no eligible row
    (a 96-dim call used to select fine and die inside ``wrapper.plan``)."""
    unsupported = 96
    assert unsupported not in FA3_HEAD_DIMS
    eligible = _eligible(op, device="nvidia:SM90", head_dim=unsupported)
    assert f"flashinfer.{op}.fa3" not in eligible
    if op == "attention":
        assert (
            _selector("nvidia:SM90").select(_query(op, head_dim=unsupported)).kernel_id
            == "eager.attention"
        )
    else:
        assert not eligible


def test_blackwell_rows_need_blackwell_and_fa2_is_the_universal_fallback():
    for arch, expected in (("SM90", False), ("SM100", True)):
        assert (
            "flashinfer.attention_paged.trtllm-gen"
            in _eligible("attention_paged", device=f"nvidia:{arch}")
        ) is expected
        assert (
            "flashinfer.attention.cutlass"
            in _eligible("attention", device=f"nvidia:{arch}")
        ) is expected
    for arch in ("SM80", "SM90", "SM100", "SM120"):
        assert "flashinfer.attention_paged.fa2" in _eligible(
            "attention_paged", device=f"nvidia:{arch}"
        )


# --------------------------------------------------------------------------- #
# rule scoping, the defect the global config field could not express
# --------------------------------------------------------------------------- #


def _pi05_policy() -> Policy:
    return policy_from_mapping(
        {
            "schema": 1,
            "rules": [
                {
                    "id": "pi05-expert-joint-fa2",
                    "priority": 10,
                    "match": {
                        "op": "attention_paged",
                        "role": "expert",
                        "model.family": "pi05",
                    },
                    "restrict_to": "flashinfer.attention_paged.fa2",
                }
            ],
        },
        build_catalog(),
        source="test",
    )


def test_a_role_and_model_scoped_rule_pins_only_what_was_measured():
    """pi0.5 measured FA2 against auto on the action expert's joint attention
    alone. A global field also pinned the LLM prefix and the vision tower,
    neither of which was measured, and leaked to every other model."""
    pi05 = _selector("nvidia:SM90", _pi05_policy(), model=ModelContext(family="pi05"))
    assert (
        pi05.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged.fa2"
    )
    assert (
        pi05.select(_query("attention_paged", role="prefix")).kernel_id
        == "flashinfer.attention_paged"
    )
    assert pi05.select(_query("attention")).kernel_id == "flashinfer.attention"
    cosmos = _selector(
        "nvidia:SM90", _pi05_policy(), model=ModelContext(family="cosmos3")
    )
    assert (
        cosmos.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged"
    )


def test_rule_params_reach_the_pinned_row_and_only_the_pinned_row():
    """A row pins its kernel by existing; ``params:`` still wins over it, a
    mis-spelled parameter raises instead of silently vanishing, and params
    never leak onto the reference fallback the policy appends."""
    override = _selector(
        "nvidia:SM90",
        _rule(
            match={"op": "attention_paged"},
            restrict_to="flashinfer.attention_paged.fa2",
            params={"prefill_backend": "fa3"},
        ),
    )
    decision = override.select(_query("attention_paged"))
    assert (
        decision.kernel_id == "flashinfer.attention_paged.fa2"
        and decision.params["prefill_backend"] == "fa3"
    )

    typo = _selector(
        "nvidia:SM90",
        _rule(
            match={"op": "attention_paged"},
            restrict_to="flashinfer.attention_paged.fa2",
            params={"prefill_backend_": "fa2"},
        ),
    )
    with pytest.raises(PolicyError, match="prefill_backend_"):
        typo.select(_query("attention_paged"))

    on_cpu = _selector(
        "cpu",
        _rule(
            match={"op": "attention"},
            restrict_to="flashinfer.attention.fa2",
            params={"prefill_backend": "fa2"},
        ),
    )
    selection = on_cpu.select(_query("attention"))
    assert selection.kernel_id == "eager.attention" and selection.params == {}

    forced_auto = _selector(
        "nvidia:SM90",
        _rule(
            match={"op": "attention_paged"}, restrict_to="flashinfer.attention_paged"
        ),
    )
    assert (
        forced_auto.select(_query("attention_paged")).kernel_id
        == "flashinfer.attention_paged"
    )


# --------------------------------------------------------------------------- #
# optional pi0.5 policies
# --------------------------------------------------------------------------- #

#: Models do not load this file automatically. A user may select it through
#: PHYAI_KERNEL_CONFIG; otherwise the catalog priority order is the fallback.
PI05_EXAMPLE_POLICY = (
    Path(__file__).parents[3] / "examples" / "pi05" / "kernel_policy.yaml"
)
PI05_RLINF_BF16_POLICY = (
    Path(__file__).parents[3]
    / "examples"
    / "configs"
    / "kernel_policies"
    / "pi05"
    / "rlinf_bf16.yaml"
)


def test_pi05_example_policy_parses_and_is_scoped_to_its_measurement():
    """A typo in the YAML must fail here, not silently lose 2.5x at runtime."""
    assert PI05_EXAMPLE_POLICY.exists()
    policy = load_policy(PI05_EXAMPLE_POLICY, build_catalog())
    assert [rule.rule_id for rule in policy.rules] == ["pi05-expert-joint-fa2"]
    rule = policy.rules[0]
    assert rule.restrict_to == "flashinfer.attention_paged.fa2"
    assert rule.source_match["model.family"] == "pi05"
    assert rule.source_match["role"] == "expert"  # the prefix shares the op
    assert rule.source_match["device.arch"] == "sm90"  # measured on sm90 only


def test_pi05_example_policy_pins_the_expert_on_sm90_and_nothing_else():
    policy = load_policy(PI05_EXAMPLE_POLICY, build_catalog())
    pi05 = _selector("nvidia:SM90", policy, model=ModelContext(family="pi05"))
    assert (
        pi05.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged.fa2"
    )
    assert (
        pi05.select(_query("attention_paged", role="prefix")).kernel_id
        == "flashinfer.attention_paged"
    )
    assert pi05.select(_query("attention")).kernel_id == "flashinfer.attention"
    other_model = _selector("nvidia:SM90", policy, model=ModelContext(family="cosmos3"))
    assert (
        other_model.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged"
    )
    # Thor (sm110) and Blackwell were never measured: the expert keeps the heuristic.
    for arch in ("SM110", "SM100"):
        other_device = _selector(
            f"nvidia:{arch}", policy, model=ModelContext(family="pi05")
        )
        assert (
            other_device.select(_query("attention_paged", role="expert")).kernel_id
            == "flashinfer.attention_paged"
        )


def test_pi05_rlinf_policy_pins_all_paged_attention_to_fa2():
    """The RL rollout policy keeps prefix and expert kernels stable."""
    policy = load_policy(PI05_RLINF_BF16_POLICY, build_catalog())
    pi05 = _selector("nvidia:SM90", policy, model=ModelContext(family="pi05"))

    for role in ("prefix", "expert"):
        assert (
            pi05.select(_query("attention_paged", role=role)).kernel_id
            == "flashinfer.attention_paged.fa2"
        )

    other_model = _selector("nvidia:SM90", policy, model=ModelContext(family="cosmos3"))
    assert (
        other_model.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged"
    )


def test_fa2_row_raises_the_workspace_floor_and_the_old_env_var_names_the_rule():
    """The floor moved from the engine config onto the resolved kernel, so a
    run that never resolves to FA2 no longer pays for FA2's scratch."""
    from phyai.engine_config import EngineConfig
    from phyai.layers.attention.utils import (
        PREFILL_WORKSPACE_FLOORS,
        resolve_workspace_bytes,
    )

    base = resolve_workspace_bytes()
    assert resolve_workspace_bytes(prefill_backend="fa2") == max(
        base, PREFILL_WORKSPACE_FLOORS["fa2"]
    )
    assert resolve_workspace_bytes(prefill_backend="fa3") == base
    assert resolve_workspace_bytes(prefill_backend="auto") == base

    import os

    os.environ["PHYAI_FLASHINFER_PREFILL_BACKEND"] = "fa2"
    try:
        with pytest.raises(ValueError, match="restrict_to"):
            EngineConfig.from_env()
    finally:
        del os.environ["PHYAI_FLASHINFER_PREFILL_BACKEND"]
