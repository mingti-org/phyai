"""The policy DSL: grammar, compilation, and load-time validation.

One grammar rule removes the previous ambiguity: a list means membership, a
mapping is nested-path sugar, and ``any_of`` / ``all_of`` / ``none_of`` are the
only reserved words. The load-time errors matter as much, because the failure
mode being replaced is a mistyped key that silently produced a rule which could
never match.
"""

from __future__ import annotations

import pytest
from phyai.kernel.facts import facts_from_query
from phyai.kernel.types import KernelQuery
from phyai.kernel.policy import (
    Policy,
    PolicyError,
    load_policy,
    compile_matcher,
    policy_from_mapping,
)
from phyai.kernel.registry import build_catalog


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def facts_for(catalog, op: str, **kwargs):
    spec = catalog.op(op)
    query = KernelQuery.build(op, **kwargs)
    return facts_from_query(query, spec, libraries={"lib.flashinfer": True})


def gemm_facts(catalog, **overrides):
    base = {
        "device": "nvidia:SM100",
        "dtype": {"input": "bf16", "output": "bf16"},
        "quant": {"format": "nvfp4", "layout": "128x4", "block_shape": (1, 16)},
        "shape": {"M": 8, "N": 4096, "K": 4096},
    }
    base.update(overrides)
    return facts_for(catalog, "gemm", **base)


def rules(*items: dict) -> dict:
    return {"rules": list(items)}


PREFER_TORCH = {"prefer": ["torch.gemm.bf16"]}


# --------------------------------------------------------------------------- #
# The one grammar rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mapping, rendered",
    [
        ({"device.vendor": "nvidia"}, "device.vendor == nvidia"),  # scalar: equality
        (
            {"device.vendor": ["nvidia", "amd"]},
            "device.vendor in {amd, nvidia}",
        ),  # list: membership
        (
            {"model.tags": ["debug", "reference"]},
            "model.tags intersects {'debug', 'reference'}",
        ),
        (
            {"device.arch": ">=sm100"},
            "device.arch >= sm100",
        ),  # comparator on an ordered fact
        ({"quant.format": None}, "quant.format is none"),  # null: known absent
        (
            {"any_of": [{"device.vendor": "nvidia"}, {"device.vendor": "amd"}]},
            "device.vendor == nvidia | device.vendor == amd",
        ),
        ({"none_of": [{"device.vendor": "cpu"}]}, "!(device.vendor == cpu)"),
    ],
)
def test_matchers_compile_by_the_one_grammar_rule(mapping, rendered):
    assert compile_matcher(mapping).render() == rendered


def test_op_scoped_forms_and_nested_mappings(catalog):
    """``device: {vendor: x}`` and ``device.vendor: x`` compile alike; op-scoped
    dimensions and the ``when`` implication need the op's schema."""
    nested = compile_matcher({"device": {"vendor": "nvidia", "arch": ">=sm90"}})
    flat = compile_matcher({"device.vendor": "nvidia", "device.arch": ">=sm90"})
    assert nested.render() == flat.render()

    gemm_paths = {"shape.K": catalog.op("gemm").kind_of("shape.K")}
    assert (
        compile_matcher({"shape.K": "%16"}, spec_paths=gemm_paths).render()
        == "shape.K % 16 == 0"
    )
    layernorm = catalog.op("layernorm")
    layernorm_paths = {
        path: layernorm.kind_of(path) for path in layernorm.known_paths()
    }
    conditional = compile_matcher(
        {"when": {"if": {"attrs.bias": True}, "then": {"dtype.bias": "fp32"}}},
        spec_paths=layernorm_paths,
    )
    assert conditional.render() == "attrs.bias == true -> dtype.bias == fp32"

    # The ambiguity the old grammar needed heuristics to resolve.
    with pytest.raises(PolicyError, match="cannot appear inside"):
        compile_matcher({"device": {"any_of": [{"vendor": "nvidia"}]}})
    with pytest.raises(PolicyError, match="not a namespace"):
        compile_matcher({"op": {"min": 3}})
    with pytest.raises(PolicyError, match="unknown fact 'device.vendr'"):
        compile_matcher({"device.vendr": "nvidia"})
    with pytest.raises(PolicyError, match="did you mean 'device.vendor'"):
        compile_matcher({"device.vendor_": "nvidia"})


# --------------------------------------------------------------------------- #
# Load-time validation: the silent-False replacement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "document, complaint",
    [
        (
            rules(
                {"id": "x", "match": {"op": "gemm", "shape.KK": "%16"}, **PREFER_TORCH}
            ),
            r"did you mean 'shape\.K'",
        ),
        (
            rules(
                {"id": "x", "match": {"op": "gemm", "shape.k": "%16"}, **PREFER_TORCH}
            ),
            "case-sensitive",
        ),
        (
            rules({"id": "x", "match": {"op": "gemmm"}, **PREFER_TORCH}),
            "unknown op 'gemmm'",
        ),
        (
            rules({"id": "x", "match": {"op": "gemm"}, "prefer": ["torch.gemm.bf17"]}),
            "did you mean 'torch.gemm.bf16'",
        ),
        ({"rulez": []}, "unknown top-level field"),
        (
            rules({"id": "x", "match": {"op": "gemm"}, **PREFER_TORCH, "prefr": []}),
            "unknown field",
        ),
        # A rule that says nothing silently never does anything; saying both
        # would be two answers to one question (prefer orders, restrict_to shortens).
        (rules({"id": "x", "match": {"op": "gemm"}}), "exactly one of"),
        (
            rules(
                {
                    "id": "x",
                    "match": {"op": "gemm"},
                    **PREFER_TORCH,
                    "restrict_to": "flashinfer.gemm.*",
                }
            ),
            "exactly one of",
        ),
        ({"overrides": [{"id": "x", "match": {"op": "gemm"}}]}, "exactly one of"),
        (
            {
                "overrides": [
                    {
                        "id": "x",
                        "match": {"op": "gemm"},
                        "use": "torch.gemm.bf16",
                        "restrict_to": "torch.gemm.*",
                    }
                ]
            },
            "exactly one of",
        ),
        (
            {
                "overrides": [
                    {"id": "x", "match": {"op": "gemm"}, "restrict_to": "cutlass.*"}
                ]
            },
            "matches no kernel",
        ),
        ({"schema": "phyai.kernel/v9"}, "unsupported schema"),
        ({"profile": "turbo"}, "profile"),
        ({"defaults": {"fallback": "explode"}}, "fallback"),
    ],
)
def test_documents_that_cannot_mean_anything_fail_to_load(catalog, document, complaint):
    with pytest.raises(PolicyError, match=complaint):
        policy_from_mapping(document, catalog)


def test_v1_schema_spellings_are_all_accepted(catalog):
    for schema in ("phyai.kernel/v1", "v1", 1, "1"):
        assert policy_from_mapping({"schema": schema}, catalog).profile == "static"


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #


def test_the_highest_priority_rule_wins_and_a_tie_is_an_error(catalog):
    policy = policy_from_mapping(
        rules(
            {"id": "low", "priority": 1, "match": {"op": "gemm"}, **PREFER_TORCH},
            {
                "id": "high",
                "priority": 2,
                "match": {"op": "gemm"},
                "prefer": ["flashinfer.gemm.nvfp4_128x4"],
            },
        ),
        catalog,
    )
    decision = policy.decide(gemm_facts(catalog), catalog)
    assert decision.matched_rules == ("high",)
    assert decision.candidates == ("flashinfer.gemm.nvfp4_128x4",)

    tied = policy_from_mapping(
        rules(
            {"id": "a", "match": {"op": "gemm"}, **PREFER_TORCH},
            {
                "id": "b",
                "match": {"op": "gemm"},
                "prefer": ["flashinfer.gemm.nvfp4_128x4"],
            },
        ),
        catalog,
    )
    with pytest.raises(PolicyError, match="conflicting rules"):
        tied.decide(gemm_facts(catalog), catalog)


def test_overrides_are_strict_and_restrict_to_expands_a_family(catalog):
    policy = policy_from_mapping(
        {
            "rules": [
                {
                    "id": "r",
                    "priority": 999,
                    "match": {"op": "gemm"},
                    "prefer": ["flashinfer.gemm.nvfp4_128x4"],
                }
            ],
            "overrides": [
                {
                    "id": "o",
                    "priority": 1,
                    "match": {"op": "gemm"},
                    "use": "torch.gemm.bf16",
                }
            ],
        },
        catalog,
    )
    decision = policy.decide(gemm_facts(catalog), catalog)
    assert decision.matched_rules == ("o",) and decision.strict
    assert decision.candidates == ("torch.gemm.bf16",)

    # What "force this backend" has always meant: narrow to a family, then order.
    family = policy_from_mapping(
        {
            "overrides": [
                {"id": "o", "match": {"op": "gemm"}, "restrict_to": "torch.gemm.*"}
            ]
        },
        catalog,
    )
    assert set(family.decide(gemm_facts(catalog), catalog).candidates) == {
        "torch.gemm.bf16",
        "torch.gemm.fp8_block",
        "torch.gemm.fp8_per_channel",
        "torch.gemm.fp8_per_tensor",
        "torch.gemm.nvfp4_linear",
    }


def test_rules_that_do_not_match_express_no_preference(catalog):
    policy = policy_from_mapping(
        rules(
            {
                "id": "amd-only",
                "match": {"op": "gemm", "device.vendor": "amd"},
                **PREFER_TORCH,
            },
            {
                "id": "nvfp4-only",
                "match": {
                    "op": "gemm",
                    "quant": {"format": "nvfp4", "layout": "128x4"},
                },
                "prefer": ["flashinfer.gemm.nvfp4_128x4"],
            },
        ),
        catalog,
    )
    assert policy.decide(gemm_facts(catalog), catalog).matched_rules == ("nvfp4-only",)
    unquantized = policy.decide(gemm_facts(catalog, quant=None), catalog)
    assert unquantized.matched_rules == () and unquantized.candidates == ()


def test_a_rule_can_narrow_to_a_backend_family_or_to_one_role(catalog):
    """What the deleted global ``force this backend`` setting used to do, in
    YAML, plus the thing it could not do: move one role and leave the rest."""
    family = policy_from_mapping(
        rules(
            {
                "id": "flashinfer-gemm",
                "priority": 900,
                "match": {"op": "gemm"},
                "restrict_to": "flashinfer.gemm.*",
            }
        ),
        catalog,
    )
    decision = family.decide(gemm_facts(catalog), catalog)
    assert decision.matched_rules == ("flashinfer-gemm",)
    assert not decision.strict  # soft: a CPU box falls back, it does not fail
    assert set(decision.candidates) == {
        "flashinfer.gemm.bf16",
        "flashinfer.gemm.fp8_block",
        "flashinfer.gemm.nvfp4_128x4",
    }
    norm = facts_for(
        catalog,
        "rmsnorm",
        device="nvidia:SM90",
        dtype={"input": "bf16", "weight": "bf16"},
        shape={"tokens": 8, "hidden": 4096},
        attrs={"variant": "rms"},
    )
    assert (
        family.decide(norm, catalog).matched_rules == ()
    )  # a gemm rule stays a gemm rule

    role = policy_from_mapping(
        rules(
            {
                "id": "ab-mlp-down",
                "priority": 100,
                "match": {"op": "gemm", "role": "mlp.down"},
                **PREFER_TORCH,
            }
        ),
        catalog,
    )
    bf16 = {
        "device": "nvidia:SM100",
        "dtype": {"input": "bf16", "output": "bf16"},
        "quant": {"format": "bf16"},
        "shape": {"M": 512, "N": 4096, "K": 4096},
    }
    targeted = facts_for(catalog, "gemm", role="mlp.down", **bf16)
    assert role.decide(targeted, catalog).candidates[0] == "torch.gemm.bf16"
    other = facts_for(catalog, "gemm", role="qkv_proj", **bf16)
    assert role.decide(other, catalog).matched_rules == ()


# --------------------------------------------------------------------------- #
# Versioning and loading
# --------------------------------------------------------------------------- #


def test_version_tracks_matchers_and_params_but_not_spelling(catalog):
    def build(match: dict, **extra) -> Policy:
        return policy_from_mapping(
            rules({"id": "r", "match": match, **PREFER_TORCH, **extra}), catalog
        )

    assert (
        build({"op": "gemm", "device.arch": ">=sm90"}).version
        != build({"op": "gemm", "device.arch": ">=sm100"}).version
    )
    assert (
        build({"op": "gemm"}, params={"tile": 64}).version
        != build({"op": "gemm"}, params={"tile": 128}).version
    )
    # Nested and flat forms compile to the same predicate, so same version.
    assert (
        build({"op": "gemm", "device": {"vendor": "nvidia"}}).version
        == build({"op": "gemm", "device.vendor": "nvidia"}).version
    )


def test_loading_defaults_deterministically_and_names_the_bad_file(catalog, tmp_path):
    policy = load_policy(None, catalog)
    assert (policy.profile, policy.fallback, policy.rules) == (
        "static",
        "reference",
        (),
    )

    path = tmp_path / "bad.yaml"
    path.write_text("rules:\n  - id: x\n    match: {op: nope}\n", encoding="utf-8")
    with pytest.raises(PolicyError, match=str(path.name)):
        load_policy(path, catalog)
