"""Selector semantics.

These pin the behaviours the previous resolver established deliberately and a
rewrite could plausibly lose: an omitted device is filled from the engine
profile but an *explicit* one (including CPU) is never replaced; a losing
candidate is never prepared; ``fallback: error`` does not quietly substitute a
reference implementation; the selection cache is keyed on both the catalog and
the policy fingerprint; capture mode excludes rows that are not capture-safe.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import device as device_ns, dtype as dtype_ns, lib
from phyai.kernel.library import library_available
from phyai.kernel.opspec import (
    Impl,
    OpSpec,
    Priority,
    any_float,
    fixed,
    matches_activation,
)
from phyai.kernel.policy import Policy, policy_from_mapping
from phyai.kernel.predicate import all_of
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import NoKernelError, Selector
from phyai.kernel.types import KernelQuery


TOY = OpSpec(
    name="toy", dims=("M",), dtypes=("input",), attributes=(), signature="(x) -> Tensor"
)


def _row(
    kernel_id: str,
    *,
    priority=Priority.OPTIMIZED + 2,
    when=None,
    prepare=None,
    **kwargs,
) -> Impl:
    return Impl(
        kernel_id=kernel_id,
        op="toy",
        priority=priority,
        when=dtype_ns.input.is_set() if when is None else when,
        prepare=prepare or (lambda facts, params: lambda value: value),
        **kwargs,
    )


def _broken(facts, params):
    raise RuntimeError("no kernel image")


def toy_catalog(*rows: Impl, prepared: list[str] | None = None) -> Catalog:
    """fast (bf16 only) + ref (anything) that record which rows were prepared."""
    log = prepared if prepared is not None else []

    def make(name: str):
        def prepare(facts, params):
            log.append(name)
            return lambda value: (name, value)

        return prepare

    catalog = Catalog()
    catalog.register_op(TOY)
    if not rows:
        rows = (
            _row("fast.toy", when=dtype_ns.input == "bf16", prepare=make("fast")),
            _row(
                "ref.toy",
                priority=Priority.REFERENCE,
                reference=True,
                prepare=make("ref"),
            ),
        )
    catalog.register_many(rows)
    return catalog


def toy_query(**kwargs):
    base = dict(dtype={"input": "bf16"}, shape={"M": 8})
    base.update(kwargs)
    return KernelQuery.build("toy", **base)


def rmsnorm_query(**kwargs):
    base = dict(
        dtype={"input": "bf16", "weight": "bf16"},
        shape={"tokens": 8, "hidden": 4096},
        attrs={"variant": "rms"},
    )
    base.update(kwargs)
    return KernelQuery.build("rmsnorm", **base)


def steer(catalog, prefer: str, **defaults) -> Policy:
    return policy_from_mapping(
        {
            **defaults,
            "rules": [{"id": "r", "match": {"op": "toy"}, "prefer": [prefer]}],
        },
        catalog,
    )


# --------------------------------------------------------------------------- #
# Device normalization
# --------------------------------------------------------------------------- #


def test_omitted_device_is_filled_from_the_profile_but_an_explicit_one_survives():
    selector = Selector(build_catalog(), device="nvidia:SM100")
    trace = selector.explain(rmsnorm_query())
    assert (trace.facts["device.vendor"], trace.facts["device.arch"]) == (
        "nvidia",
        "sm100",
    )
    # A caller asking for a CPU reference must get one, on any host.
    cpu = selector.explain(
        rmsnorm_query(device="cpu", dtype={"input": "fp32", "weight": "fp32"})
    )
    assert cpu.facts["device.vendor"] == "cpu" and cpu.selected == "torch.rmsnorm"


def test_the_architecture_is_one_canonical_fact_and_cpu_has_none():
    """There used to be three facts (``device.arch``, ``device.sm``,
    ``device.sm_major``); the numeric one returned 0 for CPU, which made
    "unknown device" read as "too old". Structure is derived by the operators."""
    for spelling, canonical in (
        ("SM90", "sm90"),
        ("sm120", "sm120"),
        ("gfx942", "gfx942"),
    ):
        vendor = "amd" if canonical.startswith("gfx") else "nvidia"
        trace = Selector(build_catalog(), device=f"{vendor}:{spelling}").explain(
            rmsnorm_query()
        )
        assert trace.facts["device.arch"] == canonical
        assert "device.sm" not in trace.facts and "device.sm_major" not in trace.facts
    cpu = Selector(build_catalog(), device="cpu").explain(
        KernelQuery.build(
            "gemm",
            dtype={"input": "bf16", "output": "bf16"},
            quant={"format": "bf16"},
            shape={"M": 8, "N": 4096, "K": 4096},
        )
    )
    assert cpu.facts["device.arch"] is None


# --------------------------------------------------------------------------- #
# Laziness and preparation failures
# --------------------------------------------------------------------------- #


def test_only_the_winning_candidate_is_prepared():
    prepared: list[str] = []
    catalog = toy_catalog(prepared=prepared)
    assert (
        Selector(catalog, device="nvidia:SM90").select(toy_query()).kernel_id
        == "fast.toy"
    )
    assert prepared == ["fast"]
    prepared.clear()
    policy = policy_from_mapping(
        {"overrides": [{"id": "o", "match": {"op": "toy"}, "use": "ref.toy"}]}, catalog
    )
    assert (
        Selector(catalog, policy, device="nvidia:SM90").select(toy_query()).kernel_id
        == "ref.toy"
    )
    assert prepared == ["ref"]


def test_a_failing_candidate_falls_through_unless_a_strict_override_named_it():
    catalog = toy_catalog(
        _row("broken.toy", prepare=_broken),
        _row("ref.toy", priority=Priority.REFERENCE, reference=True),
    )
    selector = Selector(catalog, device="nvidia:SM90")
    assert selector.select(toy_query()).kernel_id == "ref.toy"
    assert (
        selector.explain(toy_query()).candidates[0].kernel_id == "broken.toy"
    )  # recorded

    strict = policy_from_mapping(
        {"overrides": [{"id": "o", "match": {"op": "toy"}, "use": "broken.toy"}]},
        catalog,
    )
    with pytest.raises(NoKernelError, match="failed to prepare"):
        Selector(catalog, strict, device="nvidia:SM90").select(toy_query())
    # A strict override naming an ineligible kernel is an error, not a fallback.
    plain = toy_catalog()
    ineligible = policy_from_mapping(
        {"overrides": [{"id": "o", "match": {"op": "toy"}, "use": "fast.toy"}]}, plain
    )
    with pytest.raises(NoKernelError, match="cannot handle this call"):
        Selector(plain, ineligible, device="nvidia:SM90").select(
            toy_query(dtype={"input": "fp32"})
        )


# --------------------------------------------------------------------------- #
# Ordering and fallback
# --------------------------------------------------------------------------- #


def test_a_rule_is_an_ordered_allow_list_plus_the_reference_fallback():
    catalog = toy_catalog()
    # A rule can prefer a lower-priority kernel ...
    assert (
        Selector(catalog, steer(catalog, "ref.toy"), device="nvidia:SM90")
        .select(toy_query())
        .kernel_id
        == "ref.toy"
    )
    # ... and when its preferred row is ineligible the reference still runs,
    # unless ``fallback: error`` says otherwise.
    fp32 = toy_query(dtype={"input": "fp32"})
    assert (
        Selector(catalog, steer(catalog, "fast.toy"), device="nvidia:SM90")
        .select(fp32)
        .kernel_id
        == "ref.toy"
    )
    strict = steer(catalog, "fast.toy", defaults={"fallback": "error"})
    with pytest.raises(NoKernelError, match="no kernel can handle"):
        Selector(catalog, strict, device="nvidia:SM90").select(fp32)

    # An eligible, higher-priority row the rule did not name is never substituted.
    two = toy_catalog(
        _row(
            "alpha.toy",
            priority=Priority.OPTIMIZED + 2,
            prepare=lambda f, p: lambda v: "alpha",
        ),
        _row(
            "beta.toy",
            priority=Priority.OPTIMIZED + 1,
            prepare=lambda f, p: lambda v: "beta",
        ),
    )
    assert (
        Selector(two, steer(two, "beta.toy"), device="nvidia:SM90")
        .select(toy_query())
        .kernel_id
        == "beta.toy"
    )


def test_capture_mode_excludes_non_capture_safe_rows_and_normalizes_aliases():
    catalog = toy_catalog(
        _row("eageronly.toy", capture_safe=False),
        _row("ref.toy", priority=Priority.REFERENCE, reference=True),
    )
    selector = Selector(catalog, device="nvidia:SM90")
    assert selector.select(toy_query()).kernel_id == "eageronly.toy"
    assert selector.select(toy_query(mode="capture")).kernel_id == "ref.toy"
    trace = selector.explain(toy_query(mode="capture"))
    assert "CUDA graph" in next(
        c.reason for c in trace.candidates if c.kernel_id == "eageronly.toy"
    )
    # ``graph_capturing`` is what phyai.parallel.state.Mode actually produces.
    for alias in ("capture", "graph_capturing", "graph-capturing"):
        assert selector.explain(toy_query(mode=alias)).facts["mode"] == "capture"


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_selection_is_cached_per_catalog_and_policy_fingerprint():
    prepared: list[str] = []
    catalog = toy_catalog(prepared=prepared)
    selector = Selector(catalog, device="nvidia:SM90")
    assert selector.select(toy_query()) is selector.select(toy_query())
    assert prepared == ["fast"]
    selector.clear_cache()
    selector.select(toy_query())
    assert prepared == ["fast", "fast"]
    # The cache key carries the policy fingerprint, so a steered selector
    # cannot be served the plain one's answer.
    steered = Selector(catalog, steer(catalog, "ref.toy"), device="nvidia:SM90")
    assert steered.select(toy_query()).kernel_id == "ref.toy"
    assert selector.policy.version != steered.policy.version


# --------------------------------------------------------------------------- #
# Errors and traces
# --------------------------------------------------------------------------- #


def test_errors_carry_the_full_reasoning():
    catalog = toy_catalog(_row("cuda.toy", when=device_ns.vendor == "nvidia"))
    with pytest.raises(NoKernelError) as excinfo:
        Selector(catalog, device="cpu").select(toy_query(device="cpu"))
    assert "device.vendor == nvidia" in str(excinfo.value) and "got 'cpu'" in str(
        excinfo.value
    )
    with pytest.raises(KeyError, match="unknown operation"):
        Selector(build_catalog(), device="nvidia:SM90").select(
            KernelQuery.build("teleport")
        )


def test_traces_record_the_contract_and_report_vacuous_optional_facts():
    trace = Selector(build_catalog(), device="nvidia:SM90").explain(rmsnorm_query())
    payload = trace.as_dict()
    assert payload["op"] == "rmsnorm" and payload["facts"]["shape.hidden"] == 4096
    entry = next(c for c in payload["candidates"] if c["id"] == "flashinfer.rmsnorm")
    assert "dtype.input == bf16" in entry["when"]
    trace.to_json()  # must not raise on frozensets or torch dtypes

    # "Matched" must never quietly mean "you did not tell us".
    catalog = Catalog()
    catalog.register_op(
        OpSpec(
            name="toy2",
            dtypes=("input",),
            optional_dtypes=("residual",),
            signature="(x) -> Tensor",
        )
    )
    catalog.register(
        Impl(
            kernel_id="fast.toy2",
            op="toy2",
            when=all_of(dtype_ns.input == "bf16", dtype_ns.residual == "bf16"),
            prepare=lambda facts, params: lambda value: value,
        )
    )
    trace = Selector(catalog, device="nvidia:SM90").explain(
        KernelQuery.build("toy2", dtype={"input": "bf16"})
    )
    assert trace.selected == "fast.toy2"
    assert trace.candidates[0].skipped == ("dtype.residual == bf16",)


# --------------------------------------------------------------------------- #
# Autotune
# --------------------------------------------------------------------------- #


def test_autotune_picks_the_fastest_and_persists_it(tmp_path):
    catalog = toy_catalog()
    timings = {"fast.toy": 5.0, "ref.toy": 2.0}
    calls: list[str] = []

    def benchmark(impl, facts, selection):
        calls.append(impl.kernel_id)
        return timings[impl.kernel_id]

    cache = tmp_path / "autotune.json"
    tuned = Selector(
        catalog,
        Policy(profile="autotune"),
        device="nvidia:SM90",
        benchmark=benchmark,
        autotune_cache=cache,
    )
    # ref wins on measurement despite fast having the higher priority.
    assert tuned.select(toy_query()).kernel_id == "ref.toy"
    assert set(calls) == {"fast.toy", "ref.toy"}
    assert cache.exists() and cache.read_text(encoding="utf-8").strip() != "{}"
    # A fresh selector reads the persisted choice and does not re-measure.
    calls.clear()
    reloaded = Selector(
        catalog,
        Policy(profile="autotune"),
        device="nvidia:SM90",
        benchmark=benchmark,
        autotune_cache=cache,
    )
    assert reloaded.select(toy_query()).kernel_id == "ref.toy" and calls == []


def test_autotune_degrades_to_priority_order_when_it_cannot_measure(tmp_path):
    """Under capture (timing graph construction, not the kernel), when the
    benchmark raises, and when the persisted cache is corrupt."""
    calls: list[str] = []
    capture = Selector(
        toy_catalog(),
        Policy(profile="autotune"),
        device="nvidia:SM90",
        benchmark=lambda impl, facts, selection: calls.append(impl.kernel_id) or 1.0,
    )
    assert (
        capture.select(toy_query(mode="capture")).kernel_id == "fast.toy"
        and calls == []
    )
    raising = Selector(
        toy_catalog(),
        Policy(profile="autotune"),
        device="nvidia:SM90",
        benchmark=lambda i, f, s: 1 // 0,
    )
    assert raising.select(toy_query()).kernel_id == "fast.toy"
    cache = tmp_path / "autotune.json"
    cache.write_text("{not json", encoding="utf-8")
    corrupt = Selector(
        toy_catalog(),
        Policy(profile="autotune"),
        device="nvidia:SM90",
        autotune_cache=cache,
    )
    assert corrupt.select(toy_query()).kernel_id == "fast.toy"


# --------------------------------------------------------------------------- #
# Construction-time parameter dtypes
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not library_available("flashinfer"),
    reason="expects a row gated on lib.flashinfer; flashinfer is not importable",
)
def test_param_dtypes_follow_the_real_catalog_contracts():
    """layernorm gamma is fp32 (FlashInfer's contract), rmsnorm gamma follows
    the activation (FlashInfer reads it through the input type), and a CPU
    host uses the reference contract."""
    cuda = Selector(build_catalog(), device="nvidia:SM90")
    assert cuda.param_dtypes(
        "layernorm", activation="bf16", known={"attrs.bias": True}
    ) == {"weight": "fp32", "bias": "fp32"}
    assert cuda.param_dtypes("rmsnorm", activation="bf16") == {"weight": "bf16"}
    cpu = Selector(build_catalog(), device="cpu")
    assert cpu.param_dtypes("layernorm", activation="fp32") == {
        "weight": "fp32",
        "bias": "fp32",
    }


def test_param_dtypes_ignore_rows_the_device_or_libraries_rule_out():
    """An NVIDIA-only or unavailable-library contract must not dictate an
    allocation on a host that can never run it; an unsatisfiable one is an error."""

    def catalog_with(fast_when, fast_params, ref_params) -> Catalog:
        catalog = Catalog()
        catalog.register_op(OpSpec(name="toy4", dtypes=("input",), params=("weight",)))
        catalog.register(
            Impl(
                kernel_id="fast.toy4",
                op="toy4",
                priority=Priority.OPTIMIZED + 2,
                when=all_of(fast_when, dtype_ns.input.is_set()),
                prepare=lambda f, p: None,
                params=fast_params,
            )
        )
        catalog.register(
            Impl(
                kernel_id="ref.toy4",
                op="toy4",
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype_ns.input.is_set(),
                prepare=lambda f, p: None,
                params=ref_params,
            )
        )
        return catalog

    nvidia_only = catalog_with(
        device_ns.vendor == "nvidia",
        {"weight": fixed("fp32")},
        {"weight": matches_activation()},
    )
    assert Selector(nvidia_only, device="nvidia:SM90").param_dtypes(
        "toy4", activation="bf16"
    ) == {"weight": "fp32"}
    assert Selector(nvidia_only, device="cpu").param_dtypes(
        "toy4", activation="bf16"
    ) == {"weight": "bf16"}
    absent_lib = catalog_with(
        lib.has("phyai_no_such_module_xyz"),
        {"weight": fixed("fp32")},
        {"weight": any_float()},
    )
    assert Selector(absent_lib, device="nvidia:SM90").param_dtypes(
        "toy4", activation="bf16"
    ) == {"weight": "bf16"}

    only = Catalog()
    only.register_op(OpSpec(name="toy3", dtypes=("input",), params=("weight",)))
    only.register(
        Impl(
            kernel_id="only.toy3",
            op="toy3",
            when=dtype_ns.input.is_set(),
            prepare=lambda f, p: None,
            params={"weight": fixed("fp8_e4m3")},
        )
    )
    with pytest.raises(ValueError, match="no dtype satisfies"):
        Selector(only, device="nvidia:SM90").param_dtypes("toy3", activation="bf16")
