"""Hot-path cost of selection, and the memo keys that make caching sound.

Selection sits inside every forward pass, so the cost of *reaching* a cached
answer matters as much as the cache itself. The timing tests are guards, not
benchmarks: they assert an order of magnitude, generously, so they fail on a
regression of the kind that already happened (``Catalog.version`` and
``Policy.version`` recomputed on every read, 77% of a cached selection) rather
than on a slow machine. The memo-key tests are the correctness basis: keying
on too little returns a stale selection, the one silent failure in this system.
"""

from __future__ import annotations

import gc
import time

import pytest

from phyai.kernel import bootstrap
from phyai.kernel.call import (
    CallSite,
    FrozenChoiceError,
    freeze_kernel_choices,
    get_call_sites,
    reset_verify_frozen,
    unfreeze_kernel_choices,
    verify_frozen,
)
from phyai.kernel.facts import dtype
from phyai.kernel.opspec import Impl, OpSpec
from phyai.kernel.policy import Policy, policy_from_mapping
from phyai.kernel.library import library_available
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery, ModelContext

requires_flashinfer = pytest.mark.skipif(
    not library_available("flashinfer"),
    reason="expects a row gated on lib.flashinfer; flashinfer is not importable",
)


def elapsed_us(fn, n: int = 2000) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1e6


@pytest.fixture
def selector():
    return Selector(build_catalog(), device="nvidia:SM90")


@pytest.fixture(autouse=True)
def unpinned():
    """No test leaks a pin into the next one."""
    yield
    unfreeze_kernel_choices()
    reset_verify_frozen()


def rmsnorm_query() -> KernelQuery:
    return KernelQuery.build(
        "rmsnorm",
        role="norm",
        dtype={"input": "bf16", "weight": "bf16", "output": "bf16"},
        shape={"tokens": 8, "hidden": 4096},
        attrs={"variant": "rms"},
    )


def steer_gemm(catalog, match: dict | None = None) -> Policy:
    return policy_from_mapping(
        {
            "rules": [
                {
                    "id": "r",
                    "match": {"op": "gemm", **(match or {})},
                    "prefer": ["torch.gemm.bf16"],
                }
            ]
        },
        catalog,
    )


GEMM_CALL = dict(
    dtype={"input": "bf16", "output": "bf16"},
    quant={"format": "bf16"},
    dims={"M": 512, "N": 4096, "K": 4096},
)
BF16 = {"input": "bf16", "weight": "bf16"}


# --------------------------------------------------------------------------- #
# The fingerprints are computed once
# --------------------------------------------------------------------------- #


def test_fingerprints_are_memoized_but_still_invalidated_by_registration():
    """Reading ``Catalog.version`` used to render 36 predicates and hash them;
    memoization must not cost invalidation, or a stale key would pin a
    selection to a kernel whose contract has since changed."""
    catalog = build_catalog()
    first = catalog.version
    assert elapsed_us(lambda: catalog.version, n=5000) < 1.0
    assert catalog.version == first
    policy = Policy()
    assert policy.version and elapsed_us(lambda: policy.version, n=5000) < 1.0

    toy = Catalog()
    toy.register_op(OpSpec(name="toy", dtypes=("input",)))
    toy.register(
        Impl(
            kernel_id="a.toy",
            op="toy",
            when=dtype.input.is_set(),
            prepare=lambda f, p: None,
        )
    )
    before = toy.version
    toy.register(
        Impl(
            kernel_id="b.toy",
            op="toy",
            when=dtype.input == "bf16",
            prepare=lambda f, p: None,
        )
    )
    assert toy.version != before


def test_a_warm_selection_is_not_dominated_by_key_construction(selector):
    """The raw ``Selector.select`` entry point still builds a query to look up
    its own cache; the bound call site below is two orders cheaper again."""
    query = rmsnorm_query()
    selector.select(query)  # warm
    assert elapsed_us(lambda: selector.select(query)) < 200.0


# --------------------------------------------------------------------------- #
# CallSite, and the completeness of its memo key
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", [spec.name for spec in build_catalog().ops()])
def test_memo_key_covers_every_fact_any_capability_reads(op: str):
    """If an implementation reads a fact the key omits, two calls differing
    only in that fact collide and the second gets the first's answer."""
    catalog = build_catalog()
    covered = set(CallSite(op).key_paths())
    required: set[str] = set()
    for impl in catalog.impls(op):
        required |= impl.when.facts_used()
    # Library availability is a process constant; the device is keyed explicitly.
    required = {path for path in required if not path.startswith("lib.")}
    assert required <= covered, f"{op}: memo key omits {sorted(required - covered)}"


def test_memo_key_covers_the_facts_the_policy_reads(selector):
    bootstrap.set_kernel_selector(
        Selector(
            selector.catalog,
            steer_gemm(selector.catalog, {"model.family": "qwen"}),
            device="nvidia:SM90",
        )
    )
    assert "model.family" in CallSite("gemm").key_paths()


@requires_flashinfer
def test_dtype_device_and_mode_changes_are_not_served_from_the_memo():
    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    bf16 = site.select(device="nvidia:SM90", dtype=BF16)
    assert (
        bf16.kernel_id
        != site.select(
            device="nvidia:SM90", dtype={"input": "fp32", "weight": "fp32"}
        ).kernel_id
    )
    assert bf16.kernel_id != site.select(device="cpu", dtype=BF16).kernel_id

    # ``graph_capture()`` flips the ambient mode; FlashInfer's GDN backend is
    # not capture-safe, so the captured call must land on FLA instead.
    gdn = CallSite("attention_gdn", role="gdn", attrs={"layout": "paged"})
    dtypes = {
        "input": "bf16",
        "key": "bf16",
        "value": "bf16",
        "a": "bf16",
        "b": "bf16",
        "a_log": "fp32",
        "dt_bias": "bf16",
    }
    eager = gdn.select(device="nvidia:SM90", dtype=dtypes, mode="eager")
    captured = gdn.select(device="nvidia:SM90", dtype=dtypes, mode="capture")
    assert eager.kernel_id != captured.kernel_id


@requires_flashinfer
def test_the_memo_is_rebuilt_when_the_policy_or_model_context_changes(selector):
    """Otherwise a policy swap, or a model-scoped rule, would be served stale
    answers forever."""
    bootstrap.set_kernel_selector(
        Selector(selector.catalog, Policy(), device="nvidia:SM100")
    )
    site = CallSite("gemm", role="mlp.down")
    assert (
        site.select(device="nvidia:SM100", **GEMM_CALL).kernel_id
        == "flashinfer.gemm.bf16"
    )
    bootstrap.set_kernel_selector(
        Selector(selector.catalog, steer_gemm(selector.catalog), device="nvidia:SM100")
    )
    assert (
        site.select(device="nvidia:SM100", **GEMM_CALL).kernel_id == "torch.gemm.bf16"
    )

    by_model = steer_gemm(selector.catalog, {"model.family": "qwen"})
    bootstrap.set_kernel_selector(
        Selector(selector.catalog, by_model, device="nvidia:SM90")
    )
    assert (
        site.select(device="nvidia:SM90", **GEMM_CALL).kernel_id
        == "flashinfer.gemm.bf16"
    )
    bootstrap.set_kernel_selector(
        Selector(
            selector.catalog,
            by_model,
            model=ModelContext(family="qwen"),
            device="nvidia:SM90",
        )
    )
    assert site.select(device="nvidia:SM90", **GEMM_CALL).kernel_id == "torch.gemm.bf16"


def test_a_warm_call_site_is_an_order_of_magnitude_cheaper():
    """Measured at ~2 us against ~50 us for the bare entry point and ~340 us
    before the fingerprints were memoized."""
    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    assert elapsed_us(lambda: site.select(device="cuda", dtype=BF16), n=5000) < 20.0


# --------------------------------------------------------------------------- #
# Pinning: taking the selector off the hot path entirely
# --------------------------------------------------------------------------- #


def warm_site(**overrides) -> CallSite:
    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    site.select(device="cuda", dtype=BF16, **overrides)
    return site


def test_freezing_removes_the_key_from_the_hot_path():
    """Measured ~1.0 us against ~2.3 us keyed. The ambient mode still has to
    be read, because a site is frozen *per mode*."""
    site = warm_site()
    site.freeze()
    assert elapsed_us(lambda: site.select(device="cuda", dtype=BF16), n=20000) < 1.8


def test_a_frozen_site_returns_its_pin_and_refuses_polymorphic_or_cold_sites():
    site = warm_site()
    expected = site.freeze()["eager"]
    # Facts that would normally change the answer are ignored once frozen;
    # that is the point, and why freezing has to refuse polymorphic sites.
    assert site.select(device="cpu", dtype={"input": "fp32"}) is expected

    poly = warm_site()
    poly.select(device="cpu", dtype=BF16)
    with pytest.raises(FrozenChoiceError, match="polymorphic"):
        poly.freeze()
    assert poly.frozen_choices is None
    with pytest.raises(FrozenChoiceError, match="has not run yet"):
        CallSite("rmsnorm", role="norm", dims={"hidden": 4096}).freeze()


def test_freeze_kernel_choices_tallies_and_unfreeze_returns_to_keying():
    """One polymorphic site is normal and must not stop the rest going free."""
    mono = warm_site()
    poly = warm_site()
    poly.select(device="cpu", dtype=BF16)
    cold = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    tally = freeze_kernel_choices()
    assert tally["frozen"] >= 1 and tally["polymorphic"] >= 1 and tally["cold"] >= 1
    assert (
        mono.frozen_choices is not None
        and poly.frozen_choices is None
        and cold.frozen_choices is None
    )
    unfreeze_kernel_choices()
    assert mono.frozen_choices is None
    assert mono.select(device="cpu", dtype=BF16).kernel_id == "torch.rmsnorm"


def test_verify_frozen_catches_a_wrongly_frozen_site_and_is_read_once(monkeypatch):
    """The escape hatch: pay the key again, get told when the pin is a lie. It
    sits in the path pinning exists to empty, so it cannot be an env read."""
    site = warm_site()
    site.freeze()
    reset_verify_frozen()
    assert verify_frozen() is False
    monkeypatch.setenv("PHYAI_KERNEL_VERIFY_FROZEN", "1")
    assert verify_frozen() is False  # cached
    reset_verify_frozen()
    assert verify_frozen() is True
    assert site.select(device="cuda", dtype=BF16) is site.frozen_choices["eager"]
    with pytest.raises(FrozenChoiceError, match="not\\s+monomorphic"):
        site.select(device="cpu", dtype=BF16)


def test_sites_are_frozen_per_mode_and_only_for_modes_warmup_saw():
    """The regression a real pi0.5 run found: warmup runs eager and graph
    capture runs captured, so every site has two memo entries. Comparing
    ``Selection`` identity made two entries naming the *same* kernel look
    polymorphic and refused 447 of 486 sites. A mode never seen falls through
    to keying, because guessing from another mode's entry could hand a
    captured region a capture-unsafe kernel."""
    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    eager = site.select(device="nvidia:SM90", dtype=BF16, mode="eager")
    captured = site.select(device="nvidia:SM90", dtype=BF16, mode="capture")
    assert eager.kernel_id == captured.kernel_id and eager is not captured
    frozen = site.freeze()
    assert frozen == {"eager": eager, "capture": captured}

    partial = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    partial.select(device="nvidia:SM90", dtype=BF16, mode="eager")
    partial.freeze()
    assert set(partial.frozen_choices) == {"eager"}
    assert partial.select(device="nvidia:SM90", dtype=BF16, mode="capture").kernel_id


def test_call_sites_are_registered_weakly_and_freezing_is_opt_in():
    """A discarded layer must not keep its bindings; and freezing trusts warmup
    to have exercised every dtype the run will use, so the speed must be asked for."""
    from phyai.engine_config import EngineConfig

    assert EngineConfig().runtime.freeze_kernel_choices is False
    site = CallSite("rmsnorm", role="throwaway")
    assert any(s is site for s in get_call_sites())
    del site
    gc.collect()
    assert not any(s.role == "throwaway" for s in get_call_sites())


# --------------------------------------------------------------------------- #
# Selector lifecycle
# --------------------------------------------------------------------------- #


def test_the_selector_scope_restores_the_previous_state_even_on_error(selector):
    """``get_kernel_selector`` builds a default as a side effect of *looking*,
    so a save/restore written on top of it would restore that fresh default
    rather than "nothing installed"."""
    bootstrap.reset_kernel_selector()
    with bootstrap.kernel_selector_scope():
        assert bootstrap.get_kernel_selector() is not None
    assert bootstrap._selector is None

    bootstrap.set_kernel_selector(selector)
    replacement = Selector(build_catalog(), device="cpu")
    with bootstrap.kernel_selector_scope(replacement) as installed:
        assert (
            installed is replacement and bootstrap.get_kernel_selector() is replacement
        )
    assert bootstrap.get_kernel_selector() is selector
    with pytest.raises(RuntimeError, match="boom"):
        with bootstrap.kernel_selector_scope(None):
            raise RuntimeError("boom")
    assert bootstrap.get_kernel_selector() is selector
    bootstrap.reset_kernel_selector()
