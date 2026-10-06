"""Catalog structure and the invariants that stop declarations from drifting.

Two of these guard against mistakes the previous design made and could not
detect: ``capture_safe`` was hand-copied into a descriptor table and disagreed
with the backend class it described, and the operation modules must not reach
into ``phyai.layers`` at import time now that the selector is constructible
from inside a layer's forward pass.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from phyai.kernel.facts import FactKind, device, dtype
from phyai.kernel.ops import OP_MODULES
from phyai.kernel.ops import attention as attention_ops
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.predicate import all_of
from phyai.kernel.registry import (
    Catalog,
    UnknownKernelError,
    UnknownOperationError,
    build_catalog,
)


#: Operations that genuinely have no CPU implementation. Written down so a
#: fourth cannot join the set unnoticed.
NO_CPU_PATH = frozenset({"attention_paged", "attention_gdn"})


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def _toy_catalog(*rows: Impl, spec: OpSpec | None = None) -> Catalog:
    catalog = Catalog()
    catalog.register_op(spec or OpSpec(name="toy", dtypes=("input",)))
    catalog.register_many(rows)
    return catalog


def _toy_impl(kernel_id: str, when=None, **kwargs) -> Impl:
    return Impl(
        kernel_id=kernel_id,
        op="toy",
        when=dtype.input.is_set() if when is None else when,
        prepare=lambda facts, params: None,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_every_declared_module_registers_something(catalog):
    assert len(OP_MODULES) == 7
    assert len(catalog.ops()) >= len(OP_MODULES)
    for spec in catalog.ops():
        assert catalog.impls(spec.name), f"{spec.name} has no implementations"


def test_implementations_are_ordered_best_first_whatever_the_registration_order(
    catalog,
):
    rows = catalog.impls("rmsnorm")
    assert [item.kernel_id for item in rows] == [
        "flashinfer.rmsnorm",
        "phyai_kernel.rmsnorm",
        "torch.rmsnorm",
    ]
    priorities = [item.priority for item in rows]
    assert priorities == sorted(priorities, reverse=True)

    rows = [
        _toy_impl("a.toy", priority=Priority.GENERAL),
        _toy_impl("b.toy", priority=Priority.OPTIMIZED + 2),
        _toy_impl("c.toy", priority=Priority.OPTIMIZED),
    ]
    forward = [i.kernel_id for i in _toy_catalog(*rows).impls()]
    assert forward == [i.kernel_id for i in _toy_catalog(*reversed(rows)).impls()]
    assert forward == ["b.toy", "c.toy", "a.toy"]


def test_registry_rejects_unknown_ops_duplicates_and_conflicting_schemas(catalog):
    with pytest.raises(UnknownOperationError, match="unregistered operation"):
        Catalog().register(
            Impl(
                kernel_id="x.nope",
                op="nope",
                when=dtype.input.is_set(),
                prepare=lambda f, p: None,
            )
        )
    toy = _toy_catalog(_toy_impl("dup.toy"))
    with pytest.raises(ValueError, match="duplicate kernel id"):
        toy.register(_toy_impl("dup.toy"))
    with pytest.raises(ValueError, match="different schema"):
        toy.register_op(OpSpec(name="toy", dims=("N",)))
    with pytest.raises(UnknownKernelError, match="unknown kernel"):
        catalog.get("nope.gemm")
    with pytest.raises(UnknownOperationError, match="unknown operation"):
        catalog.op("teleport")


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_every_operation_has_a_reference_row_or_declares_that_it_cannot(catalog):
    """A CPU host cannot run the paged attention operations; that is a
    property of FlashInfer, not an oversight, and the exception list is closed."""
    assert catalog.coverage_gaps() == {}
    without_reference = {
        spec.name
        for spec in catalog.ops()
        if not any(item.reference for item in catalog.impls(spec.name))
    }
    assert without_reference == NO_CPU_PATH
    for name in NO_CPU_PATH:
        assert not catalog.op(name).requires_reference

    gpu_only = _toy_catalog(
        _toy_impl(
            "fast.toy", when=all_of(device.vendor == "nvidia", dtype.input.is_set())
        )
    )
    assert "toy" in gpu_only.coverage_gaps()


# --------------------------------------------------------------------------- #
# Identity and derived views
# --------------------------------------------------------------------------- #


def test_version_is_a_stable_hash_that_tracks_capabilities(catalog):
    """The old descriptor hash omitted the capability, so tightening a
    contract could leave a stale cached choice in place."""
    assert len(catalog.version) == 16
    assert catalog.version == build_catalog().version

    def build(sm: str) -> Catalog:
        return _toy_catalog(
            _toy_impl(
                "fast.toy", when=all_of(device.arch.at_least(sm), dtype.input.is_set())
            )
        )

    assert build("sm90").version != build("sm100").version


def test_manifest_and_describe_render_every_row_stably(catalog):
    payload = catalog.manifest()
    text = json.dumps(payload, sort_keys=True)
    assert payload["version"] == catalog.version
    assert len(payload["operations"]) == len(catalog.ops())
    gemm = next(item for item in payload["operations"] if item["name"] == "gemm")
    assert gemm["signature"] == "(layer, x, bias) -> Tensor"
    row = next(
        r for r in gemm["implementations"] if r["id"] == "flashinfer.gemm.nvfp4_128x4"
    )
    assert "quant.layout == 128x4" in row["when"]  # the rendered contract is the point
    assert row["libraries"] == ["flashinfer"]
    assert "nvfp4" in text
    # Byte-stable across builds so CI can diff intended changes only.
    assert text == json.dumps(build_catalog().manifest(), sort_keys=True)

    lines = catalog.describe().splitlines()
    assert len(lines) == len(catalog.impls())
    assert any("quant.format == nvfp4" in line for line in lines)


def test_libraries_and_id_patterns_are_derived_from_the_rows(catalog):
    """The selector must import only what eligibility actually depends on."""
    assert catalog.libraries() == {
        "flashinfer",
        "phyai_kernel",
        "fla",
        "flash_qla",
        "humming",
    }
    assert catalog.libraries("gemm") == {"flashinfer", "humming"}
    assert catalog.libraries("attention_gdn") == {"flashinfer", "fla", "flash_qla"}
    assert set(catalog.match_ids("flashinfer.gemm.*")) == {
        "flashinfer.gemm.bf16",
        "flashinfer.gemm.fp8_block",
        "flashinfer.gemm.nvfp4_128x4",
    }
    assert catalog.match_ids("nothing.*") == ()


# --------------------------------------------------------------------------- #
# Drift guards
# --------------------------------------------------------------------------- #


ATTENTION_BACKENDS = {
    "flashinfer.attention": attention_ops.ATTENTION_FLASHINFER_CLS,
    "sdpa.attention": attention_ops.ATTENTION_SDPA_CLS,
    "eager.attention": attention_ops.ATTENTION_EAGER_CLS,
    "flashinfer.attention_paged": attention_ops.PAGED_FLASHINFER_CLS,
    "flashinfer.attention_gdn": attention_ops.GDN_FLASHINFER_CLS,
    "fla.attention_gdn": attention_ops.GDN_FLA_CLS,
    "flash_qla.attention_gdn": attention_ops.GDN_FLASH_QLA_CLS,
}


@pytest.mark.parametrize(
    ("kernel_id", "module_path", "class_name"),
    [(k, v[0], v[1]) for k, v in ATTENTION_BACKENDS.items()],
)
def test_capture_safe_matches_the_backend_class(
    catalog, kernel_id, module_path, class_name
):
    """The previous table claimed ``capture_safe=True`` for every attention
    backend by copying a template; ``gdn.flashinfer`` never overrides
    ``supports_capture()``. Reading the class attribute from the exact
    (module, class) pair the row constructs makes the two impossible to separate."""
    import importlib

    backend_cls = getattr(importlib.import_module(module_path), class_name)
    declared = catalog.get(kernel_id).capture_safe
    actual = backend_cls.supports_capture(backend_cls)  # unbound, no instance
    assert declared == actual, f"{kernel_id} declares {declared}, class says {actual}"


def test_catalog_wide_invariants(catalog):
    # Stated directly, because it is the value that was wrong before: nothing
    # calls init_cuda_graph_state on a GDN backend, so the captured path was
    # never exercised.
    assert catalog.get("flashinfer.attention_gdn").capture_safe is False
    assert catalog.get("flash_qla.attention_gdn").capture_safe is False
    assert catalog.get("fla.attention_gdn").capture_safe is True
    # A row eligible for everything would be a bug, not a feature.
    for impl in catalog.impls():
        assert impl.when.facts_used(), f"{impl.kernel_id} constrains nothing"
    gemm = catalog.op("gemm")
    assert gemm.kind_of("shape.K") is FactKind.INT
    assert gemm.kind_of("dtype.input") is FactKind.DTYPE
    assert gemm.kind_of("attrs.anything") is FactKind.ANY


def test_building_the_catalog_imports_neither_layers_nor_flashinfer():
    """Capability is pure data: every backend import is deferred to prepare,
    and constructing the catalog stays cheap and side-effect free."""
    code = (
        "import importlib, sys\n"
        f"for name in {list(OP_MODULES)!r}:\n"
        "    importlib.import_module(f'phyai.kernel.ops.{name}')\n"
        "from phyai.kernel.registry import build_catalog\n"
        "build_catalog()\n"
        "print(sorted(k for k in sys.modules if k.startswith('phyai.layers')))\n"
        "print('flashinfer' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[:2] == ["[]", "False"], result.stdout
