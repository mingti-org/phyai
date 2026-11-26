"""Test-suite kernel-resolver defaults.

The CUDA requirement lives in the workspace-root ``conftest.py`` — the
suite aborts collection on a machine without CUDA. Layer construction
here uses the engine default ``device.target = "cuda"``; CPU tensors
appear only in device-less logic tests (index arithmetic, policy
parsing, weight-loading I/O).

This conftest isolates the process-level kernel resolver per test. See
:func:`_kernel_resolver_isolation` for why that matters.
"""

from __future__ import annotations

import pytest

from phyai.engine_config import DenseParallelConfig, ParallelConfig
from phyai.kernel.bootstrap import kernel_selector_scope
from phyai.kernel.policy import policy_from_mapping
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.parallel.layout import build_rank_layout
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


@pytest.fixture(autouse=True)
def _kernel_selector_isolation():
    """Give every test a pristine, uninstalled kernel selector.

    The selector is a process global carrying a policy, a device profile and a
    selection cache. A test that installs one with a custom catalog or a
    forcing policy would otherwise change what every later test selects — and
    pytest collects ``tests/kernel/`` before ``tests/layers/``, so the leak
    direction that matters is exactly the one that happens.

    The scope reads the global directly rather than through
    ``get_kernel_selector``, which builds a default on demand — constructing one
    just to look at it would restore that default instead of "nothing
    installed", and defeat the isolation.
    """

    with kernel_selector_scope():
        yield


@pytest.fixture
def torch_gemm():
    """Pin GEMM to the torch rows, for tests that compare a layer with ``F.linear`` bit for bit."""
    catalog = build_catalog()
    rule = {"id": "torch-gemm", "match": {"op": "gemm"}, "restrict_to": "torch.gemm.*"}
    policy = policy_from_mapping(
        {"schema": "phyai.kernel/v1", "rules": [rule]}, catalog
    )
    with kernel_selector_scope(Selector(catalog, policy, device="cuda")):
        yield


# --------------------------------------------------------------------------- #
# Shared single-process mesh for layer / weight tests                          #
# --------------------------------------------------------------------------- #


def _fake_mesh(*, name: str = "model", tp_size: int = 1, rank: int = 0) -> Mesh:
    """Register a mesh with a dense TP group of ``tp_size`` and no process groups.

    Collectives short-circuit at group size one, so ws=1 layer tests run
    without torch.distributed; larger ``tp_size`` values exercise the shard
    math (weight loaders, partition sizes) as seen by ``rank``. Re-registering
    under the same name replaces the previous mesh.
    """
    layout = build_rank_layout(
        ParallelConfig(dense=DenseParallelConfig(tp_size=tp_size))
    )
    mesh = Mesh(layout, rank=rank, name=name)
    register_mesh(mesh)
    return mesh


@pytest.fixture
def fake_mesh():
    """Factory fixture, ``fake_mesh(tp_size=4, rank=2)``; restores the registry."""
    saved = dict(_meshes)
    try:
        yield _fake_mesh
    finally:
        _meshes.clear()
        _meshes.update(saved)
