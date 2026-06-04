"""LayerNorm — backend parity, F.layer_norm reference, weight-load attach."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

import phyai.layers.linear as L
from phyai.layers.layer_norm import AdaRMSNorm, LayerNorm
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="phyai LayerNorm backends are CUDA-only",
)


# --------------------------------------------------------------------------- #
# Construction-time validation                                                #
# --------------------------------------------------------------------------- #


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown norm backend"):
        LayerNorm(64, backend="banana")


def test_zero_hidden_size_raises():
    with pytest.raises(ValueError, match="hidden_size"):
        LayerNorm(0, backend="phyai-kernel")


def test_extra_repr_contains_key_fields():
    m = LayerNorm(128, eps=1e-6, backend="phyai-kernel", bias=False)
    s = repr(m)
    assert "128" in s
    assert "eps=1e-06" in s
    assert "bias=False" in s
    assert "backend='phyai-kernel'" in s


def test_no_bias_does_not_register_parameter():
    m = LayerNorm(64, backend="phyai-kernel", bias=False)
    assert m.bias is None
    assert not m.has_bias
    assert "bias" not in dict(m.named_parameters())


# --------------------------------------------------------------------------- #
# Weight-load attach                                                          #
# --------------------------------------------------------------------------- #


def test_attach_with_bias():
    m = LayerNorm(64, backend="phyai-kernel", prefix="vision.encoder.layer_norm1")
    assert m.weight.hf_keys == [("vision.encoder.layer_norm1.weight", None)]
    assert m.bias.hf_keys == [("vision.encoder.layer_norm1.bias", None)]
    assert callable(m.weight.weight_loader)
    assert callable(m.bias.weight_loader)


def test_attach_without_bias():
    m = LayerNorm(64, backend="phyai-kernel", bias=False, prefix="text.norm")
    assert m.weight.hf_keys == [("text.norm.weight", None)]
    assert m.bias is None


def test_attach_load_weight_and_bias():
    """End-to-end: invoke the param-attached weight_loader."""
    D = 32
    m = LayerNorm(D, backend="phyai-kernel", prefix="ln")
    src_w = torch.randn(D)
    src_b = torch.randn(D)
    m.weight.weight_loader(m.weight, src_w, None)
    m.bias.weight_loader(m.bias, src_b, None)
    torch.testing.assert_close(m.weight.data, src_w)
    torch.testing.assert_close(m.bias.data, src_b)


# --------------------------------------------------------------------------- #
# Forward correctness                                                         #
# --------------------------------------------------------------------------- #


def _ref_layer_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float
) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


@cuda_only
@pytest.mark.parametrize("backend", ["flashinfer", "phyai-kernel"])
@pytest.mark.parametrize("with_bias", [True, False])
def test_forward_matches_torch_reference_bf16(backend, with_bias):
    torch.manual_seed(0)
    D = 1152  # PaliGemma SigLIP hidden
    m = LayerNorm(
        D, eps=1e-6, backend=backend, bias=with_bias, dtype=torch.bfloat16
    ).cuda()

    src_w = (torch.randn(D) * 0.05 + 1.0).to(torch.bfloat16).cuda()
    m.weight.data.copy_(src_w)
    if with_bias:
        src_b = (torch.randn(D) * 0.02).to(torch.bfloat16).cuda()
        m.bias.data.copy_(src_b)
    else:
        src_b = None

    x = (torch.randn(8, 16, D) * 0.5).to(torch.bfloat16).cuda()
    y = m(x)
    ref = _ref_layer_norm(x, src_w, src_b, 1e-6)
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


@cuda_only
def test_flashinfer_matches_phyai_kernel():
    """Both backends must produce numerically equivalent output."""
    torch.manual_seed(1)
    D = 768
    src_w = (torch.randn(D) * 0.05 + 1.0).to(torch.bfloat16).cuda()
    src_b = (torch.randn(D) * 0.02).to(torch.bfloat16).cuda()

    m_fi = LayerNorm(D, eps=1e-5, backend="flashinfer", dtype=torch.bfloat16).cuda()
    m_pk = LayerNorm(D, eps=1e-5, backend="phyai-kernel", dtype=torch.bfloat16).cuda()
    m_fi.weight.data.copy_(src_w)
    m_pk.weight.data.copy_(src_w)
    m_fi.bias.data.copy_(src_b)
    m_pk.bias.data.copy_(src_b)

    x = (torch.randn(4, 32, D) * 0.5).to(torch.bfloat16).cuda()
    y_fi = m_fi(x)
    y_pk = m_pk(x)
    torch.testing.assert_close(y_fi, y_pk, atol=2e-2, rtol=2e-2)


@cuda_only
def test_phyai_kernel_higher_rank_input():
    """4-D input flattens to (N, D) and reshapes back."""
    torch.manual_seed(2)
    B, S, H, D = 2, 4, 3, 256
    m = LayerNorm(D, backend="phyai-kernel", dtype=torch.bfloat16).cuda()
    x = (torch.randn(B, S, H, D) * 0.3).to(torch.bfloat16).cuda()
    y = m(x)
    assert y.shape == (B, S, H, D)
    ref = _ref_layer_norm(x, m.weight.data, m.bias.data, m.variance_epsilon)
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


@cuda_only
@pytest.mark.parametrize("backend", ["flashinfer", "phyai-kernel"])
def test_no_bias_path_matches_reference(backend):
    """``bias=False`` must give identical output to F.layer_norm(..., bias=None)."""
    torch.manual_seed(3)
    D = 384
    m = LayerNorm(D, backend=backend, bias=False, dtype=torch.bfloat16).cuda()
    src_w = (torch.randn(D) * 0.05 + 1.0).to(torch.bfloat16).cuda()
    m.weight.data.copy_(src_w)

    x = (torch.randn(16, D) * 0.4).to(torch.bfloat16).cuda()
    y = m(x)
    ref = _ref_layer_norm(x, src_w, None, m.variance_epsilon)
    torch.testing.assert_close(y, ref, atol=2e-2, rtol=2e-2)


@cuda_only
def test_phyai_kernel_fp32_path():
    """fp32 path on phyai-kernel: tighter tolerance."""
    torch.manual_seed(4)
    D = 1024
    m = LayerNorm(D, backend="phyai-kernel", dtype=torch.float32).cuda()
    src_w = (torch.randn(D) * 0.05 + 1.0).cuda()
    src_b = (torch.randn(D) * 0.02).cuda()
    m.weight.data.copy_(src_w)
    m.bias.data.copy_(src_b)

    x = (torch.randn(8, D) * 0.5).cuda()
    y = m(x)
    ref = _ref_layer_norm(x, src_w, src_b, m.variance_epsilon)
    torch.testing.assert_close(y, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# AdaRMSNorm — stateless cond vs precomputed modulation (CPU torch backend)   #
# --------------------------------------------------------------------------- #
#
# The Triton-kernel equivalence tests live in the CUDA-gated
# ``phyai-kernel/tests/test_adarmsnorm.py``. These run on the default CPU
# suite via the ``"torch"`` backend, guarding the stateless refactor
# (project_modulation + forward(modulation=...)) where CI actually runs.


@pytest.fixture
def _adarms_linear_env():
    """Bootstrap the linear dispatcher + a degenerate ``"model"`` mesh.

    ``AdaRMSNorm.dense`` is a :class:`ReplicatedLinear`, whose ``__init__``
    consults the linear-dispatcher singleton and the ``"model"`` mesh — so
    they must exist before construction (same setup as
    ``phyai-kernel/tests/conftest.py``). ``LayerNorm`` has no such linear, so
    this is scoped to the AdaRMSNorm tests rather than module-autouse.
    """
    saved = dict(_meshes)
    tm = MagicMock()
    tm.mesh_dim_names = ()
    tm.size.side_effect = lambda axis=None: 1
    tm.get_local_rank.side_effect = lambda axis=None: 0
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    register_mesh(Mesh(tm, name="model"))
    L.init(register_flashinfer=False, validate=False)
    try:
        yield
    finally:
        _meshes.clear()
        _meshes.update(saved)
        L._reset_for_test()


def _make_cpu_adarms(hidden: int, cond_dim: int) -> AdaRMSNorm:
    m = AdaRMSNorm(
        hidden_size=hidden,
        cond_dim=cond_dim,
        eps=1e-6,
        backend="torch",
        dtype=torch.float32,
        device="cpu",
    )
    with torch.no_grad():
        m.dense.weight.normal_(0.0, 0.05)
        m.dense.bias.normal_(0.0, 0.05)
    return m


def test_adarmsnorm_project_modulation_is_pure_and_shaped(_adarms_linear_env):
    """``project_modulation`` returns a ``(K, 3*D)`` table and stores nothing."""
    torch.manual_seed(0)
    hidden, cond_dim, k = 64, 48, 5
    m = _make_cpu_adarms(hidden, cond_dim)
    conds = torch.randn(k, cond_dim)
    mod = m.project_modulation(conds)
    assert mod.shape == (k, 3 * hidden)
    # Pure: no cache attribute left on the module.
    assert not hasattr(m, "_mod_cache")
    # Matches the dense projection directly.
    ref, _ = m.dense(conds)
    torch.testing.assert_close(mod, ref, atol=1e-6, rtol=1e-6)


def test_adarmsnorm_modulation_matches_cond_path_cpu(_adarms_linear_env):
    """``forward(x, modulation=row)`` equals ``forward(x, cond_row)`` broadcast."""
    torch.manual_seed(1)
    hidden = cond_dim = 128
    chunk, k = 20, 6
    m = _make_cpu_adarms(hidden, cond_dim)

    conds = torch.randn(k, cond_dim)
    mod = m.project_modulation(conds)

    for i in (0, 2, k - 1):
        x = torch.randn(chunk, hidden)
        out_mod, gate_mod = m(x, modulation=mod[i : i + 1])
        out_ref, gate_ref = m(x, conds[i : i + 1].expand(chunk, -1))
        torch.testing.assert_close(out_mod, out_ref, atol=1e-5, rtol=1e-5)
        assert gate_mod.shape == (1, hidden)
        torch.testing.assert_close(
            gate_mod.expand(chunk, -1).contiguous(), gate_ref, atol=1e-5, rtol=1e-5
        )


def test_adarmsnorm_requires_exactly_one_of_cond_or_modulation_cpu(_adarms_linear_env):
    torch.manual_seed(2)
    hidden = cond_dim = 32
    m = _make_cpu_adarms(hidden, cond_dim)
    x = torch.randn(4, hidden)

    with pytest.raises(ValueError, match="exactly one"):
        m(x)  # neither
    cond = torch.randn(4, cond_dim)
    mod = m.project_modulation(cond)
    with pytest.raises(ValueError, match="exactly one"):
        m(x, cond, modulation=mod[:1])  # both
