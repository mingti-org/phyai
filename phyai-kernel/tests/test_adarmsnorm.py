"""Numerical-equivalence tests for the Triton AdaRMSNorm kernel.

Validates :func:`phyai_kernel.adarmsnorm` against:

* an eager torch reference that mirrors lerobot ``PiGemmaRMSNorm`` semantics
  (``forward(x, cond) -> (out, gate)`` with ``out = normed * (1 + scale)
  + shift`` and ``gate = chunk(modulation, 3, dim=-1)[2]``),
* the same reference wrapped with ``torch.compile(mode="reduce-overhead")``
  — both because that's a realistic alternative the user would otherwise
  reach for, and because ``torch.compile``'s fp32-reduction fusions can
  diverge slightly from naive eager and the kernel must match either
  within tolerance.

Test grid covers:

* hidden sizes from 256 (Gemma head_dim) up through 8192 (single-block
  boundary) and 12288 (forces the two-pass kernel),
* per-batch broadcast (``x=(B, S, D)``, ``cond=(B, cond_dim)``) and
  per-token broadcast (``x=(B*S, D)``, ``cond=(B*S, cond_dim)``),
* fp16 / bf16 / fp32.
"""

from __future__ import annotations

import pytest
import torch

import phyai_kernel
import phyai_kernel.triton.ada_rms_norm as triton_adarms_mod

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA is required for phyai-kernel Triton tests", allow_module_level=True
    )


# --------------------------------------------------------------------------- #
# Reference (mirrors lerobot ``PiGemmaRMSNorm.forward`` exactly)              #
# --------------------------------------------------------------------------- #


def _ref_adarmsnorm(
    x: torch.Tensor,
    modulation: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager torch reference. Modulation already broadcast-shaped vs ``x``."""
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    out = xf * (1.0 + scale.float()) + shift.float()
    return out.to(dtype), gate.to(dtype)


def _broadcast_modulation(x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
    """Mirror lerobot's ``unsqueeze(1)`` for 3-D ``x`` x 2-D ``modulation``."""
    if x.dim() == 3 and modulation.dim() == 2:
        return modulation.unsqueeze(1)
    return modulation


# Pre-compiled torch reference, lazily initialised so tests that don't need
# it skip the compile cost.
_compiled_ref: dict[str, callable] = {}


def _ref_adarmsnorm_compiled(x: torch.Tensor, modulation: torch.Tensor, eps: float):
    """Same math, but routed through ``torch.compile``."""
    key = "default"
    fn = _compiled_ref.get(key)
    if fn is None:
        fn = torch.compile(_ref_adarmsnorm, mode="reduce-overhead", dynamic=True)
        _compiled_ref[key] = fn
    return fn(x, modulation, eps)


# --------------------------------------------------------------------------- #
# Shapes                                                                      #
# --------------------------------------------------------------------------- #


_HIDDEN_SIZES = [
    256,  # Gemma head_dim
    1024,  # gemma_300m action expert width (pi0.5)
    2048,  # gemma_2b LM width (pi0.5 prefix)
    3072,  # awkward shape
    8192,  # single-block boundary
    12288,  # forces two-pass kernel
]
_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _tols(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return (5e-5, 5e-5)
    if dtype == torch.bfloat16:
        return (2e-2, 2e-2)
    return (1e-3, 1e-3)


def _make_inputs(
    *,
    leading: tuple[int, ...],
    cond_leading: tuple[int, ...],
    hidden: int,
    cond_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(x, modulation)``.

    ``modulation`` is allocated with shape ``cond_leading + (3 * hidden,)``;
    the test caller picks ``cond_leading`` to exercise the broadcast pattern
    (``cond_leading = leading[:1]`` for per-batch, or ``cond_leading =
    leading`` for per-token).
    """
    torch.manual_seed(0xC0DE * (hidden + sum(leading)) + cond_dim)
    x = torch.randn(*leading, hidden, device="cuda", dtype=dtype) * 0.5
    # Modulation in the same dtype as x — most realistic since
    # ``self.dense`` runs in the activation dtype during inference.
    modulation = (
        torch.randn(*cond_leading, 3 * hidden, device="cuda", dtype=dtype) * 0.1
    )
    return x, modulation


# --------------------------------------------------------------------------- #
# Per-token mapping (cond_leading == x_leading)                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hidden", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_adarmsnorm_2d_per_token(hidden: int, dtype: torch.dtype):
    """``x=(N, D)``, ``modulation=(N, 3D)`` — 1:1 mapping."""
    x, mod = _make_inputs(
        leading=(17,), cond_leading=(17,), hidden=hidden, cond_dim=hidden, dtype=dtype
    )
    eps = 1e-6
    expected_out, expected_gate = _ref_adarmsnorm(x, mod, eps)
    actual_out, actual_gate = phyai_kernel.adarmsnorm(x, mod, eps)
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual_out, expected_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=rtol, atol=atol)


@pytest.mark.parametrize("hidden", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_adarmsnorm_3d_per_token(hidden: int, dtype: torch.dtype):
    """``x=(B, S, D)``, ``modulation=(B, S, 3D)`` — full per-token cond."""
    x, mod = _make_inputs(
        leading=(2, 7),
        cond_leading=(2, 7),
        hidden=hidden,
        cond_dim=hidden,
        dtype=dtype,
    )
    eps = 1e-6
    expected_out, expected_gate = _ref_adarmsnorm(x, mod, eps)
    actual_out, actual_gate = phyai_kernel.adarmsnorm(x, mod, eps)
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual_out, expected_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Per-batch broadcast (``cond_leading=(B,)`` -> broadcasts across S)            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("hidden", _HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_adarmsnorm_3d_broadcast_over_seq(hidden: int, dtype: torch.dtype):
    """``x=(B, S, D)`` x ``modulation=(B, 3D)``: pi0.5 action-expert pattern.

    The kernel infers ``group_size = S`` and broadcasts each modulation row
    across the sequence axis. Gate output shape mirrors the broadcast-shaped
    modulation, ``(B, 1, D)``, so ``residual + out * gate`` lifts correctly.
    """
    B, S = 3, 13
    x = torch.randn(B, S, hidden, device="cuda", dtype=dtype) * 0.5
    modulation = torch.randn(B, 3 * hidden, device="cuda", dtype=dtype) * 0.1
    # Reference: unsqueeze for 3-D x.
    expected_out, expected_gate = _ref_adarmsnorm(x, modulation.unsqueeze(1), eps=1e-6)
    actual_out, actual_gate = phyai_kernel.adarmsnorm(
        x, modulation.unsqueeze(1), eps=1e-6
    )
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual_out, expected_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=rtol, atol=atol)
    assert actual_gate.shape == (B, 1, hidden)


# --------------------------------------------------------------------------- #
# Match torch.compile reference                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "hidden",
    [256, 1024, 2048],  # smaller grid — torch.compile is slow to warm up
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_adarmsnorm_matches_torch_compile(hidden: int, dtype: torch.dtype):
    """The Triton kernel matches a ``torch.compile``'d reference at the same tols."""
    x = torch.randn(2, 7, hidden, device="cuda", dtype=dtype) * 0.5
    modulation = torch.randn(2, 1, 3 * hidden, device="cuda", dtype=dtype) * 0.1
    eps = 1e-6
    # Compile reference, then run twice — second call uses the cached graph.
    expected_out, expected_gate = _ref_adarmsnorm_compiled(x, modulation, eps)
    actual_out, actual_gate = phyai_kernel.adarmsnorm(x, modulation, eps)
    rtol, atol = _tols(dtype)
    torch.testing.assert_close(actual_out, expected_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #


def test_adarmsnorm_explicit_out_buffers():
    hidden = 1024
    x = torch.randn(4, 6, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    modulation = (
        torch.randn(4, 1, 3 * hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    out = torch.empty_like(x)
    gate = torch.empty(4, 1, hidden, device="cuda", dtype=torch.bfloat16)
    ret_out, ret_gate = phyai_kernel.adarmsnorm(
        x, modulation, eps=1e-6, out=out, gate_out=gate
    )
    assert ret_out.data_ptr() == out.data_ptr()
    assert ret_gate.data_ptr() == gate.data_ptr()
    expected_out, expected_gate = _ref_adarmsnorm(x, modulation, 1e-6)
    torch.testing.assert_close(ret_out, expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ret_gate, expected_gate, rtol=2e-2, atol=2e-2)


def test_adarmsnorm_zero_rows_no_launch():
    hidden = 1024
    x = torch.empty(0, hidden, device="cuda", dtype=torch.float16)
    modulation = torch.empty(0, 3 * hidden, device="cuda", dtype=torch.float16)
    out, gate = phyai_kernel.adarmsnorm(x, modulation, eps=1e-6)
    assert out.shape == (0, hidden)
    assert gate.shape == (0, hidden)


def test_adarmsnorm_single_block_boundary():
    """Confirm two-pass kernel matches single-block at the threshold."""
    threshold = triton_adarms_mod._SINGLE_BLOCK_MAX
    for n_cols in (threshold, threshold + 256):
        x = torch.randn(4, n_cols, device="cuda", dtype=torch.float16) * 0.5
        modulation = (
            torch.randn(4, 3 * n_cols, device="cuda", dtype=torch.float16) * 0.1
        )
        expected_out, expected_gate = _ref_adarmsnorm(x, modulation, 1e-6)
        actual_out, actual_gate = phyai_kernel.adarmsnorm(x, modulation, 1e-6)
        torch.testing.assert_close(actual_out, expected_out, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(actual_gate, expected_gate, rtol=1e-3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #


def test_cpu_input_raises():
    x = torch.randn(2, 64)
    mod = torch.randn(2, 192)
    with pytest.raises(RuntimeError, match="must live on CUDA"):
        phyai_kernel.adarmsnorm(x, mod)


def test_modulation_dim_mismatch_raises():
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    mod = torch.randn(2, 100, device="cuda", dtype=torch.bfloat16)  # not 3*64
    with pytest.raises(RuntimeError, match="modulation last dim"):
        phyai_kernel.adarmsnorm(x, mod)


def test_non_divisible_groups_raises():
    """``N_total`` must be a multiple of ``N_mod``."""
    x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
    mod = torch.randn(
        2, 192, device="cuda", dtype=torch.bfloat16
    )  # 7 not multiple of 2
    with pytest.raises(RuntimeError, match="non-zero multiple"):
        phyai_kernel.adarmsnorm(x, mod)


def test_out_shape_mismatch_raises():
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    mod = torch.randn(2, 192, device="cuda", dtype=torch.bfloat16)
    bad_out = torch.empty(2, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="`out` must match"):
        phyai_kernel.adarmsnorm(x, mod, out=bad_out)


def test_gate_out_shape_mismatch_raises():
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
    mod = torch.randn(2, 192, device="cuda", dtype=torch.bfloat16)
    bad_gate = torch.empty(2, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="`gate_out` must have shape"):
        phyai_kernel.adarmsnorm(x, mod, gate_out=bad_gate)


# --------------------------------------------------------------------------- #
# Module-level smoke (phyai.layers.AdaRMSNorm wraps the kernel)                #
# --------------------------------------------------------------------------- #


def test_phyai_layers_adarmsnorm_module_matches_reference():
    pytest.importorskip("phyai.layers")
    from phyai.layers import AdaRMSNorm

    hidden, cond_dim = 1024, 1024
    layer = (
        AdaRMSNorm(
            hidden_size=hidden,
            cond_dim=cond_dim,
            eps=1e-6,
            backend="phyai-kernel",
            prefix="model.layers.0.input_layernorm",
        )
        .cuda()
        .to(torch.bfloat16)
    )

    # Override zero-init so the test exercises a non-trivial modulation.
    with torch.no_grad():
        layer.dense.weight.normal_(0.0, 0.05)
        layer.dense.bias.normal_(0.0, 0.05)

    x = torch.randn(3, 11, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    cond = torch.randn(3, cond_dim, device="cuda", dtype=torch.bfloat16) * 0.5

    actual_out, actual_gate = layer(x, cond)

    # Reference path: re-run the projection and the eager math.
    # ReplicatedLinear.forward returns (y, optional_bias) — unpack the
    # tensor before broadcasting over the seq axis.
    modulation = layer.dense(cond)[0].unsqueeze(1)
    expected_out, expected_gate = _ref_adarmsnorm(x, modulation, layer.variance_epsilon)
    torch.testing.assert_close(actual_out, expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=2e-2, atol=2e-2)
    assert actual_gate.shape == (3, 1, hidden)


def test_phyai_layers_adarmsnorm_torch_backend_matches_kernel_backend():
    pytest.importorskip("phyai.layers")
    from phyai.layers import AdaRMSNorm

    hidden, cond_dim = 256, 256
    layer_kern = (
        AdaRMSNorm(
            hidden_size=hidden,
            cond_dim=cond_dim,
            backend="phyai-kernel",
            prefix="m.l0.in",
        )
        .cuda()
        .to(torch.bfloat16)
    )
    layer_torch = (
        AdaRMSNorm(
            hidden_size=hidden,
            cond_dim=cond_dim,
            backend="torch",
            prefix="m.l0.in",
        )
        .cuda()
        .to(torch.bfloat16)
    )
    # Sync weights so the two backends see identical inputs to the math.
    with torch.no_grad():
        layer_kern.dense.weight.normal_(0.0, 0.05)
        layer_kern.dense.bias.normal_(0.0, 0.05)
        layer_torch.dense.weight.copy_(layer_kern.dense.weight)
        layer_torch.dense.bias.copy_(layer_kern.dense.bias)

    x = torch.randn(2, 5, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    cond = torch.randn(2, cond_dim, device="cuda", dtype=torch.bfloat16) * 0.5

    out_k, gate_k = layer_kern(x, cond)
    out_t, gate_t = layer_torch(x, cond)
    torch.testing.assert_close(out_k, out_t, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gate_k, gate_t, rtol=2e-2, atol=2e-2)


def test_phyai_layers_adarmsnorm_weight_loader_keys():
    """AdaRMSNorm exposes its inner ``dense.weight`` / ``dense.bias`` to
    the generic safetensors loader via ``param.hf_keys`` (the new weight
    loading API replacing the old ``placements()`` method)."""
    pytest.importorskip("phyai.layers")
    from phyai.layers import AdaRMSNorm

    layer = AdaRMSNorm(
        hidden_size=64, cond_dim=64, prefix="m.l0.input_layernorm", backend="torch"
    )
    keys: set[tuple[str, str]] = set()
    for name, param in layer.named_parameters():
        for hf_key, _shard_id in getattr(param, "hf_keys", []):
            keys.add((hf_key, name))
    assert keys == {
        ("m.l0.input_layernorm.dense.weight", "dense.weight"),
        ("m.l0.input_layernorm.dense.bias", "dense.bias"),
    }


def test_phyai_layers_adarmsnorm_rejects_flashinfer():
    pytest.importorskip("phyai.layers")
    from phyai.layers import AdaRMSNorm

    with pytest.raises(ValueError, match="flashinfer"):
        AdaRMSNorm(hidden_size=64, cond_dim=64, backend="flashinfer", prefix="x")


# --------------------------------------------------------------------------- #
# Precomputed-modulation path (forward(x, modulation=...) == forward(x, cond))  #
# --------------------------------------------------------------------------- #


def _make_adarms(hidden: int, cond_dim: int) -> "object":
    from phyai.layers import AdaRMSNorm

    layer = (
        AdaRMSNorm(
            hidden_size=hidden,
            cond_dim=cond_dim,
            eps=1e-6,
            backend="phyai-kernel",
            prefix="model.layers.0.input_layernorm",
        )
        .cuda()
        .to(torch.bfloat16)
    )
    with torch.no_grad():
        layer.dense.weight.normal_(0.0, 0.05)
        layer.dense.bias.normal_(0.0, 0.05)
    return layer


def test_phyai_layers_adarmsnorm_modulation_matches_dense_path():
    """``forward(x, modulation=row)`` (precomputed, broadcast over all rows)
    matches the standard ``forward(x, cond)`` projection path.

    The conditioning is a fixed set of ``K`` rows; every token in one step
    shares the same row, so a single precomputed modulation row broadcast
    across the tokens must equal projecting that row per token. Near-, not
    bit-exact (the projection runs at a different row count), so use the file
    tolerance.
    """
    pytest.importorskip("phyai.layers")

    hidden = cond_dim = 1024
    chunk = 50
    n_steps = 10
    layer = _make_adarms(hidden, cond_dim)

    conds = torch.randn(n_steps, cond_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    mod = layer.project_modulation(conds)
    assert tuple(mod.shape) == (n_steps, 3 * hidden)

    for i in (0, 3, n_steps - 1):
        x = torch.randn(chunk, hidden, device="cuda", dtype=torch.bfloat16) * 0.5

        out_idx, gate_idx = layer(x, modulation=mod[i : i + 1])

        # Reference: the dense path with this row broadcast across all tokens,
        # exactly as the runtime did before the precompute (every row equal).
        cond_rows = conds[i : i + 1].expand(chunk, -1)
        out_ref, gate_ref = layer(x, cond_rows)

        torch.testing.assert_close(out_idx, out_ref, rtol=2e-2, atol=2e-2)
        # The lookup gate is a single broadcast row; compare against any
        # reference row (they are all identical) and check the shape.
        assert gate_idx.shape == (1, hidden)
        torch.testing.assert_close(
            gate_idx.expand(chunk, -1).contiguous(),
            gate_ref,
            rtol=2e-2,
            atol=2e-2,
        )


def test_phyai_layers_adarmsnorm_modulation_gate_broadcast_in_addcmul():
    """A ``(1, D)`` lookup gate broadcasts through ``torch.addcmul`` to the
    same result as a full ``(N, D)`` gate (the gated-residual use site)."""
    pytest.importorskip("phyai.layers")

    hidden = cond_dim = 256
    chunk = 50
    layer = _make_adarms(hidden, cond_dim)
    conds = torch.randn(4, cond_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    mod = layer.project_modulation(conds)

    x = torch.randn(chunk, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    residual = torch.randn(chunk, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    out_idx, gate_idx = layer(x, modulation=mod[2:3])  # gate_idx: (1, hidden)

    gated_broadcast = torch.addcmul(residual, out_idx, gate_idx)
    gated_full = torch.addcmul(residual, out_idx, gate_idx.expand(chunk, -1))
    torch.testing.assert_close(gated_broadcast, gated_full, rtol=0, atol=0)


def test_phyai_layers_adarmsnorm_modulation_3d_input():
    """The precomputed path also serves 3-D ``x`` ``(B, S, D)`` by broadcasting
    the single modulation row over both leading axes."""
    pytest.importorskip("phyai.layers")

    hidden = cond_dim = 256
    layer = _make_adarms(hidden, cond_dim)
    conds = torch.randn(6, cond_dim, device="cuda", dtype=torch.bfloat16) * 0.5
    mod = layer.project_modulation(conds)

    x = torch.randn(2, 7, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    out_idx, gate_idx = layer(x, modulation=mod[1:2])

    cond_rows = conds[1].reshape(1, 1, cond_dim).expand(2, 7, -1)
    out_ref, _ = layer(x, cond_rows)
    torch.testing.assert_close(out_idx, out_ref, rtol=2e-2, atol=2e-2)
    assert out_idx.shape == (2, 7, hidden)
    assert gate_idx.shape == (1, 1, hidden)


def test_phyai_layers_adarmsnorm_requires_exactly_one_of_cond_or_modulation():
    pytest.importorskip("phyai.layers")

    hidden = cond_dim = 128
    layer = _make_adarms(hidden, cond_dim)
    x = torch.randn(4, hidden, device="cuda", dtype=torch.bfloat16)

    # Neither cond nor modulation.
    with pytest.raises(ValueError, match="exactly one"):
        layer(x)

    # Both cond and modulation.
    cond = torch.randn(4, cond_dim, device="cuda", dtype=torch.bfloat16)
    mod = layer.project_modulation(cond)
    with pytest.raises(ValueError, match="exactly one"):
        layer(x, cond, modulation=mod[:1])
