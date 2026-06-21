"""Tests for :class:`phyai.layers.attention.attention.Attention`.

Covers the three no-cache backends — ``"eager"``, ``"sdpa"``, and
``"flashinfer"`` — across the padded (4-D) and ragged (3-D)
dispatch paths plus the ``ctx=None`` convenience flow used by the
vision tower. Numerical agreement between eager and sdpa is the
primary correctness signal; flashinfer is gated on GPU + flashinfer
import.
"""

from __future__ import annotations

import pytest
import torch

from phyai.layers.attention import (
    Attention,
    AttnCtx,
    AttnLayout,
    AttnMetadata,
    AttnMode,
)


def _has_flashinfer() -> bool:
    try:
        import flashinfer.prefill  # noqa: F401

        return True
    except ImportError:
        return False


def _can_use_flashinfer() -> bool:
    return torch.cuda.is_available() and _has_flashinfer()


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_construct_attention(backend: str):
    attn = Attention(
        num_heads=4,
        head_dim=16,
        num_kv_heads=2,
        backend=backend,
        causal=True,
        backend_kwargs={"compile": False} if backend == "sdpa" else None,
    )
    assert attn.backend == backend
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2
    assert attn.head_dim == 16
    assert attn.causal is True


def test_attention_rejects_invalid_backend():
    with pytest.raises(ValueError, match="not registered"):
        Attention(num_heads=4, head_dim=16, backend="not-a-backend")


def test_attention_rejects_bad_gqa():
    with pytest.raises(ValueError, match="must be a positive multiple"):
        Attention(
            num_heads=4,
            head_dim=16,
            num_kv_heads=3,
            backend="eager",
        )


def test_attention_rejects_swa_without_causal():
    with pytest.raises(ValueError, match="sliding_window requires causal"):
        Attention(
            num_heads=4,
            head_dim=16,
            sliding_window=4,
            causal=False,
            backend="eager",
        )


# --------------------------------------------------------------------- #
# ctx=None convenience path (vision tower / tests)                      #
# --------------------------------------------------------------------- #


def test_padded_4d_convenience_path():
    """When ctx=None, layer infers PADDED_4D from q.ndim==4 and lazily
    builds a default backend + AttnCtx in-place."""
    torch.manual_seed(0)
    B, S, H, D = 2, 8, 4, 16
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    attn = Attention(num_heads=H, head_dim=D, backend="eager", causal=True)
    out = attn(q, k, v)
    assert out.shape == (B, S, H, D)


def test_ragged_3d_convenience_path():
    torch.manual_seed(1)
    H, D = 4, 16
    cu_q = torch.tensor([0, 5, 12], dtype=torch.int32)
    N = int(cu_q[-1])
    q = torch.randn(N, H, D)
    k = torch.randn(N, H, D)
    v = torch.randn(N, H, D)
    attn = Attention(num_heads=H, head_dim=D, backend="eager", causal=True)
    out = attn(q, k, v, cu_seqlens_q=cu_q)
    assert out.shape == (N, H, D)


def test_ragged_without_cu_seqlens_raises():
    """3-D q without cu_seqlens_q must raise (ctx=None convenience path)."""
    H, D = 2, 8
    q = torch.randn(4, H, D)
    attn = Attention(num_heads=H, head_dim=D, backend="eager", causal=True)
    with pytest.raises(ValueError, match="ragged forward requires cu_seqlens_q"):
        attn(q, q, q)


def test_invalid_q_rank_raises():
    H, D = 2, 4
    q = torch.randn(2, 4, H, D, 1)  # 5-D
    attn = Attention(num_heads=H, head_dim=D, backend="eager")
    with pytest.raises(ValueError, match="q must be 3-D .ragged. or 4-D"):
        attn(q, q, q)


# --------------------------------------------------------------------- #
# Numerical correctness — eager vs sdpa                                 #
# --------------------------------------------------------------------- #


def test_eager_sdpa_padded_match_non_causal():
    torch.manual_seed(2)
    B, S, H, D = 2, 6, 4, 16
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    eager = Attention(num_heads=H, head_dim=D, backend="eager", causal=False)
    sdpa = Attention(
        num_heads=H,
        head_dim=D,
        backend="sdpa",
        causal=False,
        backend_kwargs={"compile": False},
    )
    out_e = eager(q, k, v)
    out_s = sdpa(q, k, v)
    assert torch.allclose(out_e, out_s, atol=1e-5, rtol=1e-4)


def test_eager_sdpa_padded_match_causal():
    torch.manual_seed(3)
    B, S, H, D = 1, 8, 4, 16
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    eager = Attention(num_heads=H, head_dim=D, backend="eager", causal=True)
    sdpa = Attention(
        num_heads=H,
        head_dim=D,
        backend="sdpa",
        causal=True,
        backend_kwargs={"compile": False},
    )
    out_e = eager(q, k, v)
    out_s = sdpa(q, k, v)
    assert torch.allclose(out_e, out_s, atol=1e-5, rtol=1e-4)


def test_eager_sdpa_padded_match_gqa():
    torch.manual_seed(4)
    B, S, H, H_kv, D = 1, 6, 4, 2, 16
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H_kv, D)
    v = torch.randn(B, S, H_kv, D)
    eager = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        backend="eager",
        causal=False,
    )
    sdpa = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        backend="sdpa",
        causal=False,
        backend_kwargs={"compile": False},
    )
    out_e = eager(q, k, v)
    out_s = sdpa(q, k, v)
    assert torch.allclose(out_e, out_s, atol=1e-5, rtol=1e-4)


def test_sdpa_ragged_raises():
    """SDPA is padded-only — ragged (3-D varlen) input must raise and point
    to flashinfer. SDPA has no varlen API; varlen is flashinfer's job."""
    torch.manual_seed(5)
    H, D = 4, 16
    cu_q = torch.tensor([0, 5, 12], dtype=torch.int32)
    N = int(cu_q[-1])
    q = torch.randn(N, H, D)
    k = torch.randn(N, H, D)
    v = torch.randn(N, H, D)
    sdpa = Attention(
        num_heads=H,
        head_dim=D,
        backend="sdpa",
        causal=False,
        backend_kwargs={"compile": False},
    )
    with pytest.raises(NotImplementedError, match="flashinfer"):
        sdpa(q, k, v, cu_seqlens_q=cu_q)


# --------------------------------------------------------------------- #
# Causal / SWA / soft-cap correctness (vs eager reference)              #
# --------------------------------------------------------------------- #


def test_sliding_window_zeros_above_window():
    """A window of 1 means each query attends only to its own position."""
    torch.manual_seed(6)
    B, S, H, D = 1, 6, 2, 8
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    attn = Attention(
        num_heads=H,
        head_dim=D,
        backend="eager",
        causal=True,
        sliding_window=1,
    )
    out = attn(q, k, v)
    # With window=1, output token i = (q_i · k_i) softmax over single key.
    # Since softmax over a single value is 1, output_i should equal v_i.
    expected = v
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-4)


def test_logits_soft_cap_changes_output():
    """Soft-cap with finite cap must produce different output than no cap."""
    torch.manual_seed(7)
    B, S, H, D = 1, 4, 2, 8
    q = torch.randn(B, S, H, D) * 5
    k = torch.randn(B, S, H, D) * 5
    v = torch.randn(B, S, H, D)
    no_cap = Attention(
        num_heads=H,
        head_dim=D,
        backend="eager",
        causal=False,
    )
    capped = Attention(
        num_heads=H,
        head_dim=D,
        backend="eager",
        causal=False,
        logits_soft_cap=1.0,
    )
    out_nc = no_cap(q, k, v)
    out_cap = capped(q, k, v)
    assert not torch.allclose(out_nc, out_cap, atol=1e-3)


# --------------------------------------------------------------------- #
# Explicit ctx (advanced path used by callers that own the backend)     #
# --------------------------------------------------------------------- #


def test_explicit_ctx_padded_idle_returns_zeros():
    """IDLE mode bypasses the kernel and returns zeros."""
    B, S, H, D = 2, 4, 2, 8
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    attn = Attention(num_heads=H, head_dim=D, backend="eager")
    backend = attn._ensure_backend()
    plan = backend.init_forward_metadata(
        AttnMetadata(
            mode=AttnMode.IDLE,
            layout=AttnLayout.PADDED_4D,
            batch_size=B,
            num_query_tokens=B * S,
        )
    )
    ctx = AttnCtx(
        backend=backend,
        plan=plan,
        mode=AttnMode.IDLE,
        layout=AttnLayout.PADDED_4D,
    )
    out = attn(q, k, v, ctx=ctx)
    assert torch.equal(out, torch.zeros_like(q))


# --------------------------------------------------------------------- #
# flashinfer (GPU-gated)                                                #
# --------------------------------------------------------------------- #


@pytest.mark.skipif(
    not _can_use_flashinfer(),
    reason="flashinfer requires CUDA + flashinfer-python.",
)
def test_flashinfer_padded_b1_matches_eager():
    """B=1 single-prefill path through flashinfer matches eager."""
    torch.manual_seed(8)
    B, S, H, D = 1, 6, 4, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
    fi = Attention(num_heads=H, head_dim=D, backend="flashinfer", causal=True)
    out_fi = fi(q, k, v)
    eager = Attention(num_heads=H, head_dim=D, backend="eager", causal=True)
    out_e = eager(q.float(), k.float(), v.float())
    assert torch.allclose(out_fi.float(), out_e, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------- #
# Rectangular cross-attention: 4-D padded with S_q != S_kv              #
# --------------------------------------------------------------------- #


def test_eager_sdpa_padded_match_rectangular():
    """4-D padded with S_q != S_kv (cross-attention) — eager vs sdpa, B>1 + GQA."""
    torch.manual_seed(10)
    B, S_q, S_kv, H, H_kv, D = 2, 5, 9, 4, 2, 16
    q = torch.randn(B, S_q, H, D)
    k = torch.randn(B, S_kv, H_kv, D)
    v = torch.randn(B, S_kv, H_kv, D)
    eager = Attention(
        num_heads=H, head_dim=D, num_kv_heads=H_kv, backend="eager", causal=False
    )
    sdpa = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        backend="sdpa",
        causal=False,
        backend_kwargs={"compile": False},
    )
    out_e = eager(q, k, v)
    out_s = sdpa(q, k, v)
    assert out_e.shape == (B, S_q, H, D)
    assert torch.allclose(out_e, out_s, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="needs CUDA"
            ),
        ),
    ],
)
def test_sdpa_select_kernel_matches_default(device: str):
    """The CUDA kernel-priority context must not perturb results.

    ``select_kernel`` only biases which fused kernel CUDA dispatches to; it is
    a no-op on CPU. Same inputs through ``select_kernel=True`` and ``False``
    must agree. On CPU the context is a no-op (trivially exact); the ``cuda``
    parametrization is the real guard — it enters the ``sdpa_kernel`` priority
    context and checks the chosen kernel still matches default dispatch.
    """
    torch.manual_seed(13)
    # head_dim=64 + fp16 on CUDA exercises a real fused kernel; CPU uses fp32.
    B, S, H, H_kv, D = 2, 8, 4, 2, 64
    dtype = torch.float16 if device == "cuda" else torch.float32
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H_kv, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H_kv, D, device=device, dtype=dtype)
    sel = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        backend="sdpa",
        causal=True,
        backend_kwargs={"compile": False, "select_kernel": True},
    )
    nosel = Attention(
        num_heads=H,
        head_dim=D,
        num_kv_heads=H_kv,
        backend="sdpa",
        causal=True,
        backend_kwargs={"compile": False, "select_kernel": False},
    )
    out_sel = sel(q, k, v)
    out_nosel = nosel(q, k, v)
    assert torch.allclose(out_sel, out_nosel, atol=1e-3, rtol=1e-3)


def test_padded_rectangular_matches_ragged():
    """4-D padded rectangular == the same data packed into the 3-D ragged path.

    Confirms the padded path's synthesized uniform cu_seqlens (built in
    ``_build_default_ctx``) describe the same attention as explicit ragged
    cu_seqlens — i.e. a 4-D ``attn(q, k, v)`` with S_q != S_kv needs no manual
    packing.
    """
    torch.manual_seed(11)
    B, S_q, S_kv, H, D = 2, 4, 7, 4, 16
    q = torch.randn(B, S_q, H, D)
    k = torch.randn(B, S_kv, H, D)
    v = torch.randn(B, S_kv, H, D)
    attn = Attention(num_heads=H, head_dim=D, backend="eager", causal=False)
    out_padded = attn(q, k, v)
    cu_q = torch.arange(0, (B + 1) * S_q, S_q, dtype=torch.int32)
    cu_kv = torch.arange(0, (B + 1) * S_kv, S_kv, dtype=torch.int32)
    out_ragged = attn(
        q.reshape(B * S_q, H, D),
        k.reshape(B * S_kv, H, D),
        v.reshape(B * S_kv, H, D),
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
    ).reshape(B, S_q, H, D)
    assert torch.allclose(out_padded, out_ragged, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(
    not _can_use_flashinfer(),
    reason="flashinfer requires CUDA + flashinfer-python.",
)
def test_flashinfer_padded_rectangular_matches_eager():
    """flashinfer 4-D padded with S_q != S_kv matches eager, for B==1 and B>1.

    B==1 routes through ``single_prefill`` (already rectangular); B>1 exercises
    the synthesized padded cu_seqlens + ragged-KV plan (the gap the fix closes —
    B>1 padded raised before). This is the regression guard for cosmos3's
    cross-attention after dropping its hand-rolled ``_attend``.
    """
    torch.manual_seed(12)
    H, D, S_q, S_kv = 4, 64, 5, 9
    for B in (1, 2):
        q = torch.randn(B, S_q, H, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, S_kv, H, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, S_kv, H, D, device="cuda", dtype=torch.float16)
        fi = Attention(num_heads=H, head_dim=D, backend="flashinfer", causal=False)
        out_fi = fi(q, k, v)
        eager = Attention(num_heads=H, head_dim=D, backend="eager", causal=False)
        out_e = eager(q.float(), k.float(), v.float())
        assert out_fi.shape == (B, S_q, H, D)
        assert torch.allclose(out_fi.float(), out_e, atol=1e-2, rtol=1e-2)
