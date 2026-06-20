"""RotaryEmbedding — eager / flashinfer parity, rope_type math, shape contract."""

from __future__ import annotations

import math

import pytest
import torch

from phyai.layers.rotary_embedding import (
    ROPE_INV_FREQ_FNS,
    InterleavedMRotaryEmbedding,
    RotaryEmbedding,
    _default_inv_freq,
    _linear_inv_freq,
    _llama3_inv_freq,
    apply_rotary_pos_emb,
    compute_cos_sin_from_inv_freq,
    compute_qwen3vl_mrope_cos_sin_from_inv_freq,
    rotate_half,
)


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flashinfer RoPE needs CUDA"
)


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _ref_cos_sin(
    pos_ids: torch.Tensor, rotary_dim: int, theta: float, *, double: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference cos/sin for rotate-half RoPE."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    freqs = torch.outer(pos_ids.float().reshape(-1), inv_freq).reshape(
        *pos_ids.shape, -1
    )
    cos, sin = freqs.cos(), freqs.sin()
    if double:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def _ref_rotate_half_apply(
    q: torch.Tensor, k: torch.Tensor, pos_ids: torch.Tensor, *, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = q.shape[-1]
    # Broadcast 1-D positions to match q's leading dims (B, S) or (nnz,).
    target_shape = q.shape[:-2]
    if pos_ids.shape != target_shape:
        pos_ids = pos_ids.expand(target_shape)
    cos, sin = _ref_cos_sin(pos_ids, rotary_dim, theta, double=True)
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    # apply_rotary_pos_emb adds the head axis itself via unsqueeze_dim=-2.
    return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=-2)


# ---------------------------------------------------------------------------
# rotate_half / apply_rotary_pos_emb (standalone ops)
# ---------------------------------------------------------------------------


def test_rotate_half_signs():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # [a, b, c, d] -> [-c, -d, a, b]
    torch.testing.assert_close(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rotary_pos_emb_zero_position_is_identity():
    torch.manual_seed(0)
    q = torch.randn(2, 1, 4, 8)
    k = torch.randn(2, 1, 4, 8)
    cos = torch.ones(2, 1, 8)
    sin = torch.zeros(2, 1, 8)
    q2, k2 = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=-2)
    torch.testing.assert_close(q2, q)
    torch.testing.assert_close(k2, k)


# ---------------------------------------------------------------------------
# inv_freq helpers
# ---------------------------------------------------------------------------


def test_default_inv_freq_matches_formula():
    rd, theta = 64, 10000.0
    inv, scale = _default_inv_freq(rd, theta)
    expected = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    torch.testing.assert_close(inv, expected)
    assert scale == 1.0


def test_linear_inv_freq_halves_at_factor_2():
    rd, theta = 64, 10000.0
    base, _ = _default_inv_freq(rd, theta)
    scaled, _ = _linear_inv_freq(rd, theta, factor=2.0)
    torch.testing.assert_close(scaled, base / 2.0)


def test_llama3_inv_freq_matches_manual_recipe():
    rd, theta = 128, 5e5
    factor, low_f, high_f, ctx = 8.0, 1.0, 4.0, 8192
    inv, scale = _llama3_inv_freq(
        rd,
        theta,
        factor=factor,
        low_freq_factor=low_f,
        high_freq_factor=high_f,
        original_max_position_embeddings=ctx,
    )
    base, _ = _default_inv_freq(rd, theta)
    low_w, high_w = ctx / low_f, ctx / high_f
    wavelen = 2 * math.pi / base
    expected = torch.where(wavelen > low_w, base / factor, base)
    smooth = (ctx / wavelen - low_f) / (high_f - low_f)
    smoothed = (1 - smooth) * expected / factor + smooth * expected
    is_medium = ~(wavelen < high_w) & ~(wavelen > low_w)
    expected = torch.where(is_medium, smoothed, expected)
    torch.testing.assert_close(inv, expected)
    assert scale == 1.0


@pytest.mark.parametrize("rope_type", ["yarn", "dynamic", "longrope"])
def test_unsupported_rope_types_raise(rope_type):
    with pytest.raises(NotImplementedError, match="not implemented"):
        ROPE_INV_FREQ_FNS[rope_type](64, 10000.0)


# ---------------------------------------------------------------------------
# RotaryEmbedding constructor
# ---------------------------------------------------------------------------


def test_unknown_rope_type_raises():
    with pytest.raises(ValueError, match="Unknown rope_type"):
        RotaryEmbedding(64, rope_type="banana", backend="eager")


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown RoPE backend"):
        RotaryEmbedding(64, backend="banana")


def test_odd_head_dim_raises():
    with pytest.raises(ValueError, match="head_dim must be a positive even"):
        RotaryEmbedding(63, backend="eager")


def test_partial_rotary_factor_out_of_range():
    with pytest.raises(ValueError, match="partial_rotary_factor"):
        RotaryEmbedding(64, partial_rotary_factor=0.0, backend="eager")
    with pytest.raises(ValueError, match="partial_rotary_factor"):
        RotaryEmbedding(64, partial_rotary_factor=1.5, backend="eager")


def test_extra_repr_contains_key_fields():
    m = RotaryEmbedding(
        64, max_position_embeddings=1024, rope_theta=5e4, backend="eager"
    )
    s = repr(m)
    assert "head_dim=64" in s
    assert "rope_theta=50000.0" in s
    assert "backend='eager'" in s


# ---------------------------------------------------------------------------
# Eager backend: numerical equivalence to manual rotate_half reference
# ---------------------------------------------------------------------------


def test_eager_matches_manual_4d():
    torch.manual_seed(0)
    H_q, H_k, D, S, B = 8, 8, 64, 16, 2
    theta = 10000.0
    q = torch.randn(B, S, H_q, D)
    k = torch.randn(B, S, H_k, D)
    pos = torch.arange(S).unsqueeze(0).expand(B, S)

    m = RotaryEmbedding(
        D, max_position_embeddings=64, rope_theta=theta, backend="eager"
    )
    q_out, k_out = m(pos, q, k)
    q_ref, k_ref = _ref_rotate_half_apply(q, k, pos, theta=theta)
    torch.testing.assert_close(q_out, q_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_out, k_ref, atol=1e-5, rtol=1e-5)


def test_eager_matches_manual_3d_ragged():
    torch.manual_seed(1)
    nnz, H, D = 12, 4, 64
    theta = 10000.0
    q = torch.randn(nnz, H, D)
    k = torch.randn(nnz, H, D)
    pos = torch.arange(nnz)

    m = RotaryEmbedding(
        D, max_position_embeddings=64, rope_theta=theta, backend="eager"
    )
    q_out, k_out = m(pos, q, k)
    q_ref, k_ref = _ref_rotate_half_apply(q, k, pos, theta=theta)
    torch.testing.assert_close(q_out, q_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_out, k_ref, atol=1e-5, rtol=1e-5)


def test_eager_gqa_q_k_diff_heads():
    torch.manual_seed(2)
    B, S, H_q, H_k, D = 2, 8, 8, 2, 64
    q = torch.randn(B, S, H_q, D)
    k = torch.randn(B, S, H_k, D)
    pos = torch.arange(S)

    m = RotaryEmbedding(D, max_position_embeddings=32, backend="eager")
    q_out, k_out = m(pos, q, k)
    q_ref, k_ref = _ref_rotate_half_apply(q, k, pos, theta=10000.0)
    torch.testing.assert_close(q_out, q_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_out, k_ref, atol=1e-5, rtol=1e-5)


def test_eager_partial_rotary_factor_passthrough():
    torch.manual_seed(3)
    B, S, H, D = 1, 4, 2, 64
    partial = 0.5
    rotary_dim = int(D * partial)  # 32
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    pos = torch.arange(S)

    m = RotaryEmbedding(
        D, max_position_embeddings=16, partial_rotary_factor=partial, backend="eager"
    )
    q_out, k_out = m(pos, q, k)

    # Trailing channels untouched.
    torch.testing.assert_close(q_out[..., rotary_dim:], q[..., rotary_dim:])
    torch.testing.assert_close(k_out[..., rotary_dim:], k[..., rotary_dim:])

    # Leading channels = rotate-half on (..., rotary_dim) slice.
    q_ref_lead, k_ref_lead = _ref_rotate_half_apply(
        q[..., :rotary_dim], k[..., :rotary_dim], pos, theta=10000.0
    )
    torch.testing.assert_close(
        q_out[..., :rotary_dim], q_ref_lead, atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        k_out[..., :rotary_dim], k_ref_lead, atol=1e-5, rtol=1e-5
    )


def test_eager_position_ids_1d_broadcasts_over_batch():
    torch.manual_seed(4)
    B, S, H, D = 3, 5, 2, 64
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    pos_1d = torch.arange(S)
    pos_2d = pos_1d.unsqueeze(0).expand(B, S)

    m = RotaryEmbedding(D, max_position_embeddings=16, backend="eager")
    q_a, k_a = m(pos_1d, q, k)
    q_b, k_b = m(pos_2d, q, k)
    torch.testing.assert_close(q_a, q_b)
    torch.testing.assert_close(k_a, k_b)


# ---------------------------------------------------------------------------
# Eager interleave (GPT-J style)
# ---------------------------------------------------------------------------


def test_eager_interleave_matches_manual():
    torch.manual_seed(5)
    nnz, H, D = 6, 2, 64
    q = torch.randn(nnz, H, D)
    k = torch.randn(nnz, H, D)
    pos = torch.arange(nnz)

    m = RotaryEmbedding(D, max_position_embeddings=16, interleave=True, backend="eager")
    q_out, k_out = m(pos, q, k)

    # Manual reference: standard interleaved RoPE rotation by hand.
    cos_h, sin_h = _ref_cos_sin(pos, D, 10000.0, double=False)
    cos_h = cos_h.unsqueeze(-2)  # (nnz, 1, D/2)
    sin_h = sin_h.unsqueeze(-2)

    even_q, odd_q = q[..., 0::2], q[..., 1::2]
    new_even_q = even_q * cos_h - odd_q * sin_h
    new_odd_q = even_q * sin_h + odd_q * cos_h
    q_ref = torch.stack((new_even_q, new_odd_q), dim=-1).flatten(-2)

    even_k, odd_k = k[..., 0::2], k[..., 1::2]
    new_even_k = even_k * cos_h - odd_k * sin_h
    new_odd_k = even_k * sin_h + odd_k * cos_h
    k_ref = torch.stack((new_even_k, new_odd_k), dim=-1).flatten(-2)

    torch.testing.assert_close(q_out, q_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k_out, k_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Shape-contract validation
# ---------------------------------------------------------------------------


def test_mismatched_nnz_raises_3d():
    m = RotaryEmbedding(64, max_position_embeddings=16, backend="eager")
    q = torch.randn(6, 2, 64)
    k = torch.randn(5, 2, 64)
    pos = torch.arange(6)
    with pytest.raises(ValueError, match="ragged token counts differ"):
        m(pos, q, k)


def test_mismatched_seqlen_raises_4d():
    m = RotaryEmbedding(64, max_position_embeddings=16, backend="eager")
    q = torch.randn(2, 6, 4, 64)
    k = torch.randn(2, 5, 4, 64)
    pos = torch.arange(6).unsqueeze(0).expand(2, 6)
    with pytest.raises(ValueError, match="leading.*B, S"):
        m(pos, q, k)


def test_wrong_head_dim_raises():
    m = RotaryEmbedding(64, max_position_embeddings=16, backend="eager")
    q = torch.randn(2, 4, 4, 32)
    k = torch.randn(2, 4, 4, 32)
    pos = torch.arange(4)
    with pytest.raises(ValueError, match="last dim must equal head_dim"):
        m(pos, q, k)


def test_2d_q_raises():
    m = RotaryEmbedding(64, max_position_embeddings=16, backend="eager")
    q = torch.randn(8, 64)
    k = torch.randn(8, 64)
    pos = torch.arange(8)
    with pytest.raises(ValueError, match="3-D .* or 4-D"):
        m(pos, q, k)


# ---------------------------------------------------------------------------
# rope_type wired through the module
# ---------------------------------------------------------------------------


def test_module_linear_scaling_changes_output():
    torch.manual_seed(6)
    D, S, H = 64, 8, 2
    q = torch.randn(1, S, H, D)
    k = torch.randn(1, S, H, D)
    pos = torch.arange(S)

    m_def = RotaryEmbedding(D, max_position_embeddings=16, backend="eager")
    m_lin = RotaryEmbedding(
        D,
        max_position_embeddings=16,
        rope_type="linear",
        rope_scaling={"factor": 2.0},
        backend="eager",
    )
    q_def, _ = m_def(pos, q, k)
    q_lin, _ = m_lin(pos, q, k)
    # Different inv_freq -> different rotated output.
    assert not torch.allclose(q_def, q_lin, atol=1e-3)


# ---------------------------------------------------------------------------
# flashinfer parity
# ---------------------------------------------------------------------------


@cuda_only
def test_flashinfer_matches_eager_4d_bf16():
    torch.manual_seed(7)
    B, S, H_q, H_k, D = 2, 16, 8, 2, 64
    q = (torch.randn(B, S, H_q, D) * 0.1).to(torch.bfloat16).cuda()
    k = (torch.randn(B, S, H_k, D) * 0.1).to(torch.bfloat16).cuda()
    pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, S).contiguous()

    m_eager = RotaryEmbedding(
        D, max_position_embeddings=64, backend="eager", device="cuda"
    )
    m_fi = RotaryEmbedding(
        D, max_position_embeddings=64, backend="flashinfer", device="cuda"
    )
    q_e, k_e = m_eager(pos, q, k)
    q_f, k_f = m_fi(pos, q, k)
    torch.testing.assert_close(q_f, q_e, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_f, k_e, atol=2e-2, rtol=2e-2)


@cuda_only
def test_flashinfer_matches_eager_3d_ragged_bf16():
    torch.manual_seed(8)
    nnz, H_q, H_k, D = 24, 8, 2, 64
    q = (torch.randn(nnz, H_q, D) * 0.1).to(torch.bfloat16).cuda()
    k = (torch.randn(nnz, H_k, D) * 0.1).to(torch.bfloat16).cuda()
    # Two sequences of length 12, positions reset per sequence.
    pos = torch.cat([torch.arange(12), torch.arange(12)]).to("cuda")

    m_eager = RotaryEmbedding(
        D, max_position_embeddings=64, backend="eager", device="cuda"
    )
    m_fi = RotaryEmbedding(
        D, max_position_embeddings=64, backend="flashinfer", device="cuda"
    )
    q_e, k_e = m_eager(pos, q, k)
    q_f, k_f = m_fi(pos, q, k)
    torch.testing.assert_close(q_f, q_e, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_f, k_e, atol=2e-2, rtol=2e-2)


@cuda_only
def test_flashinfer_partial_rotary_matches_eager_bf16():
    torch.manual_seed(9)
    B, S, H, D = 1, 8, 4, 64
    q = (torch.randn(B, S, H, D) * 0.1).to(torch.bfloat16).cuda()
    k = (torch.randn(B, S, H, D) * 0.1).to(torch.bfloat16).cuda()
    pos = torch.arange(S, device="cuda").unsqueeze(0).expand(B, S).contiguous()

    m_eager = RotaryEmbedding(
        D,
        max_position_embeddings=32,
        partial_rotary_factor=0.5,
        backend="eager",
        device="cuda",
    )
    m_fi = RotaryEmbedding(
        D,
        max_position_embeddings=32,
        partial_rotary_factor=0.5,
        backend="flashinfer",
        device="cuda",
    )
    q_e, k_e = m_eager(pos, q, k)
    q_f, k_f = m_fi(pos, q, k)
    torch.testing.assert_close(q_f, q_e, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_f, k_e, atol=2e-2, rtol=2e-2)


@cuda_only
def test_flashinfer_rejects_cpu_input():
    m = RotaryEmbedding(
        64, max_position_embeddings=16, backend="flashinfer", device="cuda"
    )
    q = torch.randn(2, 4, 4, 64)  # CPU
    k = torch.randn(2, 4, 4, 64)
    pos = torch.arange(4)
    with pytest.raises(RuntimeError, match="rotary_emb.to"):
        m(pos, q, k)


# ---------------------------------------------------------------------------
# Static (stateless) compute fns — parity vs the instance get_cos_sin
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
def test_compute_cos_sin_from_inv_freq_matches_instance(
    interleave, partial_rotary_factor
):
    """The cache-free static fn equals the instance cache-gather path (1-D)."""
    m = RotaryEmbedding(
        16,
        max_position_embeddings=64,
        partial_rotary_factor=partial_rotary_factor,
        interleave=interleave,
        backend="eager",
        device="cpu",
    )
    pos = torch.tensor([0, 1, 2, 3, 7, 15, 31])
    cos_i, sin_i = m.get_cos_sin(pos)
    cos_f, sin_f = compute_cos_sin_from_inv_freq(
        pos,
        m.inv_freq,
        attention_scaling=m.attention_scaling,
        interleave=interleave,
    )
    torch.testing.assert_close(cos_f, cos_i)
    torch.testing.assert_close(sin_f, sin_i)


def test_mrope_drops_cos_sin_cache_buffer():
    """mRoPE recomputes from inv_freq; it must not carry the dead seq cache."""
    mr = InterleavedMRotaryEmbedding(
        16,
        max_position_embeddings=64,
        mrope_section=[2, 3, 3],
        backend="eager",
        device="cpu",
    )
    names = {n for n, _ in mr.named_buffers()}
    assert "cos_sin_cache" not in names
    assert not hasattr(mr, "cos_sin_cache")
    assert "inv_freq" in names  # the seed it actually uses is kept


def test_compute_qwen3vl_mrope_cos_sin_from_inv_freq_matches_instance():
    """mRoPE instance get_cos_sin delegates to the free fn — byte-identical."""
    mr = InterleavedMRotaryEmbedding(
        16,
        max_position_embeddings=64,
        mrope_section=[2, 3, 3],
        backend="eager",
        device="cpu",
    )
    torch.manual_seed(0)
    pos = torch.randint(0, 50, (3, 2, 7), dtype=torch.long)
    cos_i, sin_i = mr.get_cos_sin(pos)
    cos_f, sin_f = compute_qwen3vl_mrope_cos_sin_from_inv_freq(
        pos,
        mr.inv_freq,
        mr.mrope_section,
        attention_scaling=mr.attention_scaling,
    )
    assert torch.equal(cos_f, cos_i)
    assert torch.equal(sin_f, sin_i)
