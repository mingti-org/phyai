"""TransformerBlock — five canonical configs + naming validation + HF-key mapping.

Llama / Qwen / Gemma / Mistral / Phi3 / Olmo all use HF-default norm
names (``input_layernorm`` / ``post_attention_layernorm`` /
``pre_feedforward_layernorm`` / ``post_feedforward_layernorm``), so the
block's defaults cover them with no ``norm_hf_names=`` argument. The
override dict is keyed by these HF default names, not by phyai-internal
slot identifiers — see :data:`SIGLIP_NORM_OVERRIDES` below for an
example of the override case.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

import phyai.layers.linear as L
from phyai.layers import RotaryEmbedding
from phyai.layers.transformer_block import TransformerBlock
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="phyai linear / norm / attention backends are CUDA-only",
)


# ---------------------------------------------------------------------------
# Norm-name override only needed for non-conforming families.
# ---------------------------------------------------------------------------

# SigLIP / CLIP — pre-norm with custom HF source names.
# Keys are HF defaults (= phyai default for that slot); values are the
# actual HF source names in the SigLIP checkpoint.
SIGLIP_NORM_OVERRIDES = {
    "input_layernorm": "layer_norm1",
    "post_attention_layernorm": "layer_norm2",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_mesh(
    *, sizes: dict[str, int] | None = None, ranks: dict[str, int] | None = None
) -> Mesh:
    sizes = sizes or {"tp": 1}
    ranks = ranks or {}
    tm = MagicMock()
    tm.mesh_dim_names = tuple(sizes.keys())
    _names = tm.mesh_dim_names
    tm.size.side_effect = lambda axis: sizes.get(
        axis if isinstance(axis, str) else _names[axis], 1
    )
    tm.get_local_rank.side_effect = lambda axis: ranks.get(axis, 0)
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    mesh = Mesh(tm, name="model")
    register_mesh(mesh)
    return mesh


@pytest.fixture
def fake_mesh():
    saved = dict(_meshes)
    try:
        yield _fake_mesh
    finally:
        _meshes.clear()
        _meshes.update(saved)
        L._reset_for_test()


def _init_dispatcher():
    return L.init(register_flashinfer=False, validate=False)


def _base_kwargs(**overrides) -> dict:
    """Common construction kwargs. Defaults match Llama/Qwen/Gemma."""
    base: dict = {}
    base.update(overrides)
    return base
    return base


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_unknown_norm_type_raises(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    with pytest.raises(ValueError, match="Unknown norm_type"):
        TransformerBlock(
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            **_base_kwargs(norm_type="banana"),
        )


def test_hidden_size_not_divisible_by_heads_raises(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    with pytest.raises(ValueError, match="not divisible"):
        TransformerBlock(
            hidden_size=65,
            num_heads=4,
            intermediate_size=128,
            **_base_kwargs(),
        )


def test_default_pre_norm_uses_hf_defaults(fake_mesh):
    """No norm_hf_names passed -> block uses HF defaults."""
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        prefix="model.layers.0",
    )
    keys = {
        hf_key
        for _, p in blk.named_parameters()
        for hf_key, _sid in getattr(p, "hf_keys", ())
    }
    assert "model.layers.0.input_layernorm.weight" in keys
    assert "model.layers.0.post_attention_layernorm.weight" in keys


def test_default_sandwich_norm_uses_hf_defaults(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        sandwich_norm=True,
        norm_type="gemma_rmsnorm",
        prefix="model.layers.0",
    )
    keys = {
        hf_key
        for _, p in blk.named_parameters()
        for hf_key, _sid in getattr(p, "hf_keys", ())
    }
    assert "model.layers.0.input_layernorm.weight" in keys
    assert "model.layers.0.post_attention_layernorm.weight" in keys
    assert "model.layers.0.pre_feedforward_layernorm.weight" in keys
    assert "model.layers.0.post_feedforward_layernorm.weight" in keys


def test_norm_hf_names_unknown_key_raises(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    # Old phyai-internal slot names ("input_norm" etc.) are no longer
    # accepted — the new API expects HF default names as keys.
    with pytest.raises(ValueError, match="unknown keys"):
        TransformerBlock(
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            norm_hf_names={"input_norm": "x"},
        )


def test_norm_hf_names_pre_norm_rejects_sandwich_only_keys(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    # pre_feedforward_layernorm only exists in the sandwich-norm topology.
    with pytest.raises(ValueError, match="unknown keys"):
        TransformerBlock(
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            norm_hf_names={"pre_feedforward_layernorm": "anything"},
        )


def test_explicit_head_dim_takes_priority(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        head_dim=32,
        **_base_kwargs(),
    )
    assert blk.head_dim == 32
    assert blk.q_heads_local == 4


def test_pre_norm_has_two_norms_only(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        sandwich_norm=False,
        **_base_kwargs(),
    )
    assert blk.input_norm is not None
    assert blk.pre_ff_norm is not None
    assert isinstance(blk.post_attn_norm, torch.nn.Identity)
    assert isinstance(blk.post_ff_norm, torch.nn.Identity)


def test_sandwich_norm_has_four_norms(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        sandwich_norm=True,
    )
    assert blk.input_norm is not None
    assert blk.post_attn_norm is not None
    assert blk.pre_ff_norm is not None
    assert blk.post_ff_norm is not None


def test_qk_norm_default_off(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        **_base_kwargs(),
    )
    assert isinstance(blk.q_norm, torch.nn.Identity)
    assert isinstance(blk.k_norm, torch.nn.Identity)


def test_attention_kind_accepts_layer_idx_metadata(fake_mesh):
    """No-cache attention may carry layer_idx (stack-position metadata, ignored).

    The paged kinds (ar / diffusion) still *require* it for KV-pool addressing,
    but attention treats it as optional inert metadata (surfaced in repr).
    """
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        layer_idx=3,  # previously rejected for attn_kind="attention"
        **_base_kwargs(),
    )
    assert blk.layer_idx == 3
    assert "layer_idx=3" in repr(blk)
    # None remains valid for attention.
    assert (
        TransformerBlock(hidden_size=64, num_heads=4, intermediate_size=128).layer_idx
        is None
    )
    # ar still requires it (load-bearing — addresses the shared KV pool).
    with pytest.raises(ValueError, match="requires layer_idx"):
        TransformerBlock(
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            attn_kind="ar",
        )


def test_qk_norm_present_when_enabled(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        attn_qk_norm=True,
        **_base_kwargs(),
    )
    assert blk.q_norm is not None
    assert blk.k_norm is not None
    # Q/K norm operates on head_dim, not hidden_size.
    assert blk.q_norm.weight.shape == (16,)
    assert blk.k_norm.weight.shape == (16,)


def test_rope_required_when_positions_missing(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    rope = RotaryEmbedding(16, max_position_embeddings=64, backend="eager")
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        head_dim=16,
        rope=rope,
        attn_backend="eager",
        norm_backend="phyai-kernel",
        **_base_kwargs(),
    )
    x = torch.randn(2, 8, 64)
    with pytest.raises(ValueError, match="positions"):
        blk(x)


def test_wrong_input_rank_raises(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        head_dim=16,
        attn_backend="eager",
        norm_backend="phyai-kernel",
        **_base_kwargs(),
    )
    x = torch.randn(2, 4, 8, 64)
    with pytest.raises(ValueError, match="2-D .* or 3-D"):
        blk(x)


def test_wrong_hidden_size_raises(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        head_dim=16,
        attn_backend="eager",
        norm_backend="phyai-kernel",
        **_base_kwargs(),
    )
    x = torch.randn(2, 8, 32)
    with pytest.raises(ValueError, match="hidden_size"):
        blk(x)


# ---------------------------------------------------------------------------
# Forward smoke — five canonical families
# ---------------------------------------------------------------------------


@cuda_only
def test_gemma1_style_forward(fake_mesh):
    """Gemma1 / Llama: pre-norm + RMSNorm + RoPE + gated SiLU + GQA + causal."""
    fake_mesh()
    _init_dispatcher()
    H, num_heads, kv_heads, head_dim, I = 64, 4, 2, 16, 128

    rope = RotaryEmbedding(head_dim, max_position_embeddings=128, backend="eager")
    blk = TransformerBlock(
        hidden_size=H,
        num_heads=num_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=I,
        attn_causal=True,
        rope=rope,
        mlp_gated=True,
        mlp_activation="silu",
        norm_type="rmsnorm",
        attn_backend="sdpa",
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
    ).cuda()
    rope.cuda()

    x = (torch.randn(2, 16, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.arange(16, device="cuda")
    y = blk(x, positions=pos)
    assert y.shape == (2, 16, H)
    assert y.dtype == torch.bfloat16


@cuda_only
def test_precompute_rope_pattern_b_matches_pattern_a(fake_mesh):
    """Pattern B (threaded ``cos`` / ``sin``) == Pattern A (per-layer rope).

    Covers the block-level ``cos`` / ``sin`` forward kwargs and the
    ``precompute_rope=True`` branch against the default ``positions`` path,
    using one shared rope instance and identical weights.
    """
    fake_mesh()
    _init_dispatcher()
    H, num_heads, kv_heads, head_dim, I = 64, 4, 2, 16, 128

    rope = RotaryEmbedding(
        head_dim, max_position_embeddings=128, backend="eager"
    ).cuda()

    def _build(precompute):
        return TransformerBlock(
            hidden_size=H,
            num_heads=num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=I,
            attn_causal=True,
            rope=rope,
            precompute_rope=precompute,
            mlp_gated=True,
            mlp_activation="silu",
            norm_type="rmsnorm",
            attn_backend="sdpa",
            norm_backend="phyai-kernel",
            params_dtype=torch.bfloat16,
        ).cuda()

    blk_a = _build(False)  # Pattern A: forward(positions=...)
    blk_b = _build(True)  # Pattern B: forward(cos=..., sin=...)
    # phyai layers init weights with uninitialized memory (filled by
    # load_pretrained in production); give blk_a finite values, then copy
    # them into blk_b so the two paths differ only in rope delivery.
    torch.manual_seed(0)
    for p in blk_a.parameters():
        torch.nn.init.normal_(p, std=0.02)
    blk_b.load_state_dict(blk_a.state_dict())

    x = (torch.randn(2, 16, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.arange(16, device="cuda")
    cos, sin = rope.get_cos_sin(pos)

    y_a = blk_a(x, positions=pos)
    y_b = blk_b(x, cos=cos, sin=sin)
    torch.testing.assert_close(y_b, y_a)


@cuda_only
def test_gemma2_style_sandwich_forward(fake_mesh):
    """Gemma2: sandwich + GemmaRMSNorm + soft_cap + sliding_window + GeGLU."""
    fake_mesh()
    _init_dispatcher()
    H, num_heads, kv_heads, head_dim, I = 64, 4, 2, 16, 128

    rope = RotaryEmbedding(head_dim, max_position_embeddings=128, backend="eager")
    blk = TransformerBlock(
        hidden_size=H,
        num_heads=num_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=I,
        sandwich_norm=True,
        attn_causal=True,
        attn_sliding_window=8,
        attn_logits_soft_cap=50.0,
        rope=rope,
        mlp_gated=True,
        mlp_activation="gelu_tanh",
        norm_type="gemma_rmsnorm",
        attn_backend="eager",
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
    ).cuda()
    rope.cuda()

    x = (torch.randn(1, 12, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.arange(12, device="cuda")
    y = blk(x, positions=pos)
    assert y.shape == (1, 12, H)


@cuda_only
def test_gemma3_style_sandwich_qk_norm_forward(fake_mesh):
    """Gemma3: sandwich + GemmaRMSNorm + sliding_window + Q/K head_dim norm + no soft_cap."""
    fake_mesh()
    _init_dispatcher()
    H, num_heads, kv_heads, head_dim, I = 64, 4, 2, 16, 128

    rope = RotaryEmbedding(head_dim, max_position_embeddings=128, backend="eager")
    blk = TransformerBlock(
        hidden_size=H,
        num_heads=num_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=I,
        sandwich_norm=True,
        attn_causal=True,
        attn_sliding_window=8,
        attn_qk_norm=True,
        rope=rope,
        mlp_gated=True,
        mlp_activation="gelu_tanh",
        norm_type="gemma_rmsnorm",
        attn_backend="eager",
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
    ).cuda()
    rope.cuda()

    x = (torch.randn(1, 12, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.arange(12, device="cuda")
    y = blk(x, positions=pos)
    assert y.shape == (1, 12, H)


@cuda_only
def test_qwen2_style_forward(fake_mesh):
    """Qwen2 / Qwen2.5: pre-norm + RMSNorm + RoPE + gated SiLU + GQA + Q/K/V bias."""
    fake_mesh()
    _init_dispatcher()
    H, num_heads, kv_heads, head_dim, I = 64, 4, 2, 16, 128

    rope = RotaryEmbedding(head_dim, max_position_embeddings=128, backend="eager")
    blk = TransformerBlock(
        hidden_size=H,
        num_heads=num_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=I,
        attn_causal=True,
        attn_bias=True,  # Qwen2 has Q/K/V bias
        attn_out_bias=False,  # but no O bias
        rope=rope,
        mlp_gated=True,
        mlp_activation="silu",
        norm_type="rmsnorm",
        attn_backend="sdpa",
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
    ).cuda()
    rope.cuda()

    # Verify bias is allocated where expected.
    assert blk.qkv_proj.bias is not None
    assert blk.o_proj.bias is None

    x = (torch.randn(2, 16, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.arange(16, device="cuda")
    y = blk(x, positions=pos)
    assert y.shape == (2, 16, H)


@cuda_only
def test_qwen3_style_forward(fake_mesh):
    """Qwen3: pre-norm + RMSNorm + RoPE + gated SiLU + GQA + Q/K head_dim norm + no QKV bias."""
    fake_mesh()
    _init_dispatcher()
    H, num_heads, kv_heads, head_dim, I = 64, 4, 2, 16, 128

    rope = RotaryEmbedding(head_dim, max_position_embeddings=128, backend="eager")
    blk = TransformerBlock(
        hidden_size=H,
        num_heads=num_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=I,
        attn_causal=True,
        attn_qk_norm=True,  # Qwen3 adds q_norm/k_norm
        rope=rope,
        mlp_gated=True,
        mlp_activation="silu",
        norm_type="rmsnorm",
        attn_backend="sdpa",
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
    ).cuda()
    rope.cuda()

    # Q/K/V/O all unbiased in Qwen3.
    assert blk.qkv_proj.bias is None
    assert blk.o_proj.bias is None
    assert blk.q_norm is not None
    assert blk.k_norm is not None

    x = (torch.randn(2, 16, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.arange(16, device="cuda")
    y = blk(x, positions=pos)
    assert y.shape == (2, 16, H)


@cuda_only
def test_siglip_style_forward(fake_mesh):
    """SigLIP encoder: pre-norm + LayerNorm(bias) + plain GELU-tanh MLP + non-causal + out_proj."""
    fake_mesh()
    _init_dispatcher()
    H, num_heads, head_dim, I = 96, 4, 24, 256

    blk = TransformerBlock(
        hidden_size=H,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=I,
        attn_causal=False,
        attn_bias=True,
        rope=None,
        mlp_gated=False,
        mlp_activation="gelu_tanh",
        mlp_bias=True,
        norm_type="layernorm",
        norm_eps=1e-6,
        norm_bias=True,
        attn_backend="sdpa",
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
        norm_hf_names=SIGLIP_NORM_OVERRIDES,
        attn_out_hf_name="out_proj",
    ).cuda()

    x = (torch.randn(2, 32, H) * 0.05).to(torch.bfloat16).cuda()
    y = blk(x)
    assert y.shape == (2, 32, H)


@cuda_only
def test_ragged_forward(fake_mesh):
    """2-D ragged input runs through the block and preserves shape."""
    fake_mesh()
    _init_dispatcher()
    H, num_heads, head_dim, I = 64, 4, 16, 128

    rope = RotaryEmbedding(head_dim, max_position_embeddings=64, backend="eager")
    blk = TransformerBlock(
        hidden_size=H,
        num_heads=num_heads,
        head_dim=head_dim,
        intermediate_size=I,
        rope=rope,
        attn_backend="eager",
        norm_backend="phyai-kernel",
        params_dtype=torch.bfloat16,
        **_base_kwargs(),
    ).cuda()
    rope.cuda()

    nnz = 24
    x = (torch.randn(nnz, H) * 0.05).to(torch.bfloat16).cuda()
    pos = torch.cat([torch.arange(12), torch.arange(12)]).to("cuda")
    cu = torch.tensor([0, 12, 24], dtype=torch.int32, device="cuda")
    y = blk(x, positions=pos, cu_seqlens_q=cu)
    assert y.shape == (nnz, H)


# ---------------------------------------------------------------------------
# Param-attached HF-key mapping — exact keys per family
# ---------------------------------------------------------------------------


def _hf_keys(blk: TransformerBlock) -> set[str]:
    """Collect every HF source key declared by any parameter in `blk`."""
    keys: set[str] = set()
    for _, p in blk.named_parameters():
        for hf_key, _shard_id in getattr(p, "hf_keys", ()):
            keys.add(hf_key)
    return keys


def test_pre_norm_hf_keys_llama_like(fake_mesh):
    """Llama / Gemma1 / Qwen2 / Mistral convention."""
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        sandwich_norm=False,
        attn_bias=False,
        mlp_bias=False,
        mlp_gated=True,
        norm_type="rmsnorm",
        norm_backend="phyai-kernel",
        prefix="model.layers.0",
    )
    expected = {
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    }
    assert _hf_keys(blk) == expected


def test_qwen2_hf_keys_with_qkv_bias(fake_mesh):
    """Qwen2: Q/K/V bias should appear as separate keys, O has no bias."""
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        attn_bias=True,
        attn_out_bias=False,
        norm_type="rmsnorm",
        norm_backend="phyai-kernel",
        prefix="model.layers.7",
    )
    keys = _hf_keys(blk)
    assert "model.layers.7.self_attn.q_proj.bias" in keys
    assert "model.layers.7.self_attn.k_proj.bias" in keys
    assert "model.layers.7.self_attn.v_proj.bias" in keys
    assert "model.layers.7.self_attn.o_proj.bias" not in keys


def test_qwen3_hf_keys_qk_norm(fake_mesh):
    """Qwen3: pre-norm + q_norm / k_norm HF keys."""
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        attn_qk_norm=True,
        norm_type="rmsnorm",
        norm_backend="phyai-kernel",
        prefix="model.layers.3",
    )
    keys = _hf_keys(blk)
    assert "model.layers.3.self_attn.q_norm.weight" in keys
    assert "model.layers.3.self_attn.k_norm.weight" in keys


def test_gemma2_hf_keys_sandwich(fake_mesh):
    """Gemma2: 4 sandwich norms, no q_norm / k_norm."""
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        sandwich_norm=True,
        norm_type="gemma_rmsnorm",
        norm_backend="phyai-kernel",
        prefix="model.layers.5",
    )
    keys = _hf_keys(blk)
    assert "model.layers.5.input_layernorm.weight" in keys
    assert "model.layers.5.post_attention_layernorm.weight" in keys
    assert "model.layers.5.pre_feedforward_layernorm.weight" in keys
    assert "model.layers.5.post_feedforward_layernorm.weight" in keys
    assert "model.layers.5.self_attn.q_norm.weight" not in keys


def test_gemma3_hf_keys_sandwich_qk_norm(fake_mesh):
    """Gemma3: sandwich + q_norm / k_norm."""
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        sandwich_norm=True,
        attn_qk_norm=True,
        norm_type="gemma_rmsnorm",
        norm_backend="phyai-kernel",
        prefix="model.layers.5",
    )
    keys = _hf_keys(blk)
    assert "model.layers.5.input_layernorm.weight" in keys
    assert "model.layers.5.post_attention_layernorm.weight" in keys
    assert "model.layers.5.pre_feedforward_layernorm.weight" in keys
    assert "model.layers.5.post_feedforward_layernorm.weight" in keys
    assert "model.layers.5.self_attn.q_norm.weight" in keys
    assert "model.layers.5.self_attn.k_norm.weight" in keys


def test_siglip_hf_keys(fake_mesh):
    """SigLIP: layer_norm{1,2} + out_proj + fc1/fc2 + bias on q/k/v/o/fc/norm."""
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=96,
        num_heads=4,
        head_dim=24,
        intermediate_size=256,
        attn_causal=False,
        attn_bias=True,
        rope=None,
        mlp_gated=False,
        mlp_activation="gelu_tanh",
        mlp_bias=True,
        norm_type="layernorm",
        norm_bias=True,
        norm_backend="phyai-kernel",
        prefix="vision_model.encoder.layers.0",
        norm_hf_names=SIGLIP_NORM_OVERRIDES,
        attn_out_hf_name="out_proj",
    )
    keys = _hf_keys(blk)
    expected = {
        "vision_model.encoder.layers.0.layer_norm1.weight",
        "vision_model.encoder.layers.0.layer_norm1.bias",
        "vision_model.encoder.layers.0.layer_norm2.weight",
        "vision_model.encoder.layers.0.layer_norm2.bias",
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_model.encoder.layers.0.self_attn.q_proj.bias",
        "vision_model.encoder.layers.0.self_attn.k_proj.weight",
        "vision_model.encoder.layers.0.self_attn.k_proj.bias",
        "vision_model.encoder.layers.0.self_attn.v_proj.weight",
        "vision_model.encoder.layers.0.self_attn.v_proj.bias",
        "vision_model.encoder.layers.0.self_attn.out_proj.weight",
        "vision_model.encoder.layers.0.self_attn.out_proj.bias",
        "vision_model.encoder.layers.0.mlp.fc1.weight",
        "vision_model.encoder.layers.0.mlp.fc1.bias",
        "vision_model.encoder.layers.0.mlp.fc2.weight",
        "vision_model.encoder.layers.0.mlp.fc2.bias",
    }
    assert keys == expected


def test_extra_repr_contains_key_fields(fake_mesh):
    fake_mesh()
    _init_dispatcher()
    blk = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        head_dim=16,
        intermediate_size=128,
        attn_qk_norm=True,
        norm_type="rmsnorm",
        norm_backend="phyai-kernel",
        **_base_kwargs(),
    )
    s = repr(blk)
    assert "hidden_size=64" in s
    assert "norm_type='rmsnorm'" in s
    assert "sandwich_norm=False" in s
    assert "attn_qk_norm=True" in s
