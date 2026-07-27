from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.layers.attention import AttnLayout, AttnMode
from phyai.layers.attention.gdn import (
    FlaGatedDeltaNetBackend,
    FlaGatedDeltaNetPlan,
    FlashInferGatedDeltaNetBackend,
    FlashInferGatedDeltaNetPlan,
    GatedDeltaNet,
    GatedDeltaNetCtx,
    GatedDeltaNetMetadata,
    list_backends,
)
from phyai.layers.attention.gdn.backends import fla as fla_backend
from phyai.layers.attention.gdn.backends import flashinfer as flashinfer_backend


def _has_flashinfer_gdn() -> bool:
    try:
        import flashinfer.gdn_decode  # noqa: F401
        import flashinfer.gdn_prefill  # noqa: F401

        return True
    except ImportError:
        return False


def _can_run_flashinfer_gdn() -> bool:
    if not torch.cuda.is_available() or not _has_flashinfer_gdn():
        return False
    major, _ = torch.cuda.get_device_capability()
    if major == 9:
        return True
    if major != 10 or torch.version.cuda is None:
        return False
    from flashinfer.gdn_kernels import _has_blackwell_prefill

    return int(torch.version.cuda.split(".")[0]) >= 13 and _has_blackwell_prefill


def _can_run_fla_gdn() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from fla.ops.gated_delta_rule import (  # noqa: F401
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        return True
    except ImportError:
        return False


def _gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_query_heads = q.shape[-2]
    num_value_heads = v.shape[-2]
    group_size = num_value_heads // num_query_heads
    q = q.cpu().float().repeat_interleave(group_size, dim=-2)
    k = k.cpu().float().repeat_interleave(group_size, dim=-2)
    v = v.cpu().float()
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    a_log = a_log.cpu()
    g = -a_log.exp().unsqueeze(0) * F.softplus(a.cpu().float() + dt_bias.cpu().float())
    beta = b.cpu().float().sigmoid()

    output = torch.empty_like(v)
    final_state = initial_state.cpu().float().clone()
    offsets = cu_seqlens.cpu().tolist()
    for batch_idx, (start, end) in enumerate(zip(offsets, offsets[1:])):
        state = final_state[batch_idx]
        for token_idx in range(start, end):
            state.mul_(g[token_idx].exp()[:, None, None])
            memory = torch.einsum("hvk,hk->hv", state, k[token_idx])
            delta = (v[token_idx] - memory) * beta[token_idx, :, None]
            state.add_(delta[:, :, None] * k[token_idx, :, None, :])
            output[token_idx] = torch.einsum("hvk,hk->hv", state, q[token_idx]) * scale
    return output, final_state


def test_gdn_registry_and_repr():
    layer = GatedDeltaNet(
        2,
        128,
        num_key_heads=2,
        num_value_heads=4,
    )

    assert list_backends() == ["fla", "flashinfer"]
    assert layer.num_state_heads == 4
    assert "backend='flashinfer'" in repr(layer)


def test_flashinfer_padded_prefill_builds_gates(monkeypatch):
    captured = {}
    monkeypatch.setattr(flashinfer_backend, "_check_supported_device", lambda x: None)

    def fake_prefill(q, k, v, **kwargs):
        captured.update(q=q, k=k, v=v, **kwargs)
        return torch.cat((q, q), dim=1) + 1

    monkeypatch.setattr(flashinfer_backend, "_load_prefill_op", lambda: fake_prefill)

    layer = GatedDeltaNet(2, 4, num_value_heads=4)
    q = torch.randn(2, 3, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
    a = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    b = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    a_log = torch.randn(4, dtype=torch.float32)
    dt_bias = torch.randn(4, dtype=torch.bfloat16)

    out = layer(q, k, v, a, b, a_log, dt_bias)

    assert out.shape == (2, 3, 4, 4)
    torch.testing.assert_close(out, torch.cat((q, q), dim=2) + 1)
    assert captured["q"].shape == (6, 2, 4)
    assert captured["k"].shape == (6, 2, 4)
    assert captured["v"].shape == (6, 4, 4)
    assert captured["cu_seqlens"].tolist() == [0, 3, 6]
    assert captured["cu_seqlens"].dtype == torch.int32
    expected_g = -a_log.exp().unsqueeze(0) * torch.nn.functional.softplus(
        a.reshape(-1, 4).float() + dt_bias.float()
    )
    expected_beta = b.reshape(-1, 4).float().sigmoid()
    torch.testing.assert_close(captured["g"], expected_g)
    torch.testing.assert_close(captured["beta"], expected_beta)
    assert captured["g"].dtype == torch.float32
    assert captured["beta"].dtype == torch.float32
    assert captured["use_qk_l2norm_in_kernel"] is True


def test_flashinfer_ragged_prefill_threads_state_buffers(monkeypatch):
    captured = {}
    monkeypatch.setattr(flashinfer_backend, "_check_supported_device", lambda x: None)

    def fake_prefill(q, k, v, **kwargs):
        captured.update(q=q, k=k, v=v, **kwargs)
        kwargs["output_state"].fill_(7)
        kwargs["output"].copy_(q)
        return kwargs["output"], kwargs["output_state"]

    monkeypatch.setattr(flashinfer_backend, "_load_prefill_op", lambda: fake_prefill)

    layer = GatedDeltaNet(2, 4)
    backend = FlashInferGatedDeltaNetBackend()
    plan = backend.init_forward_metadata(
        GatedDeltaNetMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=2,
            num_query_tokens=5,
            cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32),
        )
    )
    initial_state = torch.zeros(2, 2, 4, 4)
    output_state = torch.empty_like(initial_state)
    output = torch.empty(5, 2, 4)
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=plan,
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32),
        state=initial_state,
        output_state=output_state,
        output=output,
    )
    q = torch.randn(5, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(5, 2, dtype=torch.bfloat16)
    b = torch.randn(5, 2, dtype=torch.bfloat16)
    a_log = torch.randn(2, dtype=torch.float32)
    dt_bias = torch.randn(2, dtype=torch.bfloat16)

    out = layer(q, k, v, a, b, a_log, dt_bias, ctx)

    assert out.data_ptr() == output.data_ptr()
    assert captured["initial_state"] is initial_state
    assert captured["output_state"] is output_state
    assert captured["output_final_state"] is True
    assert torch.all(output_state == 7)


def test_flashinfer_decode_uses_state_pool_indices(monkeypatch):
    captured = {}
    monkeypatch.setattr(flashinfer_backend, "_check_supported_device", lambda x: None)

    def fake_decode(q, k, v, **kwargs):
        captured.update(q=q, k=k, v=v, **kwargs)
        kwargs["output"].copy_(v)
        return kwargs["output"], kwargs["initial_state"]

    monkeypatch.setattr(flashinfer_backend, "_load_decode_op", lambda: fake_decode)

    layer = GatedDeltaNet(2, 4, num_value_heads=4)
    backend = FlashInferGatedDeltaNetBackend()
    state_pool = torch.zeros(5, 4, 4, 4, dtype=torch.bfloat16)
    state_indices = torch.tensor([1, 3], dtype=torch.int32)
    output_state_indices = torch.tensor([2, 4], dtype=torch.int32)
    output = torch.empty(2, 1, 4, 4, dtype=torch.bfloat16)
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlashInferGatedDeltaNetPlan(),
        mode=AttnMode.DECODE,
        layout=AttnLayout.PADDED_4D,
        state=state_pool,
        state_indices=state_indices,
        output_state_indices=output_state_indices,
        output=output,
    )
    q = torch.randn(2, 1, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(2, 1, 4, 4, dtype=torch.bfloat16)
    a = torch.randn(2, 1, 4, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(4, dtype=torch.float32)
    dt_bias = torch.randn(4, dtype=torch.bfloat16)

    out = layer(q, k, v, a, b, a_log, dt_bias, ctx)

    assert out.data_ptr() == output.data_ptr()
    assert captured["state"] is None
    assert captured["initial_state"] is state_pool
    assert captured["initial_state_indices"] is state_indices
    assert captured["output_state_indices"] is output_state_indices
    assert captured["use_qk_l2norm"] is True
    assert captured["A_log"] is a_log
    assert captured["a"] is a
    assert captured["b"] is b
    assert captured["dt_bias"] is dt_bias


def test_fla_ragged_prefill_threads_state_and_fused_gates(monkeypatch):
    captured = {}
    monkeypatch.setattr(fla_backend, "_check_inputs", lambda *args: None)

    def fake_chunk(**kwargs):
        captured.update(kwargs)
        final_state = kwargs["initial_state"] + 7
        return kwargs["v"] + 1, final_state

    monkeypatch.setattr(fla_backend, "_load_chunk_op", lambda: fake_chunk)

    layer = GatedDeltaNet(2, 4, backend="fla")
    backend = FlaGatedDeltaNetBackend(chunk_size=32)
    initial_state = torch.zeros(2, 2, 4, 4)
    output_state = torch.empty_like(initial_state)
    output = torch.empty(5, 2, 4, dtype=torch.bfloat16)
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlaGatedDeltaNetPlan(),
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32),
        state=initial_state,
        output_state=output_state,
        output=output,
    )
    q = torch.randn(5, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(5, 2, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(2, dtype=torch.float32)
    dt_bias = torch.randn(2, dtype=torch.float32)

    out = layer(q, k, v, a, b, a_log, dt_bias, ctx)

    assert out.data_ptr() == output.data_ptr()
    assert captured["q"].shape == (1, 5, 2, 4)
    assert captured["g"].shape == (1, 5, 2)
    assert captured["beta"].shape == (1, 5, 2)
    assert captured["cu_seqlens"].dtype == torch.int64
    assert captured["initial_state"] is initial_state
    assert captured["state_v_first"] is True
    assert captured["use_gate_in_kernel"] is True
    assert captured["use_beta_sigmoid_in_kernel"] is True
    assert captured["chunk_size"] == 32
    assert torch.all(output_state == 7)
    torch.testing.assert_close(out, v + 1)


def test_fla_padded_prefill_preserves_layout(monkeypatch):
    captured = {}
    monkeypatch.setattr(fla_backend, "_check_inputs", lambda *args: None)

    def fake_chunk(**kwargs):
        captured.update(kwargs)
        return kwargs["v"] + 1, None

    monkeypatch.setattr(fla_backend, "_load_chunk_op", lambda: fake_chunk)

    layer = GatedDeltaNet(2, 4, backend="fla")
    q = torch.randn(2, 3, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(2, 3, 2, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(2, dtype=torch.float32)
    dt_bias = torch.randn(2, dtype=torch.float32)

    out = layer(q, k, v, a, b, a_log, dt_bias)

    assert captured["q"].shape == q.shape
    assert captured["cu_seqlens"] is None
    assert out.shape == v.shape
    torch.testing.assert_close(out, v + 1)


def test_fla_decode_gathers_and_scatters_state_pool(monkeypatch):
    captured = {}
    monkeypatch.setattr(fla_backend, "_check_inputs", lambda *args: None)

    def fake_recurrent(**kwargs):
        captured.update(kwargs)
        return kwargs["v"] + 1, kwargs["initial_state"] + 5

    monkeypatch.setattr(fla_backend, "_load_recurrent_op", lambda: fake_recurrent)

    layer = GatedDeltaNet(2, 4, backend="fla")
    backend = FlaGatedDeltaNetBackend()
    state_pool = torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1)
    state_pool = state_pool.expand(6, 2, 4, 4).clone()
    state_indices = torch.tensor([1, -1, 3], dtype=torch.int32)
    output_state_indices = torch.tensor([2, -1, 4], dtype=torch.int32)
    output = torch.empty(3, 1, 2, 4, dtype=torch.bfloat16)
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlaGatedDeltaNetPlan(),
        mode=AttnMode.DECODE,
        layout=AttnLayout.PADDED_4D,
        state=state_pool,
        state_indices=state_indices,
        output_state_indices=output_state_indices,
        output=output,
    )
    q = torch.randn(3, 1, 2, 4, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(3, 1, 2, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(2, dtype=torch.float32)
    dt_bias = torch.randn(2, dtype=torch.float32)

    out = layer(q, k, v, a, b, a_log, dt_bias, ctx)

    assert out.data_ptr() == output.data_ptr()
    assert captured["initial_state"][:, 0, 0, 0].tolist() == [1, 0, 3]
    assert captured["state_v_first"] is True
    assert captured["use_gate_in_kernel"] is True
    assert captured["use_beta_sigmoid_in_kernel"] is True
    assert torch.all(state_pool[2] == 6)
    assert torch.all(state_pool[4] == 8)
    assert torch.count_nonzero(out[1]) == 0


def test_gdn_idle_does_not_load_flashinfer(monkeypatch):
    def fail_load():
        raise AssertionError("FlashInfer op must not load for IDLE mode")

    monkeypatch.setattr(flashinfer_backend, "_load_prefill_op", fail_load)
    monkeypatch.setattr(flashinfer_backend, "_load_decode_op", fail_load)

    layer = GatedDeltaNet(2, 4)
    backend = FlashInferGatedDeltaNetBackend()
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlashInferGatedDeltaNetPlan(),
        mode=AttnMode.IDLE,
        layout=AttnLayout.RAGGED_3D,
    )
    q = torch.randn(3, 2, 4)
    zeros = torch.zeros(3, 2)
    a_log = torch.zeros(2, dtype=torch.float32)

    out = layer(q, q, q, zeros, zeros, a_log, torch.zeros(2), ctx)

    assert out.shape == q.shape
    assert torch.count_nonzero(out) == 0


def test_flashinfer_rejects_unsupported_device_before_loading_ops(monkeypatch):
    def fail_load():
        raise AssertionError("FlashInfer op must not load before device validation")

    monkeypatch.setattr(flashinfer_backend, "_load_prefill_op", fail_load)
    layer = GatedDeltaNet(2, 4)
    q = torch.randn(3, 2, 4, dtype=torch.bfloat16)
    gates = torch.randn(3, 2, dtype=torch.bfloat16)

    try:
        layer(
            q,
            q,
            q,
            gates,
            gates,
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.bfloat16),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )
    except RuntimeError as exc:
        assert "requires CUDA tensors" in str(exc)
    else:
        raise AssertionError("CPU input must fail before loading FlashInfer")


def test_flashinfer_validates_kernel_dtypes(monkeypatch):
    monkeypatch.setattr(flashinfer_backend, "_check_supported_device", lambda x: None)
    layer = GatedDeltaNet(2, 4)
    q = torch.randn(3, 2, 4)
    gates = torch.randn(3, 2)

    with pytest.raises(ValueError, match="q must be float16 or bfloat16"):
        layer(
            q,
            q,
            q,
            gates,
            gates,
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2),
            cu_seqlens=torch.tensor([0, 3], dtype=torch.int32),
        )


@pytest.mark.skipif(
    not _can_run_flashinfer_gdn(),
    reason="FlashInfer GDN tests require a supported SM90/SM100 CUDA device.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flashinfer_gdn_prefill_matches_recurrent_reference(dtype):
    torch.manual_seed(23)
    num_query_heads, num_value_heads, head_dim = 2, 4, 128
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(5, num_query_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(5, num_value_heads, head_dim, device="cuda", dtype=dtype)
    a = torch.randn(5, num_value_heads, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(num_value_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(num_value_heads, device="cuda", dtype=torch.float32)
    initial_state = (
        torch.randn(
            2,
            num_value_heads,
            head_dim,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    output_state = torch.empty_like(initial_state)
    layer = GatedDeltaNet(
        num_query_heads,
        head_dim,
        num_value_heads=num_value_heads,
    )
    backend = FlashInferGatedDeltaNetBackend()
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlashInferGatedDeltaNetPlan(),
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        cu_seqlens=cu_seqlens,
        state=initial_state,
        output_state=output_state,
    )

    output = layer(q, k, v, a, b, a_log, dt_bias, ctx)
    ref_output, ref_state = _gdn_reference(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        cu_seqlens,
        initial_state,
        layer.scale,
    )

    atol = 3e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(output.cpu().float(), ref_output, atol=atol, rtol=2e-2)
    torch.testing.assert_close(output_state.cpu(), ref_state, atol=atol, rtol=2e-2)


@pytest.mark.skipif(
    not _can_run_flashinfer_gdn(),
    reason="FlashInfer GDN tests require a supported SM90/SM100 CUDA device.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flashinfer_gdn_decode_matches_recurrent_reference(dtype):
    torch.manual_seed(29)
    batch_size, num_heads, head_dim = 2, 2, 128
    q = torch.randn(batch_size, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(batch_size, 1, num_heads, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32)
    state = (
        torch.randn(
            batch_size,
            num_heads,
            head_dim,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    initial_state = state.clone()
    layer = GatedDeltaNet(num_heads, head_dim)
    backend = FlashInferGatedDeltaNetBackend()
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlashInferGatedDeltaNetPlan(),
        mode=AttnMode.DECODE,
        layout=AttnLayout.PADDED_4D,
        state=state,
    )

    output = layer(q, k, v, a, b, a_log, dt_bias, ctx)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
    ref_output, ref_state = _gdn_reference(
        q.reshape(batch_size, num_heads, head_dim),
        k.reshape(batch_size, num_heads, head_dim),
        v.reshape(batch_size, num_heads, head_dim),
        a.reshape(batch_size, num_heads),
        b.reshape(batch_size, num_heads),
        a_log,
        dt_bias,
        cu_seqlens,
        initial_state,
        layer.scale,
    )

    atol = 3e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(
        output.cpu().reshape_as(ref_output).float(), ref_output, atol=atol, rtol=2e-2
    )
    torch.testing.assert_close(state.cpu(), ref_state, atol=atol, rtol=2e-2)


@pytest.mark.skipif(
    not _can_run_fla_gdn(),
    reason="FLA GDN tests require CUDA + flash-linear-attention.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fla_gdn_prefill_matches_recurrent_reference(dtype):
    torch.manual_seed(31)
    num_query_heads, num_value_heads, head_dim = 2, 4, 64
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    q = torch.randn(5, num_query_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(5, num_value_heads, head_dim, device="cuda", dtype=dtype)
    a = torch.randn(5, num_value_heads, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(num_value_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(num_value_heads, device="cuda", dtype=torch.float32)
    initial_state = (
        torch.randn(
            2,
            num_value_heads,
            head_dim,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    output_state = torch.empty_like(initial_state)
    layer = GatedDeltaNet(
        num_query_heads,
        head_dim,
        num_value_heads=num_value_heads,
        backend="fla",
    )
    backend = FlaGatedDeltaNetBackend()
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlaGatedDeltaNetPlan(),
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        cu_seqlens=cu_seqlens,
        state=initial_state,
        output_state=output_state,
    )

    output = layer(q, k, v, a, b, a_log, dt_bias, ctx)
    ref_output, ref_state = _gdn_reference(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        cu_seqlens,
        initial_state,
        layer.scale,
    )

    atol = 3e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(output.cpu().float(), ref_output, atol=atol, rtol=2e-2)
    torch.testing.assert_close(output_state.cpu(), ref_state, atol=atol, rtol=2e-2)


@pytest.mark.skipif(
    not _can_run_fla_gdn(),
    reason="FLA GDN tests require CUDA + flash-linear-attention.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fla_gdn_decode_matches_recurrent_reference(dtype):
    torch.manual_seed(37)
    batch_size, num_heads, head_dim = 2, 2, 64
    q = torch.randn(batch_size, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    a = torch.randn(batch_size, 1, num_heads, device="cuda", dtype=dtype)
    b = torch.randn_like(a)
    a_log = torch.randn(num_heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(num_heads, device="cuda", dtype=torch.float32)
    state = (
        torch.randn(
            batch_size,
            num_heads,
            head_dim,
            head_dim,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.02
    )
    initial_state = state.clone()
    layer = GatedDeltaNet(num_heads, head_dim, backend="fla")
    backend = FlaGatedDeltaNetBackend()
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlaGatedDeltaNetPlan(),
        mode=AttnMode.DECODE,
        layout=AttnLayout.PADDED_4D,
        state=state,
    )

    output = layer(q, k, v, a, b, a_log, dt_bias, ctx)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")
    ref_output, ref_state = _gdn_reference(
        q.reshape(batch_size, num_heads, head_dim),
        k.reshape(batch_size, num_heads, head_dim),
        v.reshape(batch_size, num_heads, head_dim),
        a.reshape(batch_size, num_heads),
        b.reshape(batch_size, num_heads),
        a_log,
        dt_bias,
        cu_seqlens,
        initial_state,
        layer.scale,
    )

    atol = 3e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(
        output.cpu().reshape_as(ref_output).float(), ref_output, atol=atol, rtol=2e-2
    )
    torch.testing.assert_close(state.cpu(), ref_state, atol=atol, rtol=2e-2)


def test_gdn_validates_decode_shape_and_state_parameters():
    layer = GatedDeltaNet(2, 4)
    q = torch.randn(2, 2, 2, 4)
    gates = torch.randn(2, 2, 2)
    backend = FlashInferGatedDeltaNetBackend()
    ctx = GatedDeltaNetCtx(
        backend=backend,
        plan=FlashInferGatedDeltaNetPlan(),
        mode=AttnMode.DECODE,
        layout=AttnLayout.PADDED_4D,
        state=torch.zeros(2, 2, 4, 4),
    )

    try:
        layer(
            q,
            q,
            q,
            gates,
            gates,
            torch.zeros(2, dtype=torch.float32),
            torch.zeros(2),
            ctx,
        )
    except ValueError as exc:
        assert "decode requires q shape" in str(exc)
    else:
        raise AssertionError("decode with sequence length > 1 must fail")

    try:
        layer(
            q,
            q,
            q,
            gates,
            gates,
            torch.zeros(2, dtype=torch.bfloat16),
            torch.zeros(2),
        )
    except ValueError as exc:
        assert "a_log must be float32" in str(exc)
    else:
        raise AssertionError("non-float32 a_log must fail")
