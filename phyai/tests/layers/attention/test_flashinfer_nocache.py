"""CUDA checks for explicitly selected FlashInfer prefill kernels."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from phyai.kernel.bootstrap import initialize_kernel_system
from phyai.kernel.config import KernelConfig
from phyai.layers.attention import Attention, AttnLayout
from phyai.runtime.cuda_graph_manager import CudaGraph


def pin_backend(name):
    initialize_kernel_system(
        KernelConfig(
            policy_config={
                "schema": "phyai.kernel/v1",
                "overrides": [
                    {
                        "match": {"op": "attention"},
                        "use": f"flashinfer.attention.{name}",
                    }
                ],
            }
        )
    )


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fa2_policy_reaches_cuda_kernel(monkeypatch, batch, ragged, causal, dtype):
    import flashinfer.prefill as prefill

    pin_backend("fa2")
    single = Mock(wraps=prefill.single_prefill_with_kv_cache)
    wrapper = Mock(wraps=prefill.BatchPrefillWithRaggedKVCacheWrapper)
    monkeypatch.setattr(prefill, "single_prefill_with_kv_cache", single)
    monkeypatch.setattr(prefill, "BatchPrefillWithRaggedKVCacheWrapper", wrapper)
    q = torch.randn(batch, 17, 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(batch, 29, 2, 64, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    layer = Attention(4, 64, num_kv_heads=2, causal=causal, backend="sdpa")
    if ragged:
        actual = layer(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            cu_seqlens_q=torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 17,
            cu_seqlens_kv=torch.arange(batch + 1, device="cuda", dtype=torch.int32)
            * 29,
        ).reshape_as(q)
    else:
        actual = layer(q, k, v)
    mask = None
    if causal:
        mask = (
            torch.arange(29, device="cuda")[None, :]
            <= torch.arange(17, device="cuda")[:, None] + 12
        )
    with sdpa_kernel(SDPBackend.MATH):
        expected = (
            F.scaled_dot_product_attention(
                q.float().transpose(1, 2),
                k.float().transpose(1, 2),
                v.float().transpose(1, 2),
                attn_mask=mask,
                enable_gqa=True,
            )
            .transpose(1, 2)
            .to(dtype)
        )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    assert torch.isfinite(actual).all()
    assert {choice.kernel_id for choice in layer._call_site._memo.values()} == {
        "flashinfer.attention.fa2"
    }
    if batch == 1:
        assert single.call_args.kwargs["backend"] == "fa2"
        wrapper.assert_not_called()
    else:
        assert wrapper.call_args.kwargs["backend"] == "fa2"
        single.assert_not_called()


@pytest.mark.parametrize("batch", [1, 2])
def test_fa2_cuda_graph_replay(batch):
    pin_backend("fa2")
    layer = Attention(2, 64, causal=False)
    inputs = {
        "q": torch.randn(batch, 17, 2, 64, device="cuda", dtype=torch.bfloat16),
        "k": torch.randn(batch, 29, 2, 64, device="cuda", dtype=torch.bfloat16),
        "v": torch.randn(batch, 29, 2, 64, device="cuda", dtype=torch.bfloat16),
    }
    graph = CudaGraph()
    ctx = layer._build_default_ctx(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        AttnLayout.PADDED_4D,
        None,
        None,
        mode="capture",
    )

    def forward(q, k, v):
        return layer(q, k, v, ctx=ctx)

    with torch.inference_mode():
        graph.capture(forward, inputs)
        inputs["q"].normal_()
        inputs["k"].normal_()
        actual = graph.replay(inputs).clone()
        expected = layer(**inputs)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    assert all(
        backend._prefill_backend == "fa2"
        for backend in layer._resolved_backends.values()
    )
    graph.reset()


def test_wrapper_only_backend_keeps_single_sequence_on_wrapper(monkeypatch):
    import flashinfer.prefill as prefill

    pin_backend("cudnn")
    single = Mock(side_effect=AssertionError("cuDNN has no single-prefill entry point"))
    wrapper = Mock()
    wrapper.run.side_effect = lambda q, k, v: q.clone()
    factory = Mock(return_value=wrapper)
    monkeypatch.setattr(prefill, "single_prefill_with_kv_cache", single)
    monkeypatch.setattr(prefill, "BatchPrefillWithRaggedKVCacheWrapper", factory)
    q = torch.randn(1, 5, 2, 64, device="cuda", dtype=torch.bfloat16)
    layer = Attention(2, 64, causal=False)
    layer(q, q, q)
    assert factory.call_args.kwargs["backend"] == "cudnn"
    wrapper.plan.assert_called_once()
    wrapper.run.assert_called_once()
    single.assert_not_called()
