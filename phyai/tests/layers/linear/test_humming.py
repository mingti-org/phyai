"""Humming PTQ loading, kernel selection, and quantized Linear execution."""

from __future__ import annotations

import builtins
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import phyai.layers.linear as L
from phyai.kernel.facts import facts_from_query
from phyai.kernel.ops.gemm import register
from phyai.kernel.registry import Catalog
from phyai.kernel.types import KernelQuery
from phyai.layers.quant import (
    AllocationRequest,
    Granularity,
    QDType,
    QuantScheme,
    TensorQuant,
)
from phyai.layers.quant.humming import HummingSpec
from phyai.weights.loader import WeightLoadSession


def _scheme(weight=QDType.INT4, activation=None):
    block_k = {QDType.MXFP4: 32, QDType.NVFP4: 16}.get(weight, 128)
    return QuantScheme(
        weight=TensorQuant(
            weight,
            Granularity.BLOCK,
            block_shape=(1, block_k),
            micro_scaled=weight in {QDType.MXFP4, QDType.NVFP4},
        ),
        input=(
            None
            if activation is None
            else TensorQuant(activation, Granularity.PER_CHANNEL, dynamic=True)
        ),
        online=True,
    )


def _weight(n, k, offset=0):
    return ((torch.arange(n * k).reshape(n, k) + offset) % 127).to(torch.bfloat16)


def _block_fp8_scheme(weight=QDType.FP8_E4M3, activation=None):
    scheme = _scheme(weight, activation)
    return replace(scheme, weight=replace(scheme.weight, block_shape=(128, 128)))


@pytest.fixture
def prepared_weights(monkeypatch):
    calls = []

    def prepare(self, layer, weight):
        calls.append(weight.clone())
        n, k = weight.shape
        return SimpleNamespace(shape_n=n, shape_k=k), {
            "weight": torch.zeros(n, k // 8, dtype=torch.int32),
            "weight_scale": torch.ones(n, k // 128, dtype=torch.bfloat16),
        }

    monkeypatch.setattr(HummingSpec, "_prepare_weights", prepare)
    return calls


@pytest.mark.parametrize("kind", ["replicated", "column", "row"])
def test_ptq_receives_local_tp_weights(fake_mesh, prepared_weights, kind):
    fake_mesh(tp_size=4, rank=2)
    kwargs = dict(
        spec=HummingSpec(_scheme()),
        params_dtype=torch.bfloat16,
        device="cpu",
        prefix="projection",
        bias=False,
    )
    if kind == "replicated":
        layer = L.ReplicatedLinear(128, 64, **kwargs)
        weight = _weight(64, 128)
        expected = weight
    elif kind == "column":
        layer = L.ColumnParallelLinear(128, 256, **kwargs)
        weight = _weight(256, 128)
        expected = weight[128:192]
    else:
        layer = L.RowParallelLinear(512, 64, **kwargs)
        weight = _weight(64, 512)
        expected = weight[:, 256:384]

    session = WeightLoadSession(layer)
    session.load({"projection.weight": weight})
    assert not session.report.casts
    assert prepared_weights == []
    session.finish()

    assert len(prepared_weights) == 1
    torch.testing.assert_close(prepared_weights[0], expected)
    assert layer.logical_shape == tuple(expected.shape)
    assert layer.weight.dtype == torch.int32
    assert getattr(layer, "_humming_pending_weight", None) is None


def test_ptq_rejects_local_checkpoint_shape_before_tp_broadcast(
    fake_mesh, prepared_weights
):
    fake_mesh(tp_size=64, rank=0)
    layer = L.ColumnParallelLinear(
        128,
        4096,
        spec=HummingSpec(_scheme()),
        params_dtype=torch.bfloat16,
        device="cpu",
        prefix="projection",
        bias=False,
    )
    assert layer.logical_shape == (64, 128)
    session = WeightLoadSession(layer)
    with pytest.raises(ValueError, match="[Ss]hape"):
        session.load({"projection.weight": _weight(64, 128)})
    assert layer._humming_loaded_shards == set()
    assert layer._humming_pending_weight is None
    assert prepared_weights == []

    weight = _weight(4096, 128)
    session.load({"projection.weight": weight})
    session.finish()
    torch.testing.assert_close(prepared_weights[0], weight[:64])


def test_ptq_fuses_complete_gate_up_in_any_load_order(fake_mesh, prepared_weights):
    fake_mesh(tp_size=4, rank=2)
    layer = L.MergedColumnParallelLinear(
        128,
        [256, 512],
        spec=HummingSpec(_scheme()),
        params_dtype=torch.bfloat16,
        device="cpu",
        prefix="mlp.gate_up_proj",
        bias=False,
    )
    gate, up = _weight(256, 128), _weight(512, 128, 17)
    session = WeightLoadSession(layer)
    session.load({"mlp.up_proj.weight": up})
    session.load({"mlp.gate_proj.weight": gate})
    session.finish()

    torch.testing.assert_close(
        prepared_weights[0], torch.cat([gate[128:192], up[256:384]])
    )
    assert layer.logical_widths == [64, 128]


def test_ptq_qkv_uses_gqa_replication_before_quantizing(fake_mesh, prepared_weights):
    fake_mesh(tp_size=4, rank=3)
    layer = L.QKVParallelLinear(
        128,
        64,
        8,
        num_kv_heads=2,
        group="dense_tp",
        spec=HummingSpec(_scheme()),
        params_dtype=torch.bfloat16,
        device="cpu",
        prefix="attn.qkv_proj",
        bias=False,
    )
    q, k, v = _weight(512, 128), _weight(128, 128, 19), _weight(128, 128, 31)
    session = WeightLoadSession(layer)
    session.load({"attn.v_proj.weight": v, "attn.q_proj.weight": q})
    session.load({"attn.k_proj.weight": k})
    session.finish()
    torch.testing.assert_close(
        prepared_weights[0], torch.cat([q[384:512], k[64:128], v[64:128]])
    )


@pytest.mark.parametrize("require_all", [False, True])
def test_non_strict_loading_cannot_quantize_missing_fused_legs(
    fake_mesh, prepared_weights, require_all
):
    fake_mesh()
    layer = L.MergedColumnParallelLinear(
        128,
        [64, 64],
        spec=HummingSpec(_scheme()),
        params_dtype=torch.bfloat16,
        device="cpu",
        prefix="mlp.gate_up_proj",
        bias=False,
    )
    session = WeightLoadSession(layer)
    session.load({"mlp.gate_proj.weight": _weight(64, 128)})
    with pytest.raises(RuntimeError, match="[Mm]issing|[Ii]ncomplete|[Cc]omplete"):
        session.finish(strict=False, require_all=require_all)
    assert prepared_weights == []
    assert not layer._humming_prepared


def test_post_load_is_idempotent_and_preserves_loader_metadata(
    fake_mesh, prepared_weights
):
    fake_mesh()
    layer = L.ReplicatedLinear(
        128,
        64,
        spec=HummingSpec(_scheme()),
        params_dtype=torch.bfloat16,
        device="cpu",
        prefix="projection",
        bias=False,
    )
    old_keys = layer.weight.hf_keys
    old_loader = layer.weight.weight_loader
    session = WeightLoadSession(layer)
    session.load({"projection.weight": _weight(64, 128)})
    session.finish()
    packed = layer.weight
    layer.post_load()
    assert layer.weight is packed
    assert len(prepared_weights) == 1
    assert layer.weight.hf_keys == old_keys
    assert layer.weight.weight_loader is old_loader
    assert not layer.weight.requires_grad
    assert not layer.weight_scale.requires_grad

    with pytest.raises(RuntimeError, match="reload"):
        WeightLoadSession(layer).load({"projection.weight": _weight(64, 128)})
    assert layer.weight is packed


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((64, 96), torch.bfloat16),
        ((64, 128), torch.float32),
        ((4, 64, 128), torch.bfloat16),
    ],
)
def test_ptq_rejects_invalid_local_weight_configuration(shape, dtype):
    with pytest.raises((ValueError, TypeError)):
        HummingSpec(_scheme()).allocate(
            nn.Module(),
            AllocationRequest(
                weight_shape=shape,
                logical_widths=[shape[0]],
                params_dtype=dtype,
                device="cpu",
            ),
        )


def test_mxfp4_rejects_fp16_compute_for_pinned_humming_release():
    with pytest.raises(ValueError, match="MXFP4.*BF16"):
        HummingSpec(_scheme(QDType.MXFP4)).allocate(
            nn.Module(),
            AllocationRequest(
                weight_shape=(128, 128),
                logical_widths=[128],
                params_dtype=torch.float16,
                device="cpu",
            ),
        )


def _eligible(scheme, *, arch="sm90", available=True, n=128, k=128, dtype="bf16"):
    catalog = Catalog()
    register(catalog)
    query = KernelQuery.build(
        "gemm",
        device=f"nvidia:{arch}",
        dtype={"input": dtype, "output": dtype, "weight": "int32"},
        quant=HummingSpec(scheme).physical_signature,
        shape={"M": 6, "N": n, "K": k},
    )
    facts = facts_from_query(
        query,
        catalog.op("gemm"),
        libraries={"lib.humming": available, "lib.flashinfer": True},
    )
    return {
        row.kernel_id for row in catalog.impls("gemm") if row.when.eval(facts) is None
    }


@pytest.mark.parametrize(
    "scheme,kernel",
    [
        (_scheme(), "humming.gemm.w4a16"),
        (_scheme(QDType.INT8), "humming.gemm.w8a16"),
        (_scheme(QDType.FP8_E4M3), "humming.gemm.fp8_a16"),
        (_scheme(QDType.FP8_E5M2), "humming.gemm.fp8_a16"),
        (_scheme(QDType.MXFP4), "humming.gemm.mxfp4_a16"),
        (_scheme(QDType.NVFP4), "humming.gemm.nvfp4_a16"),
        (_scheme(QDType.FP8_E4M3, QDType.FP8_E4M3), "humming.gemm.fp8_a8"),
        (_scheme(QDType.FP8_E5M2, QDType.FP8_E5M2), "humming.gemm.fp8_a8"),
        (_scheme(QDType.INT8, QDType.INT8), "humming.gemm.int8_a8"),
        (_scheme(QDType.INT4, QDType.FP8_E4M3), "humming.gemm.int4_a8"),
        (_scheme(QDType.INT4, QDType.INT8), "humming.gemm.int4_a8"),
        (_block_fp8_scheme(), "humming.gemm.fp8_a16"),
        (_block_fp8_scheme(activation=QDType.FP8_E4M3), "humming.gemm.fp8_a8"),
    ],
)
def test_packed_weight_only_selects_its_humming_kernel(scheme, kernel):
    assert _eligible(scheme) == {kernel}
    assert _eligible(scheme, available=False) == set()


def test_humming_eligibility_checks_compute_architecture_and_alignment():
    assert _eligible(_scheme(), arch="sm75") == set()
    assert _eligible(_scheme(), arch="sm75", dtype="fp16") == {"humming.gemm.w4a16"}
    assert _eligible(_scheme(), n=96) == set()
    assert _eligible(_scheme(), k=112) == set()
    assert _eligible(_scheme(QDType.INT4, QDType.FP8_E4M3), arch="sm80") == set()
    assert _eligible(_scheme(QDType.FP8_E5M2, QDType.FP8_E5M2), dtype="fp16") == set()
    assert _eligible(_scheme(QDType.MXFP4), dtype="fp16") == set()


def test_missing_humming_dependency_fails_without_replacing_staged_weights(
    fake_mesh, monkeypatch
):
    fake_mesh()
    original_import = builtins.__import__

    def without_humming(name, *args, **kwargs):
        if name == "humming" or name.startswith("humming."):
            raise ImportError("Humming intentionally unavailable for this test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_humming)
    layer = L.ReplicatedLinear(
        128,
        64,
        spec=HummingSpec(_scheme()),
        params_dtype=torch.bfloat16,
        device="cuda",
        prefix="projection",
        bias=False,
    )
    session = WeightLoadSession(layer)
    session.load({"projection.weight": _weight(64, 128)})
    with pytest.raises(ImportError, match="[Hh]umming"):
        session.finish()
    assert not layer._humming_prepared
    assert layer.weight.numel() == 0
    torch.testing.assert_close(layer._humming_pending_weight, _weight(64, 128))


NUMERICAL_CASES = [
    pytest.param(_scheme(QDType.INT4), 0.18, id="int4-a16"),
    pytest.param(_scheme(QDType.INT8), 0.03, id="int8-a16"),
    pytest.param(_scheme(QDType.FP8_E4M3), 0.08, id="fp8-e4m3-a16"),
    pytest.param(_scheme(QDType.FP8_E5M2), 0.12, id="fp8-e5m2-a16"),
    pytest.param(_scheme(QDType.MXFP4), 0.18, id="mxfp4-a16"),
    pytest.param(_scheme(QDType.NVFP4), 0.18, id="nvfp4-a16"),
    pytest.param(_scheme(QDType.FP8_E4M3, QDType.FP8_E4M3), 0.08, id="fp8-a8"),
    pytest.param(_scheme(QDType.FP8_E5M2, QDType.FP8_E5M2), 0.12, id="fp8-e5m2-a8"),
    pytest.param(_scheme(QDType.INT8, QDType.INT8), 0.03, id="int8-a8"),
    pytest.param(_scheme(QDType.INT4, QDType.FP8_E4M3), 0.18, id="int4-a8"),
    pytest.param(_scheme(QDType.INT4, QDType.INT8), 0.18, id="int4-int8-a8"),
    pytest.param(_block_fp8_scheme(), 0.08, id="fp8-block-a16"),
    pytest.param(_block_fp8_scheme(QDType.FP8_E5M2), 0.12, id="fp8-e5m2-block-a16"),
    pytest.param(
        _block_fp8_scheme(activation=QDType.FP8_E4M3), 0.08, id="fp8-block-a8"
    ),
    pytest.param(
        _block_fp8_scheme(QDType.FP8_E5M2, QDType.FP8_E5M2),
        0.12,
        id="fp8-e5m2-block-a8",
    ),
]
NUMERICAL_CASES += [
    pytest.param(
        replace(
            scheme,
            weight=replace(scheme.weight, granularity=granularity, block_shape=None),
        ),
        tolerance,
        id=f"{name}-{granularity.value}",
    )
    for granularity in (Granularity.PER_CHANNEL, Granularity.PER_TENSOR)
    for scheme, tolerance, name in (
        (_scheme(QDType.INT4), 0.20, "int4-a16"),
        (_scheme(QDType.INT8), 0.03, "int8-a16"),
        (_scheme(QDType.FP8_E4M3), 0.08, "fp8-e4m3-a16"),
        (_scheme(QDType.FP8_E4M3, QDType.FP8_E4M3), 0.08, "fp8-a8"),
    )
]


def _require_humming(scheme, dtype):
    pytest.importorskip("humming")
    if scheme.weight.dtype is QDType.MXFP4 and dtype is torch.float16:
        pytest.skip("humming-kernels 0.1.16 supports MXFP4 with BF16 compute only")
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    minimum = 80 if dtype == torch.bfloat16 else 75
    if scheme.input is not None and scheme.input.dtype in {
        QDType.FP8_E4M3,
        QDType.FP8_E5M2,
    }:
        minimum = 89
        if scheme.input.dtype is QDType.FP8_E5M2 and dtype is torch.float16:
            pytest.skip("FP8 E5M2 activations require BF16 output")
    if sm < minimum:
        pytest.skip(f"this Humming compute format requires SM{minimum}+")


def test_cuda_kernel_policy_quantizes_a_safetensors_linear(fake_mesh, tmp_path):
    _require_humming(_scheme(), torch.bfloat16)
    from safetensors.torch import save_file

    from phyai.kernel.bootstrap import kernel_selector_scope
    from phyai.kernel.policy import load_policy
    from phyai.kernel.selector import Selector
    from phyai.layers.linear.layers import resolve_linear_kernel
    from phyai.layers.quant.active import use_quant_plan
    from phyai.weights.loader import load_pretrained

    fake_mesh()
    policy_path = tmp_path / "kernel_policy.yaml"
    policy_path.write_text(
        """schema: phyai.kernel/v1
profile: static
quantization:
  method: rtn
  stage: load
  rules:
    - match: {glob: "mlp.*"}
      weight: {dtype: int4, block_shape: [1, 128]}
      input: null
rules:
  - id: humming-from-policy
    match: {op: gemm, quant.layout: humming}
    prefer: [humming.gemm.w4a16]
    params:
      compute_config: {use_batch_invariant: true}
""",
        encoding="utf-8",
    )
    catalog = Catalog()
    register(catalog)
    policy = load_policy(policy_path, catalog)
    selector = Selector(catalog, policy, device="cuda")
    checkpoint = tmp_path / "layer.safetensors"
    torch.manual_seed(2026)
    weight = torch.randn(128, 128, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(128, dtype=torch.bfloat16) * 0.01
    save_file({"mlp.proj.weight": weight, "mlp.proj.bias": bias}, str(checkpoint))

    with kernel_selector_scope(selector):
        with use_quant_plan(policy.quant_plan):
            layer = L.ReplicatedLinear(
                128,
                128,
                params_dtype=torch.bfloat16,
                device="cuda",
                prefix="mlp.proj",
                bias=True,
            )
        assert isinstance(layer.spec, HummingSpec)
        report = load_pretrained(layer, checkpoint, strict=True, progress=False)
        assert set(report.loaded) == {"mlp.proj.weight", "mlp.proj.bias"}
        assert not report.missing
        assert not report.unexpected
        assert not report.casts
        assert layer._humming_prepared
        assert layer._humming_pending_weight is None
        assert layer.weight.dtype is torch.int32

        x = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
        y, returned_bias = layer(x)
        selected = resolve_linear_kernel(layer, x, M=6, N=128, K=128)
        assert selected.kernel_id == "humming.gemm.w4a16"
        assert selected.params == {"compute_config": {"use_batch_invariant": True}}
        assert returned_bias is None
        reference = F.linear(x.float(), weight.cuda().float(), bias.cuda().float())
        relative_error = (y.float() - reference).norm() / reference.norm()
        assert relative_error < 0.18


@pytest.mark.parametrize("scheme,ptq_tolerance", NUMERICAL_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_ptq_matches_dequantized_weights_and_preserves_bias(
    fake_mesh, monkeypatch, scheme, ptq_tolerance, dtype
):
    _require_humming(scheme, dtype)
    from humming.schema.humming import HummingWeightSchema

    fake_mesh()
    torch.manual_seed(2026)
    reference_weights = []
    original_quant = HummingWeightSchema.quant_tensor

    def quant_tensor(cls, tensor, schema, param_dtype, **kwargs):
        tensors = original_quant(tensor, schema, param_dtype, **kwargs)
        reference_weights.append(schema.dequant_tensors(tensors).float())
        return tensors

    monkeypatch.setattr(HummingWeightSchema, "quant_tensor", classmethod(quant_tensor))
    if scheme.weight.dtype is QDType.NVFP4:
        import phyai.layers.quant.nvfp4 as nvfp4
        from phyai.layers.linear.backends.torch import _dequant_nvfp4_weight

        original_nvfp4_quant = nvfp4._quantize_nvfp4_linear

        def quant_nvfp4(tensor, block_size):
            packed, scales, global_scale = original_nvfp4_quant(tensor, block_size)
            reference_weights.append(
                _dequant_nvfp4_weight(
                    SimpleNamespace(
                        weight=packed,
                        weight_scale=scales,
                        weight_global_scale=global_scale,
                    )
                )
            )
            return packed, scales, global_scale

        monkeypatch.setattr(nvfp4, "_quantize_nvfp4_linear", quant_nvfp4)
    layer = L.ReplicatedLinear(
        128,
        128,
        scheme=scheme,
        params_dtype=dtype,
        device="cuda",
        prefix="projection",
        bias=True,
    )
    weight = torch.randn(128, 128, dtype=dtype) * 0.1
    bias = torch.randn(128, dtype=dtype) * 0.01
    session = WeightLoadSession(layer)
    session.load({"projection.weight": weight, "projection.bias": bias})
    session.finish()
    assert layer.weight.dtype == torch.int32
    assert layer.weight.device.type == "cuda"
    assert getattr(layer, "_humming_pending_weight", None) is None

    x = torch.randn(2, 3, 256, dtype=dtype, device="cuda")[..., ::2]
    y, returned_bias = layer(x)
    assert returned_bias is None
    assert y.shape == (2, 3, 128)
    assert y.dtype == dtype
    dequantized = F.linear(x.float(), reference_weights[0], layer.bias.float())
    error = (y.float() - dequantized).norm() / dequantized.norm()
    assert error < (0.06 if scheme.input is not None else 0.03)

    dense = F.linear(
        x.float(), weight.to(device="cuda", dtype=torch.float32), layer.bias.float()
    )
    ptq_error = (y.float() - dense).norm() / dense.norm()
    assert ptq_error < ptq_tolerance

    layer.skip_bias_add = True
    unbiased, returned_bias = layer(x)
    assert returned_bias is layer.bias
    torch.testing.assert_close(y, unbiased + layer.bias, atol=0, rtol=0)


@pytest.mark.parametrize("rank", [0, 3])
def test_cuda_row_parallel_adds_bias_on_rank_zero_only(fake_mesh, rank):
    _require_humming(_scheme(), torch.bfloat16)
    fake_mesh(tp_size=4, rank=rank)
    layer = L.RowParallelLinear(
        512,
        128,
        scheme=_scheme(),
        params_dtype=torch.bfloat16,
        device="cuda",
        prefix="projection",
        bias=True,
        input_is_parallel=False,
        reduce_results=False,
    )
    session = WeightLoadSession(layer)
    session.load(
        {
            "projection.weight": torch.randn(128, 512, dtype=torch.bfloat16) * 0.1,
            "projection.bias": torch.full((128,), 0.5, dtype=torch.bfloat16),
        }
    )
    session.finish()
    x = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
    y, returned_bias = layer(x)
    assert returned_bias is None
    layer.skip_bias_add = True
    unbiased, returned_bias = layer(x)
    assert returned_bias is layer.bias
    expected = unbiased + layer.bias if rank == 0 else unbiased
    torch.testing.assert_close(y, expected, atol=0, rtol=0)


def test_cuda_humming_graph_replays_with_new_inputs(fake_mesh):
    _require_humming(_scheme(), torch.bfloat16)
    fake_mesh()
    layer = L.ReplicatedLinear(
        128,
        128,
        scheme=_scheme(),
        params_dtype=torch.bfloat16,
        device="cuda",
        prefix="projection",
        bias=False,
    )
    session = WeightLoadSession(layer)
    session.load(
        {"projection.weight": torch.randn(128, 128, dtype=torch.bfloat16) * 0.1}
    )
    session.finish()
    x = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = layer(x)
    for factor in [0.5, -1.0]:
        x.mul_(factor)
        graph.replay()
        eager, _ = layer(x)
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)


def test_cuda_humming_graphs_own_locks_and_replay_independently(fake_mesh, monkeypatch):
    _require_humming(_scheme(), torch.bfloat16)
    import phyai.layers.linear.backends.humming as backend

    fake_mesh()
    torch.manual_seed(2026)
    layer = L.ReplicatedLinear(
        2048,
        128,
        scheme=_scheme(),
        params_dtype=torch.bfloat16,
        device="cuda",
        prefix="projection",
        bias=False,
    )
    session = WeightLoadSession(layer)
    session.load(
        {"projection.weight": torch.randn(128, 2048, dtype=torch.bfloat16) * 0.1}
    )
    session.finish()
    inputs = [
        torch.randn(6, 2048, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            for x in inputs:
                layer(x)
    torch.cuda.current_stream().wait_stream(warmup)
    expected = [layer(x)[0] for x in inputs]

    original_workspace = backend._workspace
    captured_locks = []
    capture_streams = []

    def record_workspace(module, device):
        locks = original_workspace(module, device)
        if torch.cuda.is_current_stream_capturing():
            captured_locks.append(locks)
            capture_streams.append(torch.cuda.current_stream(device).cuda_stream)
        return locks

    monkeypatch.setattr(backend, "_workspace", record_workspace)
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    outputs = []
    for graph, x in zip(graphs, inputs):
        with torch.cuda.graph(graph):
            outputs.append(layer(x)[0])

    assert len(captured_locks) == 2
    assert capture_streams[0] == capture_streams[1]
    # Check before replay: shared uninitialized locks can deadlock the GPU.
    assert captured_locks[0].data_ptr() != captured_locks[1].data_ptr()

    def check_output(output, reference):
        # Stream-K atomically accumulates BF16 partials in scheduling order.
        # Separate eager calls also differ, so compare their relative L2 error.
        error = (output.float() - reference.float()).norm() / reference.float().norm()
        assert error < 0.02

    for index in (1, 0):
        graphs[index].replay()
        check_output(outputs[index], expected[index])

    replay_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    current = torch.cuda.current_stream()
    for stream, graph in zip(replay_streams, graphs):
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            graph.replay()
    for stream in replay_streams:
        current.wait_stream(stream)
    for output, reference in zip(outputs, expected):
        check_output(output, reference)
