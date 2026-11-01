"""Linear layer integration tests.

Most tests run at ws=1 via the shared ``fake_mesh`` fixture. The parallel ops
bail out without communication, so we can exercise layer construction, weight
allocation, and forward() on CUDA. One multi-rank test spins up gloo workers
(CPU collectives by design) to verify the collective glue at ws>1.
"""

from __future__ import annotations

import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import phyai.layers.linear as L


# ---------------------------------------------------------------------------
# ws=1 fixture-driven tests
# ---------------------------------------------------------------------------


def test_replicated_linear_matches_F_linear_and_can_defer_its_bias(fake_mesh):
    fake_mesh()
    layer = L.ReplicatedLinear(
        in_features=32, out_features=16, bias=True, params_dtype=torch.bfloat16
    )
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)
    x = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")
    y, bias_out = layer(x)
    assert bias_out is None
    torch.testing.assert_close(y, F.linear(x, layer.weight, layer.bias), atol=0, rtol=0)

    # skip_bias_add hands the bias back instead of adding it.
    deferred = L.ReplicatedLinear(
        in_features=32,
        out_features=16,
        bias=True,
        skip_bias_add=True,
        params_dtype=torch.bfloat16,
    )
    deferred.weight.data.copy_(layer.weight.data)
    deferred.bias.data.copy_(layer.bias.data)
    y, bias_out = deferred(x)
    assert bias_out is deferred.bias
    torch.testing.assert_close(y, F.linear(x, layer.weight), atol=0, rtol=0)


def test_column_and_row_parallel_match_F_linear_at_ws1(fake_mesh):
    fake_mesh(tp_size=1)
    col = L.ColumnParallelLinear(
        in_features=32,
        out_features=16,
        group="dense_tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    row = L.RowParallelLinear(
        in_features=32,
        out_features=16,
        group="dense_tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(col.weight, std=0.05)
    nn.init.normal_(row.weight, std=0.05)
    x = torch.randn(5, 32, dtype=torch.bfloat16, device="cuda")

    y, _ = col(x)
    assert y.shape == (5, 16)  # ws=1: the per-rank output is the whole output
    torch.testing.assert_close(y, F.linear(x, col.weight), atol=0, rtol=0)
    y, _ = row(x)
    torch.testing.assert_close(y, F.linear(x, row.weight), atol=0, rtol=0)

    # Row-parallel adds its bias on rank 0 only; at ws=1 that is this rank.
    biased = L.RowParallelLinear(
        in_features=16,
        out_features=8,
        group="dense_tp",
        bias=True,
        params_dtype=torch.bfloat16,
    )
    biased.weight.data.zero_()
    biased.bias.data.fill_(3.0)
    y, _ = biased(torch.zeros(2, 16, dtype=torch.bfloat16, device="cuda"))
    assert torch.all(y == 3.0)


def test_construction_rejects_indivisible_shards(fake_mesh):
    fake_mesh(tp_size=4)
    with pytest.raises(ValueError, match="not divisible"):
        L.ColumnParallelLinear(
            in_features=8, out_features=30, group="dense_tp", bias=False
        )
    with pytest.raises(ValueError, match="not divisible"):
        L.RowParallelLinear(in_features=30, out_features=16, group="dense_tp")
    fake_mesh(tp_size=3)
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        L.QKVParallelLinear(
            hidden_size=32,
            head_dim=8,
            num_heads=6,
            num_kv_heads=2,
            group="dense_tp",
            bias=False,
        )


# ---------------------------------------------------------------------------
# MergedColumnParallelLinear
# ---------------------------------------------------------------------------


def test_merged_column_attaches_two_legs(fake_mesh):
    fake_mesh(tp_size=2)
    layer = L.MergedColumnParallelLinear(
        in_features=16,
        output_sizes=[16, 16],
        group="dense_tp",
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="model.layers.0.mlp.gate_up_proj",
    )
    # Per-rank partition sizes.
    assert layer.output_partition_sizes == [8, 8]
    assert layer.weight.shape == (16, 16)

    # Two HF source legs, each tagged with the leg index.
    assert layer.weight.hf_keys == [
        ("model.layers.0.mlp.gate_proj.weight", 0),
        ("model.layers.0.mlp.up_proj.weight", 1),
    ]

    # Apply: feed disk gate/up tensors via the loader, check fused weight is
    # laid out as [gate_rank0; up_rank0].
    disk_gate = torch.full((16, 16), 1.0, dtype=torch.bfloat16)
    disk_up = torch.full((16, 16), 2.0, dtype=torch.bfloat16)
    layer.weight.weight_loader(layer.weight, disk_gate, 0)
    layer.weight.weight_loader(layer.weight, disk_up, 1)
    assert torch.all(layer.weight.data[0:8] == 1.0)
    assert torch.all(layer.weight.data[8:16] == 2.0)


def test_merged_column_attaches_with_custom_hf_legs(fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.MergedColumnParallelLinear(
        in_features=16,
        output_sizes=[8, 8],
        group="dense_tp",
        bias=False,
        hf_legs=("w_gate", "w_up"),
        prefix="block.mlp.w12",
    )
    assert layer.weight.hf_keys == [
        ("block.mlp.w_gate.weight", 0),
        ("block.mlp.w_up.weight", 1),
    ]


# ---------------------------------------------------------------------------
# QKVParallelLinear
# ---------------------------------------------------------------------------


def test_attention_and_dense_weights_use_their_own_group_ranks(fake_mesh):
    from phyai.engine_config import (
        AttentionParallelConfig,
        DenseParallelConfig,
        ParallelConfig,
    )
    from phyai.parallel.mesh import Mesh
    from phyai.parallel.state import register_mesh
    from phyai.parallel.layout import build_rank_layout

    fake_mesh()
    layout = build_rank_layout(
        ParallelConfig(
            dense=DenseParallelConfig(tp_size=4),
            attention=AttentionParallelConfig(dp_size=2),
        )
    )
    register_mesh(Mesh(layout, rank=2))
    qkv = L.QKVParallelLinear(
        32, 8, 4, prefix="attn.qkv_proj", params_dtype=torch.float32
    )
    dense = L.ColumnParallelLinear(32, 32, prefix="mlp.fc", params_dtype=torch.float32)
    assert (qkv.tp_size, qkv.tp_rank) == (2, 0)
    assert (dense.tp_size, dense.tp_rank) == (4, 2)
    weight = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    qkv.weight.weight_loader(qkv.weight, weight, "q")
    dense.weight.weight_loader(dense.weight, weight, None)
    torch.testing.assert_close(qkv.weight[:16].cpu(), weight[:16])
    torch.testing.assert_close(dense.weight.cpu(), weight[16:24])


def test_qkv_linear_attaches_q_k_v_legs(fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.QKVParallelLinear(
        hidden_size=16,
        head_dim=4,
        num_heads=2,
        num_kv_heads=2,
        group="dense_tp",
        bias=False,
        params_dtype=torch.bfloat16,
        prefix="model.layers.0.self_attn.qkv_proj",
    )

    assert layer.weight.hf_keys == [
        ("model.layers.0.self_attn.q_proj.weight", "q"),
        ("model.layers.0.self_attn.k_proj.weight", "k"),
        ("model.layers.0.self_attn.v_proj.weight", "v"),
    ]

    disk_q = torch.full((2 * 4, 16), 1.0, dtype=torch.bfloat16)
    disk_k = torch.full((2 * 4, 16), 2.0, dtype=torch.bfloat16)
    disk_v = torch.full((2 * 4, 16), 3.0, dtype=torch.bfloat16)
    layer.weight.weight_loader(layer.weight, disk_q, "q")
    layer.weight.weight_loader(layer.weight, disk_k, "k")
    layer.weight.weight_loader(layer.weight, disk_v, "v")
    # [q | k | v] in fused param
    assert torch.all(layer.weight.data[0:8] == 1.0)
    assert torch.all(layer.weight.data[8:16] == 2.0)
    assert torch.all(layer.weight.data[16:24] == 3.0)


def test_qkv_linear_attaches_with_custom_hf_legs(fake_mesh):
    fake_mesh(tp_size=1)
    layer = L.QKVParallelLinear(
        hidden_size=16,
        head_dim=4,
        num_heads=2,
        num_kv_heads=2,
        group="dense_tp",
        bias=False,
        hf_legs={"q": "query", "k": "key", "v": "value"},
        prefix="block.attn.qkv_proj",
    )
    assert layer.weight.hf_keys == [
        ("block.attn.query.weight", "q"),
        ("block.attn.key.weight", "k"),
        ("block.attn.value.weight", "v"),
    ]


def test_qkv_gqa_replicas_share_kv_rows(fake_mesh):
    """tp=4 with 2 KV heads: each KV head is replicated on two ranks.

    effective_kv_heads = tp_size = 4, so the fused weight is q (64/4=16 rows)
    + k (8) + v (8) per rank. The K/V legs shard with replication_factor 2:
    ranks 0/1 read slot 0 (rows 0..8 of the source), ranks 2/3 read slot 1.
    """
    fake_mesh(tp_size=4, rank=0)
    layer_r0 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        group="dense_tp",
        bias=False,
        prefix="block.qkv_proj",
        params_dtype=torch.float32,
    )
    assert layer_r0.num_kv_replicas == 2
    assert layer_r0.output_partition_sizes == [16, 8, 8]
    assert layer_r0.weight.shape == (32, 32)
    # K has num_kv_heads * head_dim = 16 global rows, split into two
    # replicated slots. The fixture must match the checkpoint contract.
    disk_k = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32)
    layer_r0.weight.weight_loader(layer_r0.weight, disk_k, "k")
    # Q legs are 16 rows (offset 0); K leg goes to offset 16, size 8.
    k_at_r0 = layer_r0.weight.data.narrow(0, 16, 8).clone()

    fake_mesh(tp_size=4, rank=1)
    layer_r1 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        group="dense_tp",
        bias=False,
        prefix="block.qkv_proj",
        params_dtype=torch.float32,
    )
    layer_r1.weight.weight_loader(layer_r1.weight, disk_k, "k")
    k_at_r1 = layer_r1.weight.data.narrow(0, 16, 8).clone()
    # Ranks 0 and 1 share slot 0 -> same K rows.
    torch.testing.assert_close(k_at_r0, k_at_r1)

    fake_mesh(tp_size=4, rank=2)
    layer_r2 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        group="dense_tp",
        bias=False,
        prefix="block.qkv_proj",
        params_dtype=torch.float32,
    )
    layer_r2.weight.weight_loader(layer_r2.weight, disk_k, "k")
    k_at_r2 = layer_r2.weight.data.narrow(0, 16, 8).clone()
    # Rank 2 -> slot 1 -> DIFFERENT rows from rank 0/1.
    assert not torch.equal(k_at_r0, k_at_r2)


# ---------------------------------------------------------------------------
# Replicated / Column / Row attach smoke tests
# ---------------------------------------------------------------------------


def test_loaders_shard_by_group_rank_and_skip_unprefixed_params(fake_mesh):
    fake_mesh(tp_size=2, rank=1)
    col = L.ColumnParallelLinear(
        in_features=16,
        out_features=32,
        group="dense_tp",
        bias=False,
        prefix="block.fc",
        params_dtype=torch.float32,
    )
    assert col.weight.hf_keys == [("block.fc.weight", None)]
    disk = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    col.weight.weight_loader(col.weight, disk, None)
    torch.testing.assert_close(col.weight.data.cpu(), disk.narrow(0, 16, 16))

    fake_mesh(tp_size=4, rank=2)
    row = L.RowParallelLinear(
        in_features=64,
        out_features=16,
        group="dense_tp",
        bias=True,
        prefix="block.out_proj",
        params_dtype=torch.float32,
    )
    assert row.weight.hf_keys == [("block.out_proj.weight", None)]
    assert row.bias.hf_keys == [("block.out_proj.bias", None)]  # bias is replicated
    disk_w = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64)
    row.weight.weight_loader(row.weight, disk_w, None)
    torch.testing.assert_close(row.weight.data.cpu(), disk_w.narrow(1, 32, 16))

    fake_mesh()
    plain = L.ReplicatedLinear(
        in_features=4, out_features=8, bias=True, prefix="block.fc"
    )
    assert plain.weight.hf_keys == [("block.fc.weight", None)]
    assert plain.bias.hf_keys == [("block.fc.bias", None)]
    # No prefix -> no hf_keys attached -> the loader skips this param.
    assert not hasattr(L.ReplicatedLinear(4, 8, prefix="").weight, "hf_keys")


def test_replicated_linear_nvfp4_loader_quantizes_bf16_weight(fake_mesh):
    fake_mesh()
    layer = L.ReplicatedLinear(
        in_features=16,
        out_features=8,
        bias=False,
        spec=L.Nvfp4Spec(scale_layout="linear"),
        prefix="block.fc",
    )

    disk = torch.ones(8, 16, dtype=torch.bfloat16)
    layer.weight.weight_loader(layer.weight, disk, None)
    assert layer._nvfp4_pending_weight.shape == disk.shape
    layer.post_load()
    assert layer.weight.shape == (8, 8)
    assert layer.weight.dtype == torch.uint8
    assert layer._nvfp4_pending_weight is None


# ---------------------------------------------------------------------------
# Kernel policy integration
# ---------------------------------------------------------------------------


def policy_narrowing_gemm_to(tmp_path, pattern: str, *, role: str | None = None):
    """Write the YAML that replaced the deleted ``force this backend`` setting."""

    import yaml

    match: dict[str, object] = {"op": "gemm"}
    if role is not None:
        match["role"] = role
    document = {
        "schema": "phyai.kernel/v1",
        "rules": [
            {"id": "ab", "priority": 900, "match": match, "restrict_to": pattern}
        ],
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def selector_for(path, device: str):
    from phyai.kernel.bootstrap import resolve_policy
    from phyai.kernel.config import KernelConfig
    from phyai.kernel.registry import build_catalog
    from phyai.kernel.selector import Selector

    catalog = build_catalog()
    config = KernelConfig(config_path=str(path))
    return Selector(catalog, resolve_policy(config, catalog), device=device)


def bf16_gemm_query(*, role: str = "", M: int = 1024):
    from phyai.kernel.types import KernelQuery

    return KernelQuery.build(
        "gemm",
        role=role,
        dtype={"input": "bf16", "output": "bf16"},
        quant={"format": "bf16"},
        shape={
            "M": M,
            "N": 64,
            "K": 64,
        },
    )


def test_a_policy_rule_can_narrow_one_role_and_leave_the_rest(fake_mesh, tmp_path):
    """A rule can move one role onto torch while every other role stays put.

    The deleted ``PHYAI_FORCE_LINEAR_KERNEL`` setting could only move *every*
    linear; an A/B almost always wants one role, or the measurement means
    nothing.
    """

    fake_mesh()
    path = policy_narrowing_gemm_to(tmp_path, "torch.gemm.*", role="mlp.down")
    selector = selector_for(path, "nvidia:SM100")
    assert selector.explain(bf16_gemm_query(role="mlp.down")).selected == (
        "torch.gemm.bf16"
    )
    assert selector.explain(bf16_gemm_query(role="qkv_proj")).selected == (
        "flashinfer.gemm.bf16"
    )


def test_narrowing_to_an_unavailable_backend_falls_back_rather_than_failing(
    fake_mesh, tmp_path
):
    """A rule, not an override, so it stays soft.

    A strict force would make "use FlashInfer for GEMM" unusable on a CPU box;
    the rule says where work should go, not that it can.
    """

    fake_mesh()
    path = policy_narrowing_gemm_to(tmp_path, "flashinfer.gemm.*")
    selector = selector_for(path, "cpu")
    query = bf16_gemm_query(M=8)
    assert selector.explain(query).selected == "torch.gemm.bf16"


def test_narrowing_to_an_unknown_kernel_is_a_load_time_error(fake_mesh, tmp_path):
    """Caught when the policy loads, not when a forward first runs."""

    from phyai.kernel.policy import PolicyError

    fake_mesh()
    path = policy_narrowing_gemm_to(tmp_path, "cutlass.gemm.*")
    with pytest.raises(PolicyError, match="matches no kernel"):
        selector_for(path, "nvidia:SM100")


# ---------------------------------------------------------------------------
# Runtime capability errors
# ---------------------------------------------------------------------------


def test_fp8_on_sm86_reports_arch_rejection():
    """An unsupported real query names the failed architecture predicate."""
    from phyai.kernel.call import select
    from phyai.kernel.selector import NoKernelError

    with pytest.raises(NoKernelError) as excinfo:
        select(
            "gemm",
            device="nvidia:SM86",
            dtype={"input": "bf16", "output": "bf16"},
            quant={
                "format": "fp8_e4m3",
                "granularity": "per_tensor",
                "scale_dtype": "fp32",
            },
            shape={"M": 8, "N": 64, "K": 64},
        )
    assert "device.arch >= sm89" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Real ws>1 via gloo — confirms collective wiring
# ---------------------------------------------------------------------------


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _gloo_worker(rank, world_size, port, test_fn, err_queue):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        try:
            test_fn(rank, world_size)
        finally:
            try:
                dist.destroy_process_group()
            except Exception:
                pass
    except BaseException as e:
        err_queue.put((rank, repr(e), traceback.format_exc()))


def _run_gloo(test_fn, *, world_size: int, timeout_s: float = 30.0) -> None:
    # Use ``fork`` instead of ``spawn`` so the workers inherit the parent's
    # already-imported modules — the test module isn't on ``sys.path`` under
    # pytest ``--import-mode=importlib`` and spawn would fail unpickling.
    # Fork is safe here because the gloo backend doesn't touch CUDA.
    ctx = mp.get_context("fork")
    err_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_gloo_worker,
            args=(r, world_size, port, test_fn, err_queue),
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=timeout_s)
        if p.is_alive():
            p.terminate()
            p.join()
            raise TimeoutError("gloo worker hung")
    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get_nowait())
    if errors:
        r, e, tb = errors[0]
        raise AssertionError(f"worker rank={r} failed: {e}\n{tb}")
    for i, p in enumerate(procs):
        if p.exitcode != 0:
            raise AssertionError(f"gloo worker rank={i} exited with code {p.exitcode}")


def _w_column_tp2_column_row_equiv(rank, world_size):
    """Column-then-Row at tp=2 must reproduce the un-sharded reference.

    ``reduce_results=True`` exercises the layer's exit all_reduce on the tp
    group (gloo here), so the row output on every rank equals the full
    F.linear ∘ F.linear result.
    """
    import phyai.parallel as P
    import phyai.layers.linear as L

    torch.manual_seed(0)
    from phyai.engine_config import ParallelConfig, DenseParallelConfig

    P.init(
        ParallelConfig(dense=DenseParallelConfig(tp_size=world_size)),
        device="cpu",
        backend="gloo",
        enable_pynccl=False,
    )

    hidden = 16
    inter = 32
    W1 = torch.randn(inter, hidden, dtype=torch.float32) * 0.1
    W2 = torch.randn(hidden, inter, dtype=torch.float32) * 0.1
    x = torch.randn(4, hidden, dtype=torch.float32) * 0.1

    col = L.ColumnParallelLinear(
        in_features=hidden,
        out_features=inter,
        group="dense_tp",
        bias=False,
        params_dtype=torch.float32,
        device="cpu",
        prefix="col",
    )
    row = L.RowParallelLinear(
        in_features=inter,
        out_features=hidden,
        group="dense_tp",
        bias=False,
        reduce_results=True,
        params_dtype=torch.float32,
        device="cpu",
        prefix="row",
    )
    col.weight.weight_loader(col.weight, W1, None)
    row.weight.weight_loader(row.weight, W2, None)

    y_col, _ = col(x)
    assert y_col.shape == (
        4,
        inter // world_size,
    ), f"rank={rank} got y_col.shape={y_col.shape}"
    # y_col per rank is x @ W1_this_rank_cols.T — verify.
    start = rank * (inter // world_size)
    end = start + (inter // world_size)
    expected_col = F.linear(x, W1[start:end, :])
    torch.testing.assert_close(y_col, expected_col, atol=1e-6, rtol=1e-6)

    # Row forward all-reduces the per-rank partials: every rank now holds
    # the full x @ W1.T @ W2.T.
    y_row, _ = row(y_col)
    assert y_row.shape == (4, hidden)
    expected = F.linear(F.linear(x, W1), W2)
    torch.testing.assert_close(y_row, expected, atol=1e-5, rtol=1e-5)


def test_column_then_row_tp2_numerical_equivalence():
    _run_gloo(_w_column_tp2_column_row_equiv, world_size=2)
