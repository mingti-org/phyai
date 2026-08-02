"""Linear layer integration tests.

Most tests run at ws=1 via a mocked Mesh — the parallel ops bail out
without touching the dispatcher, so we can exercise layer construction,
weight allocation, and forward() on CPU. A couple of multi-rank tests
spin up gloo workers to verify the collective glue really fires at ws>1.
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
from phyai.parallel.exceptions import NoBackendError


# ---------------------------------------------------------------------------
# ws=1 fixture-driven tests
# ---------------------------------------------------------------------------


def _init_default_dispatcher():
    """Re-init the dispatcher without flashinfer so tests don't depend on it."""
    return L.init(register_flashinfer=False, validate=False)


def test_replicated_linear_bf16_matches_F_linear(fake_mesh):
    fake_mesh(name="model")
    _init_default_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=32,
        out_features=16,
        bias=True,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)

    x = torch.randn(4, 32, dtype=torch.bfloat16)
    y, bias_out = layer(x)
    assert bias_out is None
    ref = F.linear(x, layer.weight, layer.bias)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_replicated_linear_skip_bias_add_returns_bias(fake_mesh):
    fake_mesh()
    _init_default_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=16,
        out_features=8,
        bias=True,
        skip_bias_add=True,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)
    nn.init.normal_(layer.bias, std=0.05)

    x = torch.randn(2, 16, dtype=torch.bfloat16)
    y, bias_out = layer(x)
    assert bias_out is layer.bias
    # y should NOT include bias (skip_bias_add=True).
    ref = F.linear(x, layer.weight, None)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_column_parallel_ws1_matches_F_linear(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.ColumnParallelLinear(
        in_features=32,
        out_features=16,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)

    x = torch.randn(5, 32, dtype=torch.bfloat16)
    y, _ = layer(x)
    ref = F.linear(x, layer.weight)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)
    # With ws=1 and gather_output default False, shape is the per-rank output.
    assert y.shape == (5, 16)


def test_column_parallel_rejects_indivisible_split(fake_mesh):
    fake_mesh(sizes={"tp": 4})
    _init_default_dispatcher()
    with pytest.raises(ValueError, match="not divisible"):
        L.ColumnParallelLinear(
            in_features=8,
            out_features=30,  # 30 % 4 != 0
            axis="tp",
            bias=False,
        )


def test_row_parallel_ws1_matches_F_linear(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.RowParallelLinear(
        in_features=32,
        out_features=16,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    nn.init.normal_(layer.weight, std=0.05)

    x = torch.randn(3, 32, dtype=torch.bfloat16)
    y, _ = layer(x)
    ref = F.linear(x, layer.weight)
    torch.testing.assert_close(y, ref, atol=0, rtol=0)


def test_row_parallel_bias_only_on_rank0(fake_mesh):
    """At ws=1 the bias IS added (rank==0)."""
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.RowParallelLinear(
        in_features=16,
        out_features=8,
        axis="tp",
        bias=True,
        params_dtype=torch.bfloat16,
    )
    layer.weight.data.zero_()
    layer.bias.data.fill_(3.0)

    x = torch.zeros(2, 16, dtype=torch.bfloat16)
    y, _ = layer(x)
    assert torch.all(y == 3.0)


def test_row_parallel_rejects_indivisible_in(fake_mesh):
    fake_mesh(sizes={"tp": 4})
    _init_default_dispatcher()
    with pytest.raises(ValueError, match="not divisible"):
        L.RowParallelLinear(in_features=30, out_features=16, axis="tp")


# ---------------------------------------------------------------------------
# MergedColumnParallelLinear
# ---------------------------------------------------------------------------


def test_merged_column_construct_fused_weight(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.MergedColumnParallelLinear(
        in_features=32,
        output_sizes=[16, 16],  # gate/up style
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    # Weight is fused: (gate_size + up_size, in)
    assert layer.weight.shape == (32, 32)
    assert layer.output_partition_sizes == [16, 16]
    assert layer.output_sizes_global == [16, 16]


def test_merged_column_attaches_two_legs(fake_mesh):
    fake_mesh(sizes={"tp": 2})
    _init_default_dispatcher()
    layer = L.MergedColumnParallelLinear(
        in_features=16,
        output_sizes=[16, 16],
        axis="tp",
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
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.MergedColumnParallelLinear(
        in_features=16,
        output_sizes=[8, 8],
        axis="tp",
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


def test_qkv_linear_no_gqa_shapes(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=4,
        num_kv_heads=4,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    # out = q (4*8) + k (4*8) + v (4*8) = 96
    assert layer.weight.shape == (96, 32)
    assert layer.num_kv_replicas == 1


def test_qkv_linear_gqa_replicates_kv(fake_mesh):
    """tp_size=4 with 2 KV heads -> each rank gets the whole KV replicated twice."""
    fake_mesh(sizes={"tp": 4})
    _init_default_dispatcher()
    layer = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        params_dtype=torch.bfloat16,
    )
    # Effective kv_heads = tp_size (2 kv heads replicated 4//2=2 times).
    # q_size = 8 * 8 = 64, kv_size = 4 * 8 = 32.
    # per-rank: q = 64/4 = 16, kv = 32/4 = 8 each.
    assert layer.num_kv_replicas == 2
    assert layer.output_partition_sizes == [16, 8, 8]
    assert layer.weight.shape == (32, 32)


def test_qkv_linear_rejects_nonmultiple_tp(fake_mesh):
    fake_mesh(sizes={"tp": 3})
    _init_default_dispatcher()
    with pytest.raises(ValueError, match="multiple of num_kv_heads"):
        L.QKVParallelLinear(
            hidden_size=32,
            head_dim=8,
            num_heads=6,
            num_kv_heads=2,
            axis="tp",
            bias=False,
        )


def test_qkv_linear_attaches_q_k_v_legs(fake_mesh):
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.QKVParallelLinear(
        hidden_size=16,
        head_dim=4,
        num_heads=2,
        num_kv_heads=2,
        axis="tp",
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
    fake_mesh(sizes={"tp": 1})
    _init_default_dispatcher()
    layer = L.QKVParallelLinear(
        hidden_size=16,
        head_dim=4,
        num_heads=2,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        hf_legs={"q": "query", "k": "key", "v": "value"},
        prefix="block.attn.qkv_proj",
    )
    assert layer.weight.hf_keys == [
        ("block.attn.query.weight", "q"),
        ("block.attn.key.weight", "k"),
        ("block.attn.value.weight", "v"),
    ]


def test_qkv_gqa_replica_share_kv_slot(fake_mesh):
    """tp=4 with 2 KV heads: ranks 0/1 share K/V slot 0; ranks 2/3 share slot 1."""
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 0})
    _init_default_dispatcher()
    # Build a fake K source with row patterns we can identify after sharding.
    # effective_kv_heads = 4 (tp_size), kv_size_global = 4*8 = 32 rows total.
    # Per-rank kv_local = 32 / 4 = 8 rows.
    # num_kv_replicas = 4//2 = 2 -> kv_world = 4/2 = 2 -> kv_per_slot = 32/2 = 16.
    # rank 0/1 -> slot 0 -> rows 0..16 (each takes 8 of those 16).
    # Wait, the loader narrow uses size=kv_local=8 but rank=tp_rank//2.
    # rank0/1: slot=0, narrow(0, 0*8, 8) -> rows 0..8.
    # rank2/3: slot=1, narrow(0, 1*8, 8) -> rows 8..16.
    # So both ranks at slot 0 read the SAME rows 0..8.
    layer_r0 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        prefix="block.qkv_proj",
        params_dtype=torch.float32,
    )
    disk_k = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
    layer_r0.weight.weight_loader(layer_r0.weight, disk_k, "k")
    # Q legs are 16 rows (offset 0); K leg goes to offset 16, size 8.
    k_at_r0 = layer_r0.weight.data.narrow(0, 16, 8).clone()

    fake_mesh(sizes={"tp": 4}, ranks={"tp": 1})
    layer_r1 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        axis="tp",
        bias=False,
        prefix="block.qkv_proj",
        params_dtype=torch.float32,
    )
    layer_r1.weight.weight_loader(layer_r1.weight, disk_k, "k")
    k_at_r1 = layer_r1.weight.data.narrow(0, 16, 8).clone()
    # Ranks 0 and 1 share slot 0 -> same K rows.
    torch.testing.assert_close(k_at_r0, k_at_r1)

    fake_mesh(sizes={"tp": 4}, ranks={"tp": 2})
    layer_r2 = L.QKVParallelLinear(
        hidden_size=32,
        head_dim=8,
        num_heads=8,
        num_kv_heads=2,
        axis="tp",
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


def test_replicated_linear_attaches(fake_mesh):
    fake_mesh()
    _init_default_dispatcher()
    layer = L.ReplicatedLinear(
        in_features=4, out_features=8, bias=True, prefix="block.fc"
    )
    assert layer.weight.hf_keys == [("block.fc.weight", None)]
    assert layer.bias.hf_keys == [("block.fc.bias", None)]


def test_replicated_linear_nvfp4_loader_quantizes_bf16_weight(fake_mesh):
    fake_mesh()
    _init_default_dispatcher()
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


def test_column_parallel_attaches_tp2_rank1(fake_mesh):
    fake_mesh(sizes={"tp": 2}, ranks={"tp": 1})
    _init_default_dispatcher()
    layer = L.ColumnParallelLinear(
        in_features=16,
        out_features=32,
        axis="tp",
        bias=False,
        prefix="block.fc",
        params_dtype=torch.float32,
    )
    assert layer.weight.hf_keys == [("block.fc.weight", None)]
    # Apply: source has 32 rows; rank 1 writes rows 16..32 of source into local 16-row tensor.
    disk = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16)
    layer.weight.weight_loader(layer.weight, disk, None)
    torch.testing.assert_close(layer.weight.data, disk.narrow(0, 16, 16))


def test_row_parallel_attaches_dim1_shard(fake_mesh):
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 2})
    _init_default_dispatcher()
    layer = L.RowParallelLinear(
        in_features=64,
        out_features=16,
        axis="tp",
        bias=True,
        prefix="block.out_proj",
        params_dtype=torch.float32,
    )
    assert layer.weight.hf_keys == [("block.out_proj.weight", None)]
    # Bias replicated (full copy, no shard).
    assert layer.bias.hf_keys == [("block.out_proj.bias", None)]

    disk_w = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64)
    layer.weight.weight_loader(layer.weight, disk_w, None)
    # Rank 2 takes columns [32:48) along input dim.
    torch.testing.assert_close(layer.weight.data, disk_w.narrow(1, 32, 16))


def test_empty_prefix_skips_attach(fake_mesh):
    fake_mesh()
    _init_default_dispatcher()
    layer = L.ReplicatedLinear(in_features=4, out_features=8, prefix="")
    # No prefix -> no hf_keys attached -> loader skips this param.
    assert not hasattr(layer.weight, "hf_keys")


# ---------------------------------------------------------------------------
# init() + force-env integration
# ---------------------------------------------------------------------------


def test_init_registers_torch_fallback_and_validates(fake_mesh):
    fake_mesh()
    d = L.init(register_flashinfer=False, validate=True, sample_specs=["bf16"])
    assert d is L.get_linear_dispatcher()
    # Picks TorchKernel for bf16.
    k = d.select(
        spec_id="bf16",
        M=8,
        N=64,
        K=64,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert k.name == "torch"


def test_init_force_env_overrides(fake_mesh, monkeypatch):
    monkeypatch.setenv("PHYAI_FORCE_LINEAR_KERNEL", "torch")
    fake_mesh()
    d = L.init(register_flashinfer=True, validate=False)
    # Even if flashinfer preferred for bf16 prefill, force->torch.
    k = d.select(
        spec_id="bf16",
        M=1024,
        N=64,
        K=64,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert k.name == "torch"


# ---------------------------------------------------------------------------
# Hardware-gated validate(): fp8 has no kernel below sm_89, so a pure-bf16
# deployment on Ampere (sm_86) / CPU (sm_arch -> 0) must still init cleanly.
# Unsupported specs error lazily at the matmul, never at startup.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sm, expected",
    [
        (0, ["bf16"]),  # CPU
        (80, ["bf16"]),  # Ampere A100 — no fp8
        (86, ["bf16"]),  # Ampere A10/A40 — the reported failure
        (89, ["bf16", "fp8_per_tensor", "fp8_per_channel"]),  # Ada
        (90, ["bf16", "fp8_per_tensor", "fp8_per_channel"]),  # Hopper
        (
            100,
            [
                "bf16",
                "fp8_per_tensor",
                "fp8_per_channel",
                "fp8_block_128_128",
                "nvfp4_block_16_128x4",
            ],
        ),
    ],
)
def test_supported_specs_for_sm(sm, expected):
    assert L.supported_specs_for_sm(sm) == expected


def test_init_validate_succeeds_on_sm86_bf16_only(fake_mesh, monkeypatch):
    """Regression: a bf16-only model on sm_86 used to die in validate()
    demanding an fp8 kernel that no backend provides below sm_89."""
    fake_mesh()
    # init() reads the package-level sm_arch binding to choose which specs
    # validate() must cover; the dispatcher reads its own binding for _sm.
    monkeypatch.setattr("phyai.layers.linear.sm_arch", lambda *a, **k: 86)
    monkeypatch.setattr("phyai.layers.linear.dispatch.sm_arch", lambda *a, **k: 86)

    # No exception: validate only requires the specs sm_86 can actually run.
    d = L.init(register_flashinfer=False, validate=True)

    # bf16 still resolves to the torch fallback.
    k = d.select(
        spec_id="bf16",
        M=8,
        N=64,
        K=64,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    assert k.name == "torch"


def test_fp8_on_sm86_errors_lazily_at_dispatch_not_init(fake_mesh, monkeypatch):
    """The user's design intent: unsupported -> don't register, raise at the
    call site. init() succeeds; the fp8 request is what fails, and clearly."""
    fake_mesh()
    monkeypatch.setattr("phyai.layers.linear.sm_arch", lambda *a, **k: 86)
    monkeypatch.setattr("phyai.layers.linear.dispatch.sm_arch", lambda *a, **k: 86)

    d = L.init(register_flashinfer=False, validate=True)  # does NOT raise

    with pytest.raises(NoBackendError):
        d.select(
            spec_id="fp8_per_tensor",
            M=8,
            N=64,
            K=64,
            in_dtype=torch.float8_e4m3fn,
            out_dtype=torch.bfloat16,
        )


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
    """Column-then-Row at tp=2 should produce correct per-rank outputs.

    We disable the final all_reduce (``reduce_results=False``) because the
    collective layer has a pre-existing torch 2.10 compatibility issue
    that is orthogonal to the Linear tests. Summing the per-rank partials
    across ranks (done in the parent via ``err_queue``) reproduces the
    un-sharded F.linear ∘ F.linear reference.
    """
    import phyai.parallel as P
    import phyai.layers.linear as L

    torch.manual_seed(0)
    P.init(layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo")
    L.init(register_flashinfer=False, validate=False)

    hidden = 16
    inter = 32
    W1 = torch.randn(inter, hidden, dtype=torch.float32) * 0.1
    W2 = torch.randn(hidden, inter, dtype=torch.float32) * 0.1
    x = torch.randn(4, hidden, dtype=torch.float32) * 0.1

    col = L.ColumnParallelLinear(
        in_features=hidden,
        out_features=inter,
        axis="tp",
        bias=False,
        params_dtype=torch.float32,
        prefix="col",
    )
    row = L.RowParallelLinear(
        in_features=inter,
        out_features=hidden,
        axis="tp",
        bias=False,
        reduce_results=False,
        params_dtype=torch.float32,
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

    # Row forward without reduce returns per-rank partial sum.
    y_row_partial, _ = row(y_col)
    assert y_row_partial.shape == (4, hidden)
    # Each rank's partial = y_col @ W2_this_rank_cols.T using W2 row-sliced.
    W2_rank = W2[:, start:end]
    expected_partial = F.linear(y_col, W2_rank)
    torch.testing.assert_close(
        y_row_partial,
        expected_partial,
        atol=1e-6,
        rtol=1e-6,
    )


def test_column_then_row_tp2_numerical_equivalence():
    _run_gloo(_w_column_tp2_column_row_equiv, world_size=2)
