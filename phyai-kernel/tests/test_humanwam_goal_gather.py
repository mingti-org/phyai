"""Bit-preserving IPC gathers, including padded tokens and graph replay."""

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from phyai.models.humanwam_goal.collective_humanwam_goal import HumanWAMGoalGather


def _gather_worker(rank, size, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=120),
    )
    owner = HumanWAMGoalGather(max_bytes=4 << 20)
    try:
        for columns in (24, 1024, 3072):
            rows, part, text = 17, 11, 3
            tokens = part * size - 1
            value = torch.empty((1, rows, columns), device="cuda", dtype=torch.bfloat16)
            local_kv = torch.empty((part, 2, 8, 128), device="cuda", dtype=value.dtype)
            text_key = torch.empty((1, text, 8, 128), device="cuda", dtype=value.dtype)
            text_value = torch.empty_like(text_key)

            def fill(iteration):
                value.fill_(rank + iteration * size)
                local_kv[:, 0].fill_(rank + iteration * size)
                local_kv[:, 1].fill_(-rank - iteration * size)
                text_key.fill_(iteration + 99)
                text_value.fill_(-iteration - 99)

            def launch():
                gathered = owner.gather(value)
                owner.kv_workspace(part).copy_(local_kv)
                key, val = owner.gather_kv(
                    text_key, text_value, part=part, tokens=tokens
                )
                return gathered, key, val

            def check(outputs, iteration):
                gathered, key, val = outputs
                expected = torch.cat(
                    [torch.full_like(value, i + iteration * size) for i in range(size)],
                    -1,
                )
                assert torch.equal(
                    gathered.view(torch.uint8), expected.view(torch.uint8)
                )
                gen = torch.cat(
                    [
                        torch.full_like(
                            text_key[:, :1].expand(1, part, 8, 128),
                            i + iteration * size,
                        )
                        for i in range(size)
                    ],
                    1,
                )[:, :tokens]
                assert torch.equal(key, torch.cat((text_key, gen), 1))
                assert torch.equal(val, torch.cat((text_value, -gen), 1))

            fill(0)
            check(launch(), 0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = launch()
            for iteration in (1, 2):
                fill(iteration)
                graph.replay()
                check(outputs, iteration)
            graph.reset()

        with pytest.raises(ValueError, match="BF16"):
            owner.gather(torch.ones(16, device="cuda", dtype=torch.float32))
        with pytest.raises(ValueError, match="bound"):
            owner.kv_workspace(owner.max_bytes)
    finally:
        owner.close()
        owner.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
def test_ipc_gather_graph_refresh(size, tmp_path, monkeypatch):
    if torch.cuda.device_count() < size:
        pytest.skip(f"Requires {size} CUDA devices")
    if any(
        not torch.cuda.can_device_access_peer(a, b)
        for a in range(size)
        for b in range(size)
        if a != b
    ):
        pytest.skip("Requires CUDA IPC peer access between every rank")
    # Spawned workers must be able to import this module in pytest's importlib mode.
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    rendezvous = (tmp_path / "rendezvous").as_uri()
    context = mp.spawn(_gather_worker, args=(size, rendezvous), nprocs=size, join=False)
    try:
        for _ in range(24):
            if context.join(timeout=10):
                break
        else:
            pytest.fail("IPC gather workers exceeded their timeout")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
