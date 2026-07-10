"""End-to-end pi0.5 data-parallel demo under torchrun (DP=N, TP=1).

Multi-GPU ("wn") sibling of ``run_pi05.py``. Drives the ``pi05_wn`` engine
plugin: a full pi0.5 replica per GPU, the ``--batch-size`` request split
statically across ``--dp`` GPUs (each card runs ``ceil(batch/dp)`` robots), rank
0 scatters the shards and gathers the action chunks back. Only rank 0 holds the
final ``(batch, chunk, action_dim)`` tensor.

Launch under torchrun with ``--nproc_per_node`` equal to ``--dp``::

    torchrun --nproc_per_node=8 examples/pi05/run_pi05_wn.py --dp 8 \\
        --checkpoint /path/to/pi05_base/ --batch-size 32

Inputs are random (canonical tensors); action numbers are meaningless. This
verifies the DP wiring + timing. Requires CUDA + NCCL.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time

import torch


def _resolve_topology(dp: int) -> tuple[int, int, bool]:
    """Reconcile ``--dp`` with the torchrun launch env.

    Returns ``(local_rank, dp, is_main)``. World size == dp (one process/rank).
    ``dp == 1`` runs in-process (no torchrun needed). Rank 0 is the Router and
    holds the gathered result.
    """
    world = dp
    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and env_world != world:
        raise SystemExit(
            f"--dp {dp} requires torchrun --nproc_per_node={world} "
            f"(saw WORLD_SIZE={env_world}). Example:\n"
            f"  torchrun --nproc_per_node={world} examples/pi05/run_pi05_wn.py "
            f"--dp {dp} --checkpoint <ckpt> --batch-size 32"
        )
    if world == 1 and env_world != 1:
        raise SystemExit(
            f"launched under torchrun (WORLD_SIZE={env_world}) but --dp is 1; "
            f"set --dp to use all ranks."
        )
    return local_rank, dp, local_rank == 0


@contextlib.contextmanager
def _timed(label: str, store: dict):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store[label] = time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="pi05_base checkpoint dir")
    parser.add_argument(
        "--dp",
        type=int,
        default=1,
        help="Data-parallel degree; world_size = dp must equal torchrun --nproc_per_node.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Total robots across all ranks (split ceil(batch/dp) per card).",
    )
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument(
        "--vision-dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import (
        DeviceConfig,
        EngineConfig,
        ParallelConfig,
        RuntimeConfig,
    )
    from phyai.models.pi05.configuration_pi05 import PI05Config
    from phyai.models.pi05.main_pi05_wn import PI05WNArgs
    from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request
    from phyai.utils import load_config

    local_rank, dp_size, is_main = _resolve_topology(args.dp)
    device = f"cuda:{local_rank}"
    dtype = torch.bfloat16
    vision_dtype = torch.float32 if args.vision_dtype == "float32" else None

    def log(*a, **k):
        if is_main:
            print(*a, **k)

    plugin_cfg = load_config(args.checkpoint, PI05Config)
    inputs_image_shape = [
        [plugin_cfg.vision.image_size, plugin_cfg.vision.image_size, 3]
        for _ in range(args.num_images)
    ]

    log(f"[engine] creating pi0.5 data-parallel engine (dp={dp_size})...")
    engine = Engine(
        EngineArgs(
            plugin="pi05_wn",
            plugin_args=PI05WNArgs(
                checkpoint_dir=args.checkpoint,
                max_batch_size=args.batch_size,
                vision_params_dtype=vision_dtype,
                inputs_image_shape=inputs_image_shape,
            ),
            config=EngineConfig(
                device=DeviceConfig(target=device, params_dtype=dtype),
                parallel=ParallelConfig(world_size=dp_size, dp_size=dp_size, tp_size=1),
                runtime=RuntimeConfig(use_cuda_graph=True),
            ),
        )
    )

    timings: dict[str, float] = {}
    try:
        # Every rank builds a full canonical request; only rank 0's rows are
        # scattered (others' copies are ignored — they recv their shard).
        B = args.batch_size
        img = plugin_cfg.vision.image_size
        request = PI05Request(
            pixel_values=torch.rand(
                B, args.num_images, 3, img, img, dtype=dtype, device=device
            ),
            input_ids=torch.zeros(
                B, plugin_cfg.tokenizer_max_length, dtype=torch.int64, device=device
            ),
            lang_lens=torch.ones(B, dtype=torch.int64, device=device),
        )
        request.input_ids[:, 0] = 2

        with _timed("warmup", timings):
            engine.step(request)
        with _timed("inference", timings):
            actions = engine.step(request)

        if is_main:
            print(f"[run] dp={dp_size} total_batch={B}")
            print(f"action chunk shape : {tuple(actions.shape)}")
            print(f"action chunk device: {actions.device}")
            print(f"first action row   : {actions[0, 0].float().tolist()}")
            print(
                f"timing (s): warmup={timings['warmup']:.2f} "
                f"inference={timings['inference']:.2f}"
            )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
