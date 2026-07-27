"""pi0.5 data-parallel ("wn") scheduler — shard the batch over the dp axis.

Data parallelism for pi0.5 inference: every rank holds a full model replica
(tp=cfg=1) and processes a contiguous shard of the request batch. Rank 0 is the
Router — it holds the incoming batch, scatters each rank's shard over the mesh
``dp`` axis with point-to-point ``send``/``recv``, every rank runs the composed
single-card :class:`PI05WS1Scheduler` on its shard, and the per-rank action
chunks are recombined with one ``all_gather`` over ``dp`` back to rank 0.

Unlike cosmos3's CFG-parallel scheduler (which folds an all-gather *inside* the
denoise loop), pi0.5 DP is request-level: no in-loop communication, so this
scheduler composes the battle-tested ws1 compute unchanged instead of rewriting
its loop. At ``dp_size == 1`` every collective short-circuits and this degenerates
to a plain single-card run.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

import phyai.parallel as P
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request, PI05WS1Scheduler
from phyai.runtime.schedule import Scheduler
from phyai.utils import this_rank_log


logger = logging.getLogger(__name__)


def _dp_rank_size() -> tuple[int, int]:
    """Current ``(dp_rank, dp_size)`` on the data-parallel axis.

    ``(0, 1)`` when there is no mesh / no ``dp`` axis (single-process runs),
    matching how the tp/cfg helpers degrade in the cosmos3 schedulers.
    """
    try:
        mesh = P.default_mesh()
        return mesh.axis_local_rank("dp"), mesh.axis_size("dp")
    except Exception:
        return 0, 1


def _pad_rows(t: torch.Tensor, total: int) -> torch.Tensor:
    """Pad/truncate ``t`` along dim 0 to exactly ``total`` rows.

    Growing repeats the last row (kept valid rather than zeroed, so a padded
    ``lang_lens`` stays a real length); the padded tail is discarded after the
    gather anyway. Shrinking takes the first ``total`` rows.
    """
    b = t.shape[0]
    if b == total:
        return t
    if b > total:
        return t[:total].contiguous()
    pad = t[-1:].expand(total - b, *t.shape[1:])
    return torch.cat([t, pad], dim=0).contiguous()


class PI05WNScheduler(Scheduler):
    """Data-parallel pi0.5 orchestrator (rank-0 Router + dp-axis scatter/gather)."""

    def __init__(
        self,
        local: PI05WS1Scheduler,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.local = local
        self.device = torch.device(device) if device is not None else local.device
        self.dtype = local.params_dtype
        self.cfg = local.cfg
        self.num_images = local.num_images
        # Each rank's ws1 is sized for the per-rank shard; that IS per_rank_B.
        self.per_rank_B = local.max_batch_size
        self.dp_rank, self.dp_size = _dp_rank_size()

    def setup(self) -> None:
        if self.dp_size > 1:
            world = dist.get_world_size() if dist.is_initialized() else 1
            if world != self.dp_size:
                raise RuntimeError(
                    f"PI05WNScheduler assumes pure data parallelism (tp=cfg=1): "
                    f"dp_size={self.dp_size} must equal world_size={world}. The "
                    f"point-to-point scatter addresses peers by dp-axis-local rank, "
                    f"which equals the global rank only when the dp group spans the "
                    f"world."
                )
        self.local.setup()
        this_rank_log(
            logger,
            logging.INFO,
            "pi0.5 DP scheduler ready (dp=%d, per_rank_B=%d).",
            self.dp_size,
            self.per_rank_B,
        )

    @torch.no_grad()
    def step(self, request: PI05Request) -> torch.Tensor:
        """Scatter the batch across dp, run ws1 per rank, gather actions to rank 0.

        Returns ``(real_B, chunk, action_dim)`` on rank 0 (the Router) and the
        full ``(dp_size*per_rank_B, chunk, action_dim)`` on other ranks (ignored
        by the launcher, which only persists rank 0's output).
        """
        if self.dp_size == 1:
            return self.local.step(request)

        real_B = int(request.pixel_values.shape[0]) if self.dp_rank == 0 else 0
        if self.dp_rank == 0:
            # Fatal misconfiguration guard. Only rank 0 knows real_B, so this
            # raises on rank 0 alone; peers are already blocked in ``recv``
            # inside ``_scatter`` and hang until the launcher (torchrun) reaps
            # them. That is acceptable for a caller-side programming error — a
            # symmetric raise would cost a broadcast on every (valid) step.
            capacity = self.dp_size * self.per_rank_B
            if real_B > capacity:
                raise ValueError(
                    f"PI05WNScheduler: batch {real_B} exceeds DP capacity "
                    f"dp_size*per_rank_B={capacity}."
                )
        shard = self._scatter(request)
        local_actions = self.local.step(shard)  # (per_rank_B, chunk, action_dim)
        gathered = P.all_gather(local_actions, axis="dp", dim=0)
        if self.dp_rank == 0:
            return gathered[:real_B].clone()
        return gathered

    def _scatter(self, request: PI05Request) -> PI05Request:
        """Rank 0 splits the padded batch and sends each shard; others receive.

        Fields moved: ``pixel_values``/``noise`` (model dtype) and
        ``input_ids``/``lang_lens`` (int64). Every rank ends with exactly
        ``per_rank_B`` rows. When ``request.noise is None`` the Router samples one
        noise tensor and scatters it, so all shards share a single noise source.
        """
        R, N, pr = self.dp_rank, self.dp_size, self.per_rank_B
        dev, dt, cfg = self.device, self.dtype, self.cfg
        chunk, act = cfg.chunk_size, cfg.max_action_dim
        px_shape = (
            pr,
            self.num_images,
            cfg.vision.num_channels,
            cfg.vision.image_size,
            cfg.vision.image_size,
        )
        ids_shape = (pr, cfg.tokenizer_max_length)
        lens_shape = (pr,)
        noise_shape = (pr, chunk, act)

        if R == 0:
            total = N * pr
            px = _pad_rows(request.pixel_values.to(dev, dt), total)
            ids = _pad_rows(request.input_ids.to(dev, torch.int64), total)
            lens = _pad_rows(request.lang_lens.to(dev, torch.int64), total)
            if request.noise is not None:
                noise_src = request.noise.to(dev, dt)
            else:
                noise_src = torch.randn(
                    request.pixel_values.shape[0], chunk, act, device=dev, dtype=dt
                )
            noise = _pad_rows(noise_src, total)

            def shard(t: torch.Tensor, r: int) -> torch.Tensor:
                return t[r * pr : (r + 1) * pr].contiguous()

            for r in range(1, N):
                P.send(shard(px, r), axis="dp", dst=r)
                P.send(shard(ids, r), axis="dp", dst=r)
                P.send(shard(lens, r), axis="dp", dst=r)
                P.send(shard(noise, r), axis="dp", dst=r)
            return PI05Request(
                pixel_values=shard(px, 0),
                input_ids=shard(ids, 0),
                lang_lens=shard(lens, 0),
                noise=shard(noise, 0),
            )

        px = P.recv(px_shape, dt, axis="dp", src=0, device=dev)
        ids = P.recv(ids_shape, torch.int64, axis="dp", src=0, device=dev)
        lens = P.recv(lens_shape, torch.int64, axis="dp", src=0, device=dev)
        noise = P.recv(noise_shape, dt, axis="dp", src=0, device=dev)
        return PI05Request(pixel_values=px, input_ids=ids, lang_lens=lens, noise=noise)

    def close(self) -> None:
        self.local.close()


__all__ = ["PI05WNScheduler"]
