"""Shared weight-loader factories — sharding math lives here, once.

Each function returns a :data:`WeightLoader` closure with signature
``(param, loaded, shard_id) -> None``. Layers attach the closure to
their ``nn.Parameter`` so :func:`phyai.weights.load_pretrained` can
dispatch generically. The math runs at load time only — there is no
forward-time cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from phyai.parallel.mesh import Mesh


WeightLoader = Callable[[torch.nn.Parameter, torch.Tensor, "int | str | None"], None]


def _copy_exact(destination: torch.Tensor, source: torch.Tensor) -> None:
    if destination.shape != source.shape:
        raise ValueError(
            f"Loaded weight shape {tuple(source.shape)} does not match "
            f"destination shape {tuple(destination.shape)}"
        )
    destination.copy_(source)


def replicated() -> WeightLoader:
    """Full-tensor copy. Handles 0-D HF scalars expanding into 1-element params."""

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, _shard_id=None) -> None:
        if loaded.dim() == 0 and param.numel() == 1:
            param.data.fill_(loaded.item())
            return
        _copy_exact(param.data, loaded)

    return load


def sharded(
    *, dim: int, group: str = "dense_tp", mesh: Mesh, replication_factor: int = 1
) -> WeightLoader:
    """Single-group TP / EP shard along ``dim``.

    ``replication_factor`` is the GQA case: that many ranks in ``group``
    each read the same shard slot.
    """

    if (
        not isinstance(replication_factor, int)
        or isinstance(replication_factor, bool)
        or replication_factor < 1
        or mesh.group_size(group) % replication_factor
    ):
        raise ValueError(
            "replication_factor must be a positive divisor of the group size."
        )

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, _shard_id=None) -> None:
        rank = mesh.group_rank(group) // replication_factor
        world = mesh.group_size(group) // replication_factor
        if loaded.shape[dim] % world:
            raise ValueError(
                f"weight dimension {loaded.shape[dim]} is not divisible by {world} shards."
            )
        size = loaded.shape[dim] // world
        _copy_exact(param.data, loaded.narrow(dim, rank * size, size))

    return load


@dataclass(frozen=True)
class _Leg:
    """One leg of a fused-param load.

    ``offset`` / ``size`` are the **post-shard local** position and size
    in the destination's ``fuse_dim``. ``dim`` is the source's TP-shard
    dim (almost always ``0`` — column-parallel fuse). ``replication_factor`` is
    the GQA replication factor for K/V legs.
    """

    offset: int
    size: int
    dim: int = 0
    group: str = "dense_tp"
    replication_factor: int = 1


def fused(*, fuse_dim: int, legs: dict, mesh: Mesh) -> WeightLoader:
    """Multi-source fused-param loader.

    ``legs`` maps ``shard_id`` -> :class:`_Leg`. The loader looks up the
    leg by ``shard_id`` (received from the param's ``hf_keys`` entry),
    TP-shards the source, and writes into the destination's ``fuse_dim``
    slot at ``[offset, offset+size)``.

    Covers fused QKV with GQA (Q has ``replication_factor=1``, K/V have
    ``replication_factor=num_kv_replicas``) and fused gate/up.
    """

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, shard_id) -> None:
        leg = legs[shard_id]
        rank = mesh.group_rank(leg.group) // leg.replication_factor
        src = loaded.narrow(leg.dim, rank * leg.size, leg.size)
        _copy_exact(param.data.narrow(fuse_dim, leg.offset, leg.size), src)

    return load


def weight_norm_fold(*, eps: float = 1e-12) -> WeightLoader:
    """Fold a legacy ``weight_norm`` (``weight_g`` / ``weight_v``) pair into a dense weight.

    ``torch.nn.utils.weight_norm`` (now deprecated) reparametrises a weight as a
    magnitude ``g`` and direction ``v``; the forward weight is ``g * v / ‖v‖`` with
    the norm taken over every dim except ``0`` (the ``dim=0`` default — correct for
    both ``Conv`` ``(out, in, *k)`` and ``ConvTranspose`` ``(in, out, *k)`` layouts).
    This loader caches the two source tensors (``shard_id`` ``"g"`` / ``"v"``, either
    arrival order) and, once both are in, writes the dense forward weight. So a layer
    can carry a single dense ``weight`` and never run ``weight_norm`` at inference.

    A fresh closure (with its own per-parameter cache) is returned per call — attach
    one to each parameter, not a shared instance.
    """

    cache: dict[str, torch.Tensor] = {}

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, shard_id) -> None:
        if shard_id not in ("g", "v"):
            raise ValueError(
                f"weight_norm_fold expects shard_id 'g' or 'v', got {shard_id!r}"
            )
        cache[shard_id] = loaded.to(torch.float32)
        if "g" in cache and "v" in cache:
            g = cache.pop("g")
            v = cache.pop("v")
            dims = tuple(range(1, v.dim()))  # all dims except 0 (weight_norm dim=0)
            norm = v.norm(dim=dims, keepdim=True).clamp_min(eps)
            _copy_exact(param.data, (g * v / norm).to(param.dtype))

    return load


def vocab(*, group: str = "dense_tp", mesh: Mesh) -> WeightLoader:
    """Vocab-parallel embedding load with right-edge zero padding.

    The destination's ``shape[0]`` is the per-rank padded size. The HF
    tensor has ``V_real`` rows. Each rank loads the slice of real rows
    that fall in its range and zero-fills the trailing pad on the last
    rank. No full-padded source is materialised.
    """

    def load(param: torch.nn.Parameter, loaded: torch.Tensor, _shard_id=None) -> None:
        per_rank = param.shape[0]
        rank = mesh.group_rank(group)
        start = rank * per_rank
        v_real = loaded.shape[0]
        if start >= v_real:
            param.data.zero_()
            return
        n_real = min(start + per_rank, v_real) - start
        _copy_exact(param.data.narrow(0, 0, n_real), loaded.narrow(0, start, n_real))
        if n_real < per_rank:
            param.data.narrow(0, n_real, per_rank - n_real).zero_()

    return load


__all__ = [
    "WeightLoader",
    "_Leg",
    "fused",
    "replicated",
    "sharded",
    "vocab",
    "weight_norm_fold",
]
