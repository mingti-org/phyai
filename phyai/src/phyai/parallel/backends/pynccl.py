"""PyNCCLBackend — direct ctypes call to libnccl, bypassing
``torch.distributed`` host-side machinery.

Two layers:

* ``_PyNcclComm``: one NCCL communicator bound to a specific subgroup of
  ranks and a specific CUDA device. Built once via a CPU-side gloo group
  for unique_id bootstrap.
* ``PyNCCLBackend``: implements the ``Backend`` protocol; holds a dict of
  ``_PyNcclComm`` keyed by ``(mesh_name, axis)``, lazily initialised by
  :meth:`attach` during ``phyai.parallel.init``.

Why bypass ``torch.distributed`` even when it is capture-compatible?
Two practical reasons:

1. Kernel launches go directly to the **caller's current stream**, so
   overlap via ``on_stream`` is straightforward.
2. No PG watchdog thread / event polling overhead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from phyai.parallel.backend import Backend, Op, Topology
from phyai.parallel.state import Mode
from phyai.parallel.backends.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from phyai.utils import all_ranks_log

if TYPE_CHECKING:
    from phyai.parallel.mesh import Mesh

logger = logging.getLogger(__name__)


class _PyNcclComm:
    """One NCCL communicator bound to a (subgroup, device) pair."""

    def __init__(
        self,
        device_group: ProcessGroup,
        cpu_group: ProcessGroup,
        device: torch.device,
        nccl: NCCLLibrary,
    ) -> None:
        self.rank = dist.get_rank(device_group)
        self.world_size = dist.get_world_size(device_group)
        self.device = device
        self.nccl = nccl

        if self.world_size == 1:
            self.comm: ncclComm_t | None = None
            return

        # rank 0 of the subgroup creates the unique id, broadcasts via cpu_group
        if self.rank == 0:
            uid = nccl.ncclGetUniqueId()
        else:
            uid = ncclUniqueId()
        tensor = torch.ByteTensor(list(uid.internal))
        ranks = dist.get_process_group_ranks(cpu_group)
        dist.broadcast(tensor, src=ranks[0], group=cpu_group)
        for i, b in enumerate(tensor.tolist()):
            uid.internal[i] = b

        with torch.cuda.device(device):
            self.comm = nccl.ncclCommInitRank(self.world_size, uid, self.rank)
            # Warmup AR on a side stream — pulls all NCCL lazy init out of
            # the way so cuda graph capture doesn't see one-time side effects.
            warmup = torch.cuda.Stream()
            with torch.cuda.stream(warmup):
                data = torch.zeros(1, device=device)
                self._all_reduce(data, ReduceOp.SUM, stream=warmup)
            warmup.synchronize()
            del data

    def _stream(self) -> torch.cuda.Stream:
        return torch.cuda.current_stream()

    # ------------------------------------------------------------------
    # primitive bindings (all launch on caller's current stream)
    # ------------------------------------------------------------------

    def _all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp,
        *,
        stream: torch.cuda.Stream | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s = stream or self._stream()
        recv = out if out is not None else tensor
        self.nccl.ncclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(recv.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return recv

    def _all_gather(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        s = self._stream()
        self.nccl.ncclAllGather(
            buffer_type(input.data_ptr()),
            buffer_type(output.data_ptr()),
            input.numel(),
            ncclDataTypeEnum.from_torch(input.dtype),
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return output

    def _reduce_scatter(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        op: ReduceOp,
    ) -> torch.Tensor:
        s = self._stream()
        self.nccl.ncclReduceScatter(
            buffer_type(input.data_ptr()),
            buffer_type(output.data_ptr()),
            output.numel(),
            ncclDataTypeEnum.from_torch(input.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return output

    def _broadcast(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        src: int,
    ) -> torch.Tensor:
        s = self._stream()
        send_ptr = buffer_type(input.data_ptr()) if src == self.rank else buffer_type()
        self.nccl.ncclBroadcast(
            send_ptr,
            buffer_type(output.data_ptr()),
            output.numel(),
            ncclDataTypeEnum.from_torch(output.dtype),
            src,
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return output

    def _send(self, tensor: torch.Tensor, dst: int) -> None:
        s = self._stream()
        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            cudaStream_t(s.cuda_stream),
        )

    def _recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        s = self._stream()
        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(s.cuda_stream),
        )
        return tensor

    def destroy(self) -> None:
        if self.comm is not None:
            try:
                self.nccl.ncclCommDestroy(self.comm)
            except Exception as e:
                all_ranks_log(logger, logging.WARNING, "ncclCommDestroy failed: %s", e)
            self.comm = None


class PyNCCLBackend:
    """Backend that drives NCCL via a direct ctypes binding.

    Capture-safe; in eager mode it returns False from ``can_handle`` so the
    Dispatcher falls through to ``TorchDistBackend`` (which has the better
    watchdog story for production eager paths).
    """

    name = "pynccl"

    _OPS = {
        Op.ALL_REDUCE,
        Op.ALL_GATHER,
        Op.REDUCE_SCATTER,
        Op.BROADCAST,
        Op.SEND,
        Op.RECV,
    }

    def __init__(
        self,
        *,
        library_path: str | None = None,
        prefer_in_eager: bool = False,
    ) -> None:
        self._library_path = library_path
        self._nccl: NCCLLibrary | None = None
        self._comms: dict[tuple[str, str], _PyNcclComm] = {}
        self._prefer_in_eager = prefer_in_eager
        self._handlers: dict[Op, callable] = {
            Op.ALL_REDUCE: self._h_all_reduce,
            Op.ALL_GATHER: self._h_all_gather,
            Op.REDUCE_SCATTER: self._h_reduce_scatter,
            Op.BROADCAST: self._h_broadcast,
            Op.SEND: self._h_send,
            Op.RECV: self._h_recv,
        }

    # --- Backend protocol --------------------------------------------------

    def can_handle(
        self,
        *,
        op: Op,
        mode: Mode,
        nbytes: int,
        dtype: torch.dtype,
        world_size: int,
        topology: Topology,
        **extra: object,
    ) -> bool:
        if op not in self._OPS:
            return False
        if mode == Mode.EAGER and not self._prefer_in_eager:
            return False
        if world_size <= 1:
            return False
        return True

    def supports_capture(self) -> bool:
        return True

    def execute(
        self,
        *,
        op: Op,
        pg: ProcessGroup,
        **kwargs,
    ) -> torch.Tensor | None:
        handler = self._handlers.get(op)
        if handler is None:
            raise NotImplementedError(f"PyNCCLBackend.execute: op={op}")
        comm = self._comm_for(
            pg,
            mesh_name=kwargs["_mesh_name"],
            axis=kwargs["_axis"],
            device=kwargs["_device"],
        )
        return handler(comm, **kwargs)

    # --- per-op handlers (take a `comm` first arg) ------------------------

    def _h_all_reduce(self, comm, *, input, output, reduce_op, **_):
        return comm._all_reduce(input, reduce_op, out=output)

    def _h_all_gather(self, comm, *, input, output, **_):
        # ncclAllGather is dim-0; the L4 op layer requests dim=0 here
        # because GlooBackend / NcclBackend handle dim != 0 via reshape.
        return comm._all_gather(input, output)

    def _h_reduce_scatter(self, comm, *, input, output, reduce_op, **_):
        return comm._reduce_scatter(input, output, reduce_op)

    def _h_broadcast(self, comm, *, input, output, src, **_):
        return comm._broadcast(input, output, src)

    def _h_send(self, comm, *, input, dst, **_):
        comm._send(input, dst)
        return None

    def _h_recv(self, comm, *, output, src, **_):
        return comm._recv(output, src)

    # --- lifecycle --------------------------------------------------------

    def attach(self, mesh: "Mesh", axes: list[str], *, device: torch.device) -> None:
        """Build per-axis comms eagerly for every axis we expect to use.

        Eager construction sidesteps the lazy-init-during-capture trap
        that PyTorch's ProcessGroupNCCL also exhibits — pre-warming
        keeps NCCL's one-time side effects out of the recorded graph.

        Size-1 axes are skipped. Every collective short-circuits at
        ``axis_size <= 1`` (see :mod:`phyai.parallel.ops`), so such an axis
        never needs a comm. Skipping it also avoids a ``dist.new_group``
        deadlock: ``_build_cpu_group_for`` would have each rank create a
        *single-member* group containing only itself (e.g. rank 0 ->
        ``new_group([0])``, rank 1 -> ``new_group([1])``), but ``new_group``
        is a global collective that requires every rank to request the same
        groups in the same order — disjoint singletons hang. The 5-axis
        engine mesh (``dp/ep/sp/cp`` at size 1, only ``tp`` > 1) hit this.
        The skip is consistent across ranks because ``axis_size`` is the
        mesh dimension width, identical on every rank.
        """
        if self._nccl is None:
            self._nccl = NCCLLibrary(self._library_path)
        for axis in axes:
            if mesh.axis_size(axis) <= 1:
                continue
            key = (mesh.name, axis)
            if key in self._comms:
                continue
            device_group = mesh.axis_group(axis)
            cpu_group = _build_cpu_group_for(device_group)
            self._comms[key] = _PyNcclComm(
                device_group=device_group,
                cpu_group=cpu_group,
                device=device,
                nccl=self._nccl,
            )

    def _comm_for(
        self,
        pg: ProcessGroup,
        *,
        mesh_name: str,
        axis: str,
        device: torch.device,
    ) -> _PyNcclComm:
        key = (mesh_name, axis)
        comm = self._comms.get(key)
        if comm is None:
            raise RuntimeError(
                f"PyNCCLBackend.attach() was not called for {key}. "
                "Pass `pynccl_axes=[...]` to phyai.parallel.init()."
            )
        return comm


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

_cpu_group_cache: dict[tuple[int, ...], ProcessGroup] = {}


def _build_cpu_group_for(device_group: ProcessGroup) -> ProcessGroup:
    """Return (or create) a gloo group with the same ranks as
    ``device_group``. Used solely for unique_id bootstrap.

    ``use_local_synchronization=True`` is essential here: by default
    ``new_group`` ends with a *world*-wide barrier, so every rank must issue
    the identical sequence of ``new_group`` calls or it deadlocks. With more
    than one parallel axis the per-axis subgroups are DISJOINT and differ by
    rank (e.g. the cfg axis groups ranks ``[0,4]`` on rank 0 but ``[1,5]`` on
    rank 1) — those are different ``new_group`` calls that would never rendezvous
    under the global barrier. Local synchronization makes each subgroup barrier
    only within its own members, so disjoint subgroups are created concurrently
    (non-member ranks don't join). ``attach`` is still called collectively, so
    every rank creates exactly its own subgroup per axis.
    """
    ranks = tuple(dist.get_process_group_ranks(device_group))
    if ranks in _cpu_group_cache:
        return _cpu_group_cache[ranks]
    pg = dist.new_group(
        ranks=list(ranks), backend="gloo", use_local_synchronization=True
    )
    _cpu_group_cache[ranks] = pg
    return pg
