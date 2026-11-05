"""HumanWAM-Goal's fixed-workspace SUM registered with PhyAI parallel dispatch."""

import os
import ctypes as C
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
from phyai_kernel.cuda.jit.humanwam_goal_sum import humanwam_goal_sum_library

from phyai import parallel as P
from phyai.utils import get_logger

logger = get_logger(__name__)


class FixedWorkspaceSum:
    """One same-node, fully connected 2/4-GPU inference replica."""

    def __init__(self, *, dtype, max_bytes=64 << 20, blocks=96):
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("The fixed SUM supports native FP32/BF16 tensors only")
        if type(max_bytes) is not int or not 0 < max_bytes <= 1 << 30 or max_bytes % 16:
            raise ValueError(
                "SUM workspace bytes must be a positive 16-byte multiple, at most 1 GiB"
            )
        if type(blocks) is not int or not 1 <= blocks <= 128:
            raise ValueError("SUM block count must be in [1,128]")
        self.rank, self.size = dist.get_rank(), dist.get_world_size()
        if self.size not in (2, 4):
            raise ValueError("The fixed SUM requires a dedicated 2/4-rank group")
        self.device, self.dtype = torch.cuda.current_device(), dtype
        self.max_bytes, self.blocks = max_bytes, blocks
        self.local, self.opened = [], []
        self.input = None
        self.pointer, self.library, self.closed = C.c_void_p(), None, False
        self.cpu = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _gather(self, value):
        values = [None] * self.size
        dist.all_gather_object(values, value, group=self.cpu)
        return values

    def _check(self, result):
        if result:
            raise RuntimeError(self.library.error_message().decode())

    def _initialize(self):
        error, handles = None, []
        try:
            if int(os.environ.get("LOCAL_WORLD_SIZE", self.size)) != self.size:
                raise ValueError("CUDA IPC SUM requires all TP ranks on one node")
            self.library = humanwam_goal_sum_library()
            pointer, handle = C.c_void_p(), C.create_string_buffer(64)
            self._check(
                self.library.alloc_shared(
                    self.library.signal_size() + self.max_bytes,
                    C.byref(pointer),
                    handle,
                )
            )
            self.local.append(pointer.value)
            handles.append((handle.raw, 0))
            # Torch owns this allocation. Row GEMMs can write directly into it;
            # export its allocator base and retain the suballocation offset.
            self.input = torch.empty(
                self.max_bytes // torch.empty((), dtype=self.dtype).element_size(),
                dtype=self.dtype,
                device=self.device,
            )
            offset, handle = C.c_size_t(), C.create_string_buffer(64)
            self._check(
                self.library.export_shared(
                    self.input.data_ptr(), handle, C.byref(offset)
                )
            )
            handles.append((handle.raw, offset.value))
        except Exception as exc:
            logger.exception("HumanWAM-Goal SUM initialization failed")
            error = repr(exc)
        peers = self._gather(
            (
                socket.gethostname(),
                str(torch.cuda.get_device_properties(self.device).uuid),
                error,
                handles,
            )
        )
        if any(p[2] for p in peers):
            raise RuntimeError(
                f"Fixed SUM initialization failed: {[p[2] for p in peers]}"
            )
        if len({p[0] for p in peers}) != 1 or len({p[1] for p in peers}) != self.size:
            raise ValueError("Fixed SUM requires distinct GPUs on the same host")

        error = None
        signals, inputs = [], []
        try:
            for rank, peer in enumerate(peers):
                for index, destination in enumerate((signals, inputs)):
                    if rank == self.rank:
                        pointer = self.local[0] if index == 0 else self.input.data_ptr()
                    else:
                        value = C.c_void_p()
                        raw, offset = peer[3][index]
                        handle = C.create_string_buffer(raw, 64)
                        self._check(self.library.open_shared(handle, C.byref(value)))
                        self.opened.append(value.value)
                        pointer = value.value + offset
                    destination.append(pointer)
            self._check(
                self.library.create_sum(
                    (C.c_void_p * self.size)(*signals),
                    (C.c_void_p * self.size)(*inputs),
                    self.max_bytes,
                    self.rank,
                    self.size,
                    C.byref(self.pointer),
                )
            )
        except Exception as exc:
            logger.exception("HumanWAM-Goal SUM initialization failed")
            error = repr(exc)
        errors = self._gather(error)
        if any(errors):
            raise RuntimeError(f"Fixed SUM peer setup failed: {errors}")

    def sum(self, value, output):
        if self.closed:
            raise RuntimeError("Fixed SUM workspace is closed")
        if (
            not value.is_cuda
            or value.device.index != self.device
            or torch.cuda.current_device() != self.device
            or value.dtype != self.dtype
            or not value.is_contiguous()
            or value.numel() == 0
            or value.data_ptr() % 16
            or value.numel() * value.element_size() % 16
            or value.numel() * value.element_size() > self.max_bytes
        ):
            raise ValueError(
                "Fixed SUM requires aligned contiguous native-dtype tensors within its workspace bound"
            )
        if (
            output.shape != value.shape
            or output.dtype != value.dtype
            or output.device != value.device
            or not output.is_contiguous()
        ):
            raise ValueError("SUM output must match its input")
        self._check(
            self.library.reduce_sum(
                self.pointer,
                value.data_ptr(),
                output.data_ptr(),
                value.numel(),
                torch.cuda.current_stream().cuda_stream,
                self.blocks,
                int(self.dtype == torch.bfloat16),
            )
        )
        return output

    def close(self):
        if self.closed:
            return
        # A peer may still read this rank's input/scratch after its own launch.
        # Finish all ranks before destroying the shared buffers or communicator.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu)
        if self.pointer.value:
            self._check(self.library.destroy_sum(self.pointer))
            self.pointer = C.c_void_p()
        for pointer in self.opened:
            self._check(self.library.close_shared(pointer))
        dist.barrier(group=self.cpu)
        for pointer in self.local:
            self._check(self.library.free_shared(pointer))
        self.input = None
        self.local.clear()
        self.opened.clear()
        self.closed = True
        dist.destroy_process_group(self.cpu)


class HumanWAMGoalSumBackend:
    name = "humanwam_goal-sum"

    def __init__(self):
        self.owner = FixedWorkspaceSum(dtype=torch.bfloat16)
        self.executions = 0

    def supports_capture(self):
        return True

    def can_handle(
        self,
        *,
        op,
        nbytes,
        dtype,
        world_size,
        topology,
        group=None,
        mesh_name=None,
        **kwargs,
    ):
        return (
            op == P.Op.ALL_REDUCE
            and mesh_name == "model"
            and group in {"dense_tp", "attention_tp"}
            and dtype == torch.bfloat16
            and world_size == self.owner.size
            and 0 < nbytes <= self.owner.max_bytes
            and nbytes % 16 == 0
            and topology.is_single_node
            and topology.is_full_nvlink
        )

    def execute(self, *, input, output, reduce_op, **kwargs):
        if reduce_op != dist.ReduceOp.SUM:
            raise ValueError("HumanWAM-Goal supports SUM collectives only")
        self.executions += 1
        return self.owner.sum(input, output)

    def close(self):
        self.owner.close()


def install_humanwam_goal_collective():
    mesh = P.default_mesh()
    if mesh.group_size("dense_tp") == 1:
        return None
    for group in ("dense_tp", "attention_tp"):
        topology = mesh.topology(group)
        if not topology.is_single_node or not topology.is_full_nvlink:
            raise ValueError(
                "HumanWAM-Goal TP requires a fully connected NVLink group on one node"
            )
    backend = HumanWAMGoalSumBackend()
    previous = P.get_dispatcher()
    previous.registry.register(backend)
    P.set_dispatcher(
        P.Dispatcher(previous.registry, P.ForcedPolicy(backend.name, previous.policy))
    )
    return backend
