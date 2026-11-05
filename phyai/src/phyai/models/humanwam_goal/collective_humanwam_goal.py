"""Runner-owned CUDA IPC gathers for serialized HumanWAM-Goal inference."""

import os
import ctypes as C
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
from phyai_kernel.cuda.jit.humanwam_goal_gather import humanwam_goal_gather_library

from phyai.utils import get_logger

logger = get_logger(__name__)


class HumanWAMGoalGather:
    """One same-node, fully connected 2/4-GPU inference replica."""

    def __init__(self, *, max_bytes=64 << 20, blocks=96):
        if type(max_bytes) is not int or not 0 < max_bytes <= 1 << 30 or max_bytes % 16:
            raise ValueError(
                "gather workspace bytes must be a positive 16-byte multiple, at most 1 GiB"
            )
        if type(blocks) is not int or not 1 <= blocks <= 128:
            raise ValueError("gather block count must be in [1,128]")
        self.rank, self.size = dist.get_rank(), dist.get_world_size()
        if self.size not in (2, 4):
            raise ValueError("The fixed gather requires a dedicated 2/4-rank group")
        self.device, self.dtype = torch.cuda.current_device(), torch.bfloat16
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
                raise ValueError("CUDA IPC gather requires all TP ranks on one node")
            self.library = humanwam_goal_gather_library()
            pointer, handle = C.c_void_p(), C.create_string_buffer(64)
            self._check(
                self.library.alloc_shared(
                    self.library.signal_size(),
                    C.byref(pointer),
                    handle,
                )
            )
            self.local.append(pointer.value)
            handles.append((handle.raw, 0))
            # Torch owns this allocation. Fused Q/K preparation writes directly into it;
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
            logger.exception("HumanWAM-Goal gather initialization failed")
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
                f"Fixed gather initialization failed: {[p[2] for p in peers]}"
            )
        if len({p[0] for p in peers}) != 1 or len({p[1] for p in peers}) != self.size:
            raise ValueError("Fixed gather requires distinct GPUs on the same host")

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
                self.library.create_gather(
                    (C.c_void_p * self.size)(*signals),
                    (C.c_void_p * self.size)(*inputs),
                    self.max_bytes,
                    self.rank,
                    self.size,
                    C.byref(self.pointer),
                )
            )
        except Exception as exc:
            logger.exception("HumanWAM-Goal gather initialization failed")
            error = repr(exc)
        errors = self._gather(error)
        if any(errors):
            raise RuntimeError(f"Fixed gather peer setup failed: {errors}")

    def _validate(self, value: torch.Tensor) -> None:
        if self.closed:
            raise RuntimeError("HumanWAM-Goal gather workspace is closed")
        if (
            not value.is_cuda
            or value.device.index != self.device
            or torch.cuda.current_device() != self.device
            or value.dtype != self.dtype
            or not value.is_contiguous()
            or value.numel() == 0
            or value.data_ptr() % 16
            or value.nbytes % 16
            or value.nbytes > self.max_bytes
        ):
            raise ValueError(
                "Gather requires aligned contiguous BF16 tensors within the workspace bound"
            )

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        self._validate(value)
        columns = value.shape[-1]
        if columns % 8:
            raise ValueError("Gather columns must be a multiple of eight")
        output = value.new_empty((*value.shape[:-1], columns * self.size))
        self._check(
            self.library.gather_last_dimension(
                self.pointer,
                value.data_ptr(),
                output.data_ptr(),
                value.numel() // columns,
                columns,
                torch.cuda.current_stream().cuda_stream,
                self.blocks,
            )
        )
        return output

    def gather_rows(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 3 or value.shape[0] != 1:
            raise ValueError("Token gather requires a single batch")
        output = self.gather(value.contiguous().reshape(1, -1))
        return output.reshape(1, value.shape[1] * self.size, *value.shape[2:])

    def kv_workspace(self, part: int) -> torch.Tensor:
        if self.closed:
            raise RuntimeError("HumanWAM-Goal gather workspace is closed")
        if part <= 0 or part * 2048 * 2 > self.max_bytes:
            raise ValueError("Local KV exceeds the workspace bound")
        return self.input[: part * 2048].view(part, 2, 8, 128)

    def gather_kv(
        self,
        text_key: torch.Tensor,
        text_value: torch.Tensor,
        *,
        part: int,
        tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.kv_workspace(part)
        for value in (text_key, text_value):
            self._validate(value)
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[2:] != (8, 128):
                raise ValueError("Text KV must have shape [1,text,8,128]")
        if text_key.shape != text_value.shape or not 0 < tokens <= part * self.size:
            raise ValueError("Incompatible text KV or token partition")
        key = text_key.new_empty((1, text_key.shape[1] + tokens, 8, 128))
        value = torch.empty_like(key)
        self._check(
            self.library.gather_sp_kv(
                self.pointer,
                text_key.data_ptr(),
                text_value.data_ptr(),
                key.data_ptr(),
                value.data_ptr(),
                part,
                tokens,
                text_key.shape[1],
                torch.cuda.current_stream().cuda_stream,
                self.blocks,
            )
        )
        return key, value

    def close(self):
        if self.closed:
            return
        # A peer may still read this rank's input after its own launch.
        # Finish all ranks before destroying the shared buffers or communicator.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.cpu)
        if self.pointer.value:
            self._check(self.library.destroy_gather(self.pointer))
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
