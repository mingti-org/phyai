"""Pure-Python ctypes binding for libnccl.so.

CUDA-only and trimmed to the primitives we use.

Why a ctypes wrapper rather than ``torch.distributed``? PyTorch's
ProcessGroupNCCL adds host-side machinery (Work, watchdog, internal
streams) that complicates CUDA-graph capture. A thin direct binding
gives us explicit stream control and skips the watchdog.

Override the library path with ``PHYAI_NCCL_SO_PATH`` if needed.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed import ReduceOp

from phyai.utils import all_ranks_log, this_rank_log

logger = logging.getLogger(__name__)


def find_nccl_library() -> str:
    so_file = os.environ.get("PHYAI_NCCL_SO_PATH")
    if so_file:
        this_rank_log(logger, logging.INFO, "Using PHYAI_NCCL_SO_PATH=%s", so_file)
        return so_file
    if torch.version.cuda is not None:
        return "libnccl.so.2"
    raise ValueError(
        "PyNCCL backend requires CUDA. "
        "Set PHYAI_NCCL_SO_PATH to point to your NCCL library."
    )


# ---- C type aliases mirroring nccl.h ------------------------------------
ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p
ncclDataType_t = ctypes.c_int
ncclRedOp_t = ctypes.c_int


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclFloat32 = 7
    ncclFloat64 = 8
    ncclBfloat16 = 9

    _MAP: dict[torch.dtype, int] = {}

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if not cls._MAP:
            cls._MAP = {
                torch.int8: cls.ncclInt8,
                torch.uint8: cls.ncclUint8,
                torch.int32: cls.ncclInt32,
                torch.int64: cls.ncclInt64,
                torch.float16: cls.ncclFloat16,
                torch.float32: cls.ncclFloat32,
                torch.float64: cls.ncclFloat64,
                torch.bfloat16: cls.ncclBfloat16,
            }
        try:
            return cls._MAP[dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype for NCCL: {dtype}") from e


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4

    _MAP: dict[ReduceOp, int] = {}

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if not cls._MAP:
            cls._MAP = {
                ReduceOp.SUM: cls.ncclSum,
                ReduceOp.PRODUCT: cls.ncclProd,
                ReduceOp.MAX: cls.ncclMax,
                ReduceOp.MIN: cls.ncclMin,
                ReduceOp.AVG: cls.ncclAvg,
            }
        try:
            return cls._MAP[op]
        except KeyError as e:
            raise ValueError(f"Unsupported ReduceOp for NCCL: {op}") from e


@dataclass
class _Function:
    name: str
    restype: Any
    argtypes: list[Any]


class NCCLLibrary:
    """ctypes binding to libnccl.so.

    Cached per-path so loading is idempotent.
    """

    _exported: list[_Function] = [
        _Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        _Function("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        _Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        _Function(
            "ncclCommInitRank",
            ncclResult_t,
            [
                ctypes.POINTER(ncclComm_t),
                ctypes.c_int,
                ncclUniqueId,
                ctypes.c_int,
            ],
        ),
        _Function(
            "ncclAllReduce",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        _Function(
            "ncclAllGather",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        _Function(
            "ncclReduceScatter",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        _Function(
            "ncclSend",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        _Function(
            "ncclRecv",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        _Function(
            "ncclBroadcast",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        _Function("ncclCommDestroy", ncclResult_t, [ncclComm_t]),
        _Function("ncclGroupStart", ncclResult_t, []),
        _Function("ncclGroupEnd", ncclResult_t, []),
    ]

    _path_lib_cache: dict[str, Any] = {}
    _path_funcs_cache: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None) -> None:
        so_file = so_file or find_nccl_library()
        try:
            if so_file not in NCCLLibrary._path_lib_cache:
                NCCLLibrary._path_lib_cache[so_file] = ctypes.CDLL(so_file)
            self.lib = NCCLLibrary._path_lib_cache[so_file]
        except Exception as e:
            all_ranks_log(
                logger,
                logging.ERROR,
                "Failed to load NCCL library from %s on platform %s. "
                "Set PHYAI_NCCL_SO_PATH to point to a valid libnccl.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in NCCLLibrary._path_funcs_cache:
            funcs: dict[str, Any] = {}
            for fn in NCCLLibrary._exported:
                f = getattr(self.lib, fn.name)
                f.restype = fn.restype
                f.argtypes = fn.argtypes
                funcs[fn.name] = f
            NCCLLibrary._path_funcs_cache[so_file] = funcs
        self._funcs = NCCLLibrary._path_funcs_cache[so_file]

    # ------------------------------------------------------------------
    # error handling
    # ------------------------------------------------------------------

    def ncclGetErrorString(self, result: int) -> str:
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: int) -> None:
        if result != 0:
            raise RuntimeError(f"NCCL error: {self.ncclGetErrorString(result)}")

    # ------------------------------------------------------------------
    # version + comm init
    # ------------------------------------------------------------------

    def ncclGetRawVersion(self) -> int:
        v = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(v)))
        return v.value

    def ncclGetVersion(self) -> str:
        s = str(self.ncclGetRawVersion())
        major = s[0].lstrip("0")
        minor = s[1:3].lstrip("0")
        patch = s[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        uid = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](ctypes.byref(uid)))
        return uid

    def ncclCommInitRank(
        self,
        world_size: int,
        unique_id: ncclUniqueId,
        rank: int,
    ) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(
            self._funcs["ncclCommInitRank"](
                ctypes.byref(comm),
                world_size,
                unique_id,
                rank,
            )
        )
        return comm

    def ncclCommDestroy(self, comm: ncclComm_t) -> None:
        self.NCCL_CHECK(self._funcs["ncclCommDestroy"](comm))

    # ------------------------------------------------------------------
    # collectives
    # ------------------------------------------------------------------

    def ncclAllReduce(
        self,
        sendbuff,
        recvbuff,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclAllReduce"](
                sendbuff,
                recvbuff,
                count,
                datatype,
                op,
                comm,
                stream,
            )
        )

    def ncclAllGather(
        self,
        sendbuff,
        recvbuff,
        count: int,
        datatype: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclAllGather"](
                sendbuff,
                recvbuff,
                count,
                datatype,
                comm,
                stream,
            )
        )

    def ncclReduceScatter(
        self,
        sendbuff,
        recvbuff,
        count: int,
        datatype: int,
        op: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclReduceScatter"](
                sendbuff,
                recvbuff,
                count,
                datatype,
                op,
                comm,
                stream,
            )
        )

    def ncclBroadcast(
        self,
        sendbuff,
        recvbuff,
        count: int,
        datatype: int,
        root: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclBroadcast"](
                sendbuff,
                recvbuff,
                count,
                datatype,
                root,
                comm,
                stream,
            )
        )

    def ncclSend(
        self,
        sendbuff,
        count: int,
        datatype: int,
        dest: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclSend"](
                sendbuff,
                count,
                datatype,
                dest,
                comm,
                stream,
            )
        )

    def ncclRecv(
        self,
        recvbuff,
        count: int,
        datatype: int,
        src: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["ncclRecv"](
                recvbuff,
                count,
                datatype,
                src,
                comm,
                stream,
            )
        )

    def ncclGroupStart(self) -> None:
        self.NCCL_CHECK(self._funcs["ncclGroupStart"]())

    def ncclGroupEnd(self) -> None:
        self.NCCL_CHECK(self._funcs["ncclGroupEnd"]())


__all__ = [
    "NCCLLibrary",
    "ncclDataTypeEnum",
    "ncclRedOpTypeEnum",
    "ncclUniqueId",
    "ncclComm_t",
    "cudaStream_t",
    "buffer_type",
]
