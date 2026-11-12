"""Opt-in Thor BF16 GEMM routes with bounded, versioned weight storage."""

from __future__ import annotations

import ctypes
import functools
import json
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

PACK_BUDGET = 4608 * 1024 * 1024


@dataclass
class PackedWeight:
    owner: object
    signature: tuple
    value: torch.Tensor
    bytes: int
    done: object = None

    def wait(self):
        if self.done is not None:
            self.done.synchronize()


class PackedWeightCache:
    """No cached output tensors; invalidate storage on identity or version changes."""

    def __init__(self, max_bytes=PACK_BUDGET):
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        self.max_bytes = max_bytes
        self.bytes = 0
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    @staticmethod
    def signature(weight):
        try:
            version = weight._version
        except RuntimeError:
            return None
        return (
            version,
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.dtype,
            weight.device,
        )

    def _remove(self, key):
        entry = self.entries.pop(key, None)
        if entry is not None:
            entry.wait()
            self.bytes -= entry.bytes

    def get(self, weight, row_major):
        with self.lock:
            signature = self.signature(weight)
            if signature is None:
                return None
            key = (id(weight), bool(row_major))
            previous = self.entries.get(key)
            if previous is not None:
                if previous.owner() is weight and previous.signature == signature:
                    if previous.done is not None:
                        torch.cuda.current_stream(weight.device).wait_event(
                            previous.done
                        )
                    self.entries.move_to_end(key)
                    return previous
                self._remove(key)
            matrix = weight.t()
            expected = (matrix.shape[1], 1) if row_major else (1, matrix.shape[0])
            if tuple(matrix.stride()) == expected:
                return PackedWeight(weakref.ref(weight), signature, matrix, 0)
            size = weight.numel() * weight.element_size()
            if size > self.max_bytes:
                return None
            while self.entries and self.bytes + size > self.max_bytes:
                self._remove(next(iter(self.entries)))
            matrix = matrix.contiguous() if row_major else matrix.t().contiguous().t()
            entry = PackedWeight(weakref.ref(weight), signature, matrix, size)
            self.entries[key] = entry
            self.bytes += size
            return entry

    def invalidate(self, weights=None):
        with self.lock:
            selected = None if weights is None else {id(w) for w in weights}
            for key in list(self.entries):
                if selected is None or key[0] in selected:
                    self._remove(key)

    def record(self, entry, stream=None):
        if entry.value.is_cuda:
            event = torch.cuda.Event()
            event.record(stream or torch.cuda.current_stream(entry.value.device))
            entry.done = event


@functools.lru_cache(maxsize=1)
def get_pack_cache():
    return PackedWeightCache()


@functools.lru_cache(maxsize=1)
def get_routes():
    return weakref.WeakSet()


def invalidate_thor_gemm(transformer):
    weights = [m.weight for m in transformer.modules() if hasattr(m, "weight")]
    get_pack_cache().invalidate(weights)
    owner_ids = {id(w) for w in weights}
    for route in tuple(get_routes()):
        if route.owners & owner_ids:
            route.close()


@functools.lru_cache(maxsize=1)
def library():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("Thor GEMM requires a CUDA development toolkit")
    import phyai_kernel

    source = Path(phyai_kernel.__file__).resolve().parent / "cuda/csrc/thor_gemm.cc"
    path = load(
        name="phyai_thor_gemm",
        sources=[str(source)],
        with_cuda=True,
        is_python_module=False,
        extra_cflags=["-O3"],
        extra_ldflags=["-lcublasLt", "-lcudart"],
        verbose=False,
    )
    lib = ctypes.CDLL(path)
    pointer, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
    for name, args, result in (
        ("lt_make", [integer, integer, integer, integer, size, integer], pointer),
        ("lt_error", [pointer], integer),
        ("lt_configure", [pointer, ctypes.POINTER(integer), size], integer),
        (
            "lt_run",
            [pointer, integer, pointer, pointer, pointer, pointer, size, pointer],
            integer,
        ),
        ("lt_free", [pointer], None),
    ):
        function = getattr(lib, name)
        function.argtypes, function.restype = args, result
    return lib


@functools.lru_cache(maxsize=1)
def route_manifest():
    return json.loads(Path(__file__).with_name("thor_gemm_routes.json").read_text())


class ThorDenseGEMM:
    """Prepared selector implementation; unsupported runtime layouts use F.linear."""

    def __init__(self):
        self.plans = OrderedDict()
        self.workspaces = OrderedDict()
        self.owners = set()
        self.lock = threading.RLock()
        get_routes().add(self)

    def close(self):
        with self.lock:
            for _, (_, done) in list(self.workspaces.items()):
                done.synchronize()
            for plan, index in self.plans.values():
                library().lt_free(plan)
            self.plans.clear()
            self.workspaces.clear()
            self.owners.clear()

    def __call__(self, layer, x, bias):
        weight = layer.weight
        prefix = str(getattr(layer, "prefix", ""))
        if (
            bias is not None
            or not prefix.startswith("gen_layers.")
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or x.device != weight.device
            or x.ndim < 2
            or weight.ndim != 2
            or x.shape[-1] != weight.shape[1]
            or not x.is_contiguous()
            or x.numel() == 0
            or x.data_ptr() % 16
            or weight.data_ptr() % 16
            or torch.cuda.is_current_stream_capturing()
            or getattr(layer, "tp_size", 1) != 1
        ):
            return F.linear(x, weight, bias)
        category = prefix.split(".", 2)[-1].replace(".", "_")
        flat = x.view(-1, x.shape[-1])
        spec = route_manifest().get(str(flat.shape[0]), {}).get(category)
        if spec is None or spec["shape"] != [
            flat.shape[0],
            weight.shape[0],
            weight.shape[1],
        ]:
            return F.linear(x, weight, bias)
        stream = torch.cuda.current_stream(x.device)
        with self.lock, get_pack_cache().lock:
            packed = get_pack_cache().get(weight, spec["b_row"])
            if packed is None:
                return F.linear(x, weight, bias)
            self.owners.add(id(weight))
            try:
                if spec["method"] == "torch_mm":
                    output = torch.mm(flat, packed.value)
                else:
                    lib = library()
                    stream_key = (x.device, stream.cuda_stream)
                    if stream_key not in self.workspaces:
                        if len(self.workspaces) >= 2:
                            old, (_, done) = self.workspaces.popitem(last=False)
                            done.synchronize()
                            for key in list(self.plans):
                                if key[0] == old:
                                    lib.lt_free(self.plans.pop(key)[0])
                        self.workspaces[stream_key] = (
                            torch.empty(
                                64 * 1024 * 1024, device=x.device, dtype=torch.uint8
                            ),
                            torch.cuda.Event(),
                        )
                    workspace, done = self.workspaces[stream_key]
                    self.workspaces.move_to_end(stream_key)
                    key = (
                        stream_key,
                        tuple(spec["shape"]),
                        spec["b_row"],
                        tuple(spec["attrs"]),
                    )
                    if key not in self.plans:
                        m, n, k = spec["shape"]
                        plan = lib.lt_make(
                            m, n, k, int(spec["b_row"]), workspace.numel(), 1
                        )
                        if not plan or lib.lt_error(plan) != 0:
                            if plan:
                                lib.lt_free(plan)
                            return F.linear(x, weight, bias)
                        attrs = (ctypes.c_int * 9)(*spec["attrs"])
                        index = lib.lt_configure(plan, attrs, workspace.numel())
                        if index < 0:
                            lib.lt_free(plan)
                            return F.linear(x, weight, bias)
                        self.plans[key] = (plan, index)
                    plan, index = self.plans[key]
                    output = torch.empty(
                        (flat.shape[0], weight.shape[0]), device=x.device, dtype=x.dtype
                    )
                    status = lib.lt_run(
                        plan,
                        index,
                        flat.data_ptr(),
                        packed.value.data_ptr(),
                        output.data_ptr(),
                        workspace.data_ptr(),
                        workspace.numel(),
                        stream.cuda_stream,
                    )
                    if status != 0:
                        raise RuntimeError(f"Thor cuBLASLt returned status {status}")
                    done.record(stream)
                return output.view(*x.shape[:-1], weight.shape[0])
            finally:
                get_pack_cache().record(packed, stream)
