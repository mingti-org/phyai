"""phyai.runtime — runtime / inference plumbing shared by every model.

Modules:

* :mod:`cuda_graph_manager` — :class:`CudaGraph` (single captured
  graph) and :class:`CudaGraphRegistry` (multi-shape dispatch).
* :mod:`model_runner` — :class:`ModelRunner` ABC.
* :mod:`schedule` — scheduler ABC and primitives.
* :mod:`tensor_dump` — :class:`TensorDumper` (forward-hook activation
  capture; eager-only) plus :func:`register_tensor_dumper` /
  :func:`load_pass`.
* :mod:`ipc_buffer` — :class:`CudaIpcBuffer` (GPU buffer shared across
  processes via cudaIpc) and :class:`HostShmBuffer` (host POSIX shared
  memory). Both expose symmetric ``create()`` / ``attach()`` modes
  with serializable :class:`CudaIpcHandle` / :class:`HostShmHandle` —
  application-level transport (zmq, pipe, argv, …) ships handles
  between unrelated OS processes (e.g. preprocessor ↔ model worker).

Per-model runners and schedulers live alongside the model definitions
under :mod:`phyai.models`. The runtime package only owns the shared
base classes.
"""

from __future__ import annotations

from phyai.runtime.cuda_graph_manager import (
    CudaGraph,
    CudaGraphError,
    CudaGraphRegistry,
    capture_on_stream,
)
from phyai.runtime.ipc_buffer import (
    CudaIpcBuffer,
    CudaIpcHandle,
    HostShmBuffer,
    HostShmHandle,
)
from phyai.runtime.model_runner import ModelRunner
from phyai.runtime.schedule import Scheduler
from phyai.runtime.tensor_dump import (
    FilterFn,
    FilterSpec,
    TensorDumper,
    load_filter_fn,
    load_pass,
    register_tensor_dumper,
)


__all__ = [
    "CudaGraph",
    "CudaGraphError",
    "CudaGraphRegistry",
    "capture_on_stream",
    "CudaIpcBuffer",
    "CudaIpcHandle",
    "FilterFn",
    "FilterSpec",
    "HostShmBuffer",
    "HostShmHandle",
    "ModelRunner",
    "Scheduler",
    "TensorDumper",
    "load_filter_fn",
    "load_pass",
    "register_tensor_dumper",
]
