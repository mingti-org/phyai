"""CUDA-graph capture / replay primitives shared by every model runner.

A :class:`CudaGraph` owns one captured ``torch.cuda.CUDAGraph`` plus its
static input buffers. Each runner's ``forward`` resolves to the same
two-step sequence:

    1. ``graph.input_buffer(name).copy_(actual_input)``  per input
    2. ``graph.replay()``                                # captured kernels run

The output tensors returned by the captured ``fn`` are themselves static
(allocated inside the capture region) and refilled every replay; the
runner reads them by reference.

The capture path follows the standard PyTorch warmup pattern: run the
function on a side stream a few times so the CUDA-caching allocator
finalises every workspace before capture, then enter the graph context
and run once more to record the kernel sequence.

Multiple shape buckets — e.g. one graph per padded prefix length — are
managed by :class:`CudaGraphRegistry`, a small key->graph dict so the
runner can pick the right graph at replay time without an ``if`` ladder.

Captureability of external state
--------------------------------
The captured ``fn`` may freely call into non-graph-aware code (e.g.
:func:`flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper.run`) as
long as every kernel launch reads from / writes to tensors with stable
storage between captures. Any metadata buffer the kernels read (paged
KV indices, etc.) must therefore be **pre-allocated once** by the
caller and updated in place via ``.copy_()`` rather than reassigned.
The ``flashinfer`` wrapper supports exactly this when constructed with
``use_cuda_graph=True`` and the four ``paged_kv_*_buf`` arguments. The
:class:`CudaGraph` itself is agnostic to that contract — it just hands
control to ``fn`` inside the capture region.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, Callable, Hashable, Iterator

import torch

from phyai.parallel.state import graph_capture

_capture_stream_override: ContextVar["torch.cuda.Stream | None"] = ContextVar(
    "phyai_cuda_graph_capture_stream", default=None
)


@contextlib.contextmanager
def capture_on_stream(stream: "torch.cuda.Stream | None") -> Iterator[None]:
    """Pin all :meth:`CudaGraph.capture` calls in this scope onto ``stream``.

    Use it around model / runner construction that captures graphs, when the
    capture must land on a specific (e.g. green-context) stream so the captured
    kernels are confined to that stream's SM partition::

        with vgpu.activate(), capture_on_stream(vgpu.stream):
            engine = build_engine(...)   # graphs captured on the green stream

    Passing ``None`` is a no-op (restores the default fresh-stream capture).
    """
    token = _capture_stream_override.set(stream)
    try:
        yield
    finally:
        _capture_stream_override.reset(token)


class CudaGraphError(RuntimeError):
    """Raised when a :class:`CudaGraph` operation is invoked out of order."""


class CudaGraph:
    """A single captured graph with its static input buffers.

    Parameters
    ----------
    num_warmup_iters:
        How many side-stream warmup iterations to run before capture.
        Three is enough to flush every workspace allocation flashinfer
        and friends do on their first call. Bumped to four when capture
        ever races against a JIT autotuner.
    mempool:
        Optional ``torch.cuda.MemPool`` to share allocations across
        multiple graphs (so they can each free buffers without breaking
        the other captures). Defaults to a fresh per-graph pool.

    Lifecycle
    ---------
    ::

        graph = CudaGraph()
        graph.capture(fn, example_inputs)   # one-shot
        out   = graph.replay(actual_inputs)  # called many times

    Re-capturing into the same instance is rejected — construct a new
    one for the new shape (or use :class:`CudaGraphRegistry`).

    Output references
    -----------------
    :meth:`replay` returns whatever ``fn`` returned at capture time.
    Nested structures are returned by reference; tensor storage is
    stable across replays, so callers should ``.clone()`` outputs they
    intend to retain past the next replay.
    """

    def __init__(
        self,
        *,
        num_warmup_iters: int = 3,
        mempool: "torch.cuda.MemPool | None" = None,
    ) -> None:
        if num_warmup_iters < 0:
            raise ValueError(
                f"num_warmup_iters must be non-negative, got {num_warmup_iters}."
            )
        self._captured = False
        self._graph: torch.cuda.CUDAGraph | None = None
        self._input_buffers: dict[str, torch.Tensor] = {}
        self._output: Any = None
        self.num_warmup_iters = int(num_warmup_iters)
        self.mempool = mempool

    @property
    def is_captured(self) -> bool:
        return self._captured

    def input_buffer(self, name: str) -> torch.Tensor:
        """Return the static input buffer for ``name``.

        Mutating this buffer in place (e.g. ``buf.copy_(new)``) before
        :meth:`replay` is the supported way to feed new values into the
        captured graph.
        """
        if not self._captured:
            raise CudaGraphError("CudaGraph has not been captured yet.")
        try:
            return self._input_buffers[name]
        except KeyError as e:
            raise CudaGraphError(
                f"unknown input buffer {name!r}; known names are "
                f"{sorted(self._input_buffers)!r}."
            ) from e

    def input_buffers(self) -> dict[str, torch.Tensor]:
        """Read-only view of the static input buffer dict."""
        if not self._captured:
            raise CudaGraphError("CudaGraph has not been captured yet.")
        return dict(self._input_buffers)

    def capture(
        self,
        fn: Callable[..., Any],
        example_inputs: dict[str, torch.Tensor],
    ) -> None:
        """Warm up ``fn`` on a side stream and capture it into a CUDA graph.

        Every value in ``example_inputs`` must be a CUDA tensor; their
        contents seed the static input buffers (and thus the warmup
        iterations). After capture the buffers are refilled per
        :meth:`replay` call.

        ``fn`` is called as ``fn(**input_buffers)`` so the keys in
        ``example_inputs`` should match its keyword-only parameter names.
        """
        if self._captured:
            raise CudaGraphError(
                "CudaGraph already captured; construct a new instance to "
                "capture a different fn / shape."
            )
        if not torch.cuda.is_available():
            raise CudaGraphError(
                "CudaGraph.capture requires CUDA; no CUDA device available."
            )
        for name, t in example_inputs.items():
            if not isinstance(t, torch.Tensor):
                raise CudaGraphError(
                    f"example_inputs[{name!r}] must be a Tensor, got {type(t)!r}."
                )
            if not t.is_cuda:
                raise CudaGraphError(
                    f"example_inputs[{name!r}] must live on CUDA, got "
                    f"device={t.device}."
                )

        # Static input buffers — same shape/dtype/device, fresh storage.
        self._input_buffers = {
            name: torch.empty_like(t) for name, t in example_inputs.items()
        }
        for name, t in example_inputs.items():
            self._input_buffers[name].copy_(t)

        # Side-stream warmup so JIT autotuners and one-shot allocators
        # finalise before the capture region. The warmup ALWAYS runs on a fresh
        # normal stream (never the override): flashinfer's first-call path
        # (cuDNN plan build / autotuner profiling) is not valid on a
        # green-context stream and raises cuCtxPushCurrent(201) there. Running
        # it on a normal stream lets those one-shot inits complete cleanly;
        # only the capture region itself goes on the override stream so the
        # captured grids size to its SM partition.
        override = _capture_stream_override.get()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.num_warmup_iters):
                fn(**self._input_buffers)
        torch.cuda.current_stream().wait_stream(s)
        if override is not None:
            # Make the override (capture) stream wait on the warmup so all
            # one-shot inits are visible before capture begins.
            override.wait_stream(s)

        # Capture. ``graph_capture()`` flips the phyai dispatcher
        # contextvar so kernel selection (Linear backends, parallel
        # collectives) honours the capture-safety filter — any kernel
        # whose ``supports_capture()`` returns False is excluded from
        # candidates while we're inside the capture region.
        self._graph = torch.cuda.CUDAGraph()
        graph_kwargs: dict[str, Any] = {}
        if self.mempool is not None:
            graph_kwargs["pool"] = self.mempool
        if override is not None:
            graph_kwargs["stream"] = override
        with graph_capture(), torch.cuda.graph(self._graph, **graph_kwargs):
            self._output = fn(**self._input_buffers)

        self._captured = True

    def replay(self, inputs: dict[str, torch.Tensor] | None = None) -> Any:
        """Copy ``inputs`` into the static buffers and replay the graph.

        ``inputs`` may be ``None`` — the previously buffered values are
        re-used (handy for repeated denoise steps where only a couple of
        buffers change between iterations and the rest stay constant).

        Returns whatever ``fn`` produced at capture time. Tensor storage
        is reused across replays — clone the result if you need to keep
        it past the next replay.
        """
        if not self._captured:
            raise CudaGraphError("CudaGraph.replay() called before capture().")
        if inputs is not None:
            for name, t in inputs.items():
                buf = self._input_buffers.get(name)
                if buf is None:
                    raise CudaGraphError(
                        f"unknown input {name!r}; known names are "
                        f"{sorted(self._input_buffers)!r}."
                    )
                if t.shape != buf.shape:
                    raise CudaGraphError(
                        f"input {name!r} shape {tuple(t.shape)} does not match "
                        f"captured buffer shape {tuple(buf.shape)}."
                    )
                buf.copy_(t, non_blocking=t.is_cuda)
        assert self._graph is not None
        self._graph.replay()
        return self._output


class CudaGraphRegistry:
    """Key -> :class:`CudaGraph` dict for shape-bucketed dispatch.

    Runners that capture multiple graphs (e.g. one per padded prefix
    length, or one per ``B`` value) hold a registry and look up by
    shape key at forward time. No type magic — keys are arbitrary
    hashables; the runner picks the convention.
    """

    def __init__(self) -> None:
        self._graphs: dict[Hashable, CudaGraph] = {}

    def register(self, key: Hashable, graph: CudaGraph) -> None:
        if key in self._graphs:
            raise CudaGraphError(
                f"key {key!r} already registered; explicit deregister + "
                f"reregister is required to overwrite."
            )
        if not graph.is_captured:
            raise CudaGraphError(
                f"cannot register graph for key {key!r}: graph has not been "
                f"captured yet."
            )
        self._graphs[key] = graph

    def deregister(self, key: Hashable) -> CudaGraph:
        try:
            return self._graphs.pop(key)
        except KeyError as e:
            raise CudaGraphError(f"unknown key {key!r}.") from e

    def get(self, key: Hashable) -> CudaGraph | None:
        return self._graphs.get(key)

    def has(self, key: Hashable) -> bool:
        return key in self._graphs

    def __contains__(self, key: Hashable) -> bool:
        return key in self._graphs

    def __getitem__(self, key: Hashable) -> CudaGraph:
        try:
            return self._graphs[key]
        except KeyError as e:
            raise CudaGraphError(
                f"unknown key {key!r}; registered keys: "
                f"{sorted(map(repr, self._graphs))}."
            ) from e

    def __len__(self) -> int:
        return len(self._graphs)

    def keys(self):
        return self._graphs.keys()


__all__ = [
    "CudaGraph",
    "CudaGraphError",
    "CudaGraphRegistry",
]
