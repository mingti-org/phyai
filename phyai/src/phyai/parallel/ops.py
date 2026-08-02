"""Custom op registration + user-facing collective API.

Each primitive has two pieces:
  1. A ``torch.library.custom_op`` registration so Dynamo / torch.compile
     sees an opaque node and does not graph-break.
  2. A thin user-facing function (``all_reduce``, ``all_gather``, ...) that
     forwards to the custom op.

The Python implementation runs once per CUDA-graph capture (the kernel
is recorded into the graph), and ``torch.compile`` traces the op as an
opaque node so the dispatcher logic is not re-run on every call.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from phyai.utils.cuda import current_device

from phyai.parallel.backend import Op
from phyai.parallel.dispatch import get_dispatcher
from phyai.parallel.mesh import Mesh
from phyai.parallel.state import resolve_mesh


def _name(m: "str | Mesh") -> str:
    return m if isinstance(m, str) else m.name


# ``ReduceOp`` cannot cross the ``torch.library.custom_op`` boundary (only plain
# ints / tensors / strings are allowed), so the user-facing wrappers serialize it
# to ``int(op)`` and the custom-op body reconstructs it here. On torch 2.11
# ``dist.ReduceOp(some_int)`` raises ``TypeError`` (the constructor only accepts a
# ``RedOpType`` enum, not an int), and a freshly built ``ReduceOp`` does not hash-
# equal the canonical ``ReduceOp.SUM`` members that the pynccl backend's lookup
# table is keyed on. Resolving through the canonical members keyed by their int
# value sidesteps both problems: the returned object IS the singleton member, so
# it works for ``dist.all_reduce(op=...)`` and for pynccl's identity dict lookup.
_REDUCE_OP_BY_INT: dict[int, dist.ReduceOp] = {
    int(member): member
    for member in (
        dist.ReduceOp.SUM,
        dist.ReduceOp.PRODUCT,
        dist.ReduceOp.MIN,
        dist.ReduceOp.MAX,
        dist.ReduceOp.AVG,
    )
}


def _reduce_op_from_int(reduce_op: int) -> dist.ReduceOp:
    """Resolve a serialized ``int(op)`` back to its canonical ``ReduceOp`` member."""
    try:
        return _REDUCE_OP_BY_INT[reduce_op]
    except KeyError as e:
        raise ValueError(
            f"unsupported ReduceOp code {reduce_op!r}; expected one of "
            f"{sorted(_REDUCE_OP_BY_INT)}."
        ) from e


def _execute(
    *,
    op: Op,
    mesh_name: str,
    axis: str,
    tensor: Tensor | None,
    output: Tensor | None,
    extra_key: tuple = (),
    **kwargs,
) -> Tensor | None:
    """Common dispatch + execute body shared by every primitive."""
    mesh = resolve_mesh(mesh_name)
    backend = get_dispatcher().select(
        op=op,
        mesh=mesh,
        axis=axis,
        tensor=tensor,
        extra_key=extra_key,
    )
    return backend.execute(
        op=op,
        pg=mesh.axis_group(axis),
        input=tensor,
        output=output,
        _mesh_name=mesh.name,
        _axis=axis,
        _device=tensor.device
        if tensor is not None
        else (output.device if output is not None else torch.device("cuda")),
        **kwargs,
    )


# =============================================================================
#  all_reduce
# =============================================================================


@torch.library.custom_op("phyai::all_reduce", mutates_args=())
def _all_reduce_op(
    x: Tensor,
    mesh_name: str,
    axis: str,
    reduce_op: int,
) -> Tensor:
    mesh = resolve_mesh(mesh_name)
    if mesh.axis_size(axis) <= 1:
        # ws=1: AR is identity. Skip dispatcher entirely.
        return x.clone()
    output = torch.empty_like(x)
    return _execute(
        op=Op.ALL_REDUCE,
        mesh_name=mesh_name,
        axis=axis,
        tensor=x,
        output=output,
        extra_key=(reduce_op,),
        reduce_op=_reduce_op_from_int(reduce_op),
    )


@_all_reduce_op.register_fake
def _(x: Tensor, mesh_name: str, axis: str, reduce_op: int) -> Tensor:
    return torch.empty_like(x)


def all_reduce(
    x: Tensor,
    *,
    axis: str,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    mesh: "str | Mesh" = "model",
) -> Tensor:
    """All-reduce ``x`` along ``axis`` of ``mesh``."""
    return torch.ops.phyai.all_reduce.default(x, _name(mesh), axis, int(op))


# =============================================================================
#  all_gather
# =============================================================================


@torch.library.custom_op("phyai::all_gather", mutates_args=())
def _all_gather_op(
    x: Tensor,
    mesh_name: str,
    axis: str,
    dim: int,
) -> Tensor:
    mesh = resolve_mesh(mesh_name)
    ws = mesh.axis_size(axis)
    if ws <= 1:
        return x.clone()
    out_shape = list(x.shape)
    out_shape[dim] *= ws
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return _execute(
        op=Op.ALL_GATHER,
        mesh_name=mesh_name,
        axis=axis,
        tensor=x,
        output=output,
        extra_key=(dim,),
        dim=dim,
    )


@_all_gather_op.register_fake
def _(x: Tensor, mesh_name: str, axis: str, dim: int) -> Tensor:
    mesh = resolve_mesh(mesh_name)
    ws = mesh.axis_size(axis)
    out_shape = list(x.shape)
    out_shape[dim] *= ws
    return torch.empty(out_shape, dtype=x.dtype, device=x.device)


def all_gather(
    x: Tensor,
    *,
    axis: str,
    dim: int = -1,
    mesh: "str | Mesh" = "model",
) -> Tensor:
    if dim < 0:
        dim += x.ndim
    return torch.ops.phyai.all_gather.default(x, _name(mesh), axis, dim)


# =============================================================================
#  reduce_scatter
# =============================================================================


@torch.library.custom_op("phyai::reduce_scatter", mutates_args=())
def _reduce_scatter_op(
    x: Tensor,
    mesh_name: str,
    axis: str,
    dim: int,
    reduce_op: int,
) -> Tensor:
    mesh = resolve_mesh(mesh_name)
    ws = mesh.axis_size(axis)
    if ws <= 1:
        return x.clone()
    if x.shape[dim] % ws != 0:
        raise ValueError(
            f"reduce_scatter: shape[{dim}]={x.shape[dim]} not divisible "
            f"by world_size={ws}"
        )
    out_shape = list(x.shape)
    out_shape[dim] //= ws
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return _execute(
        op=Op.REDUCE_SCATTER,
        mesh_name=mesh_name,
        axis=axis,
        tensor=x,
        output=output,
        extra_key=(dim, reduce_op),
        dim=dim,
        reduce_op=_reduce_op_from_int(reduce_op),
    )


@_reduce_scatter_op.register_fake
def _(x: Tensor, mesh_name: str, axis: str, dim: int, reduce_op: int) -> Tensor:
    mesh = resolve_mesh(mesh_name)
    ws = mesh.axis_size(axis)
    out_shape = list(x.shape)
    out_shape[dim] //= ws
    return torch.empty(out_shape, dtype=x.dtype, device=x.device)


def reduce_scatter(
    x: Tensor,
    *,
    axis: str,
    dim: int = 0,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    mesh: "str | Mesh" = "model",
) -> Tensor:
    if dim < 0:
        dim += x.ndim
    return torch.ops.phyai.reduce_scatter.default(
        x,
        _name(mesh),
        axis,
        dim,
        int(op),
    )


# =============================================================================
#  all_to_all
# =============================================================================


@torch.library.custom_op("phyai::all_to_all", mutates_args=())
def _all_to_all_op(
    x: Tensor,
    mesh_name: str,
    axis: str,
    in_splits: Optional[list[int]],
    out_splits: Optional[list[int]],
) -> Tensor:
    mesh = resolve_mesh(mesh_name)
    ws = mesh.axis_size(axis)
    if ws <= 1:
        return x.clone()
    if out_splits is not None:
        out_n0 = sum(out_splits)
    elif in_splits is not None:
        # Same-size each rank when no out_splits specified
        out_n0 = x.shape[0]
    else:
        if x.shape[0] % ws != 0:
            raise ValueError(
                f"all_to_all: shape[0]={x.shape[0]} not divisible by ws={ws}"
            )
        out_n0 = x.shape[0]
    out_shape = (out_n0,) + tuple(x.shape[1:])
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return _execute(
        op=Op.ALL_TO_ALL,
        mesh_name=mesh_name,
        axis=axis,
        tensor=x,
        output=output,
        # Only the even/uneven distinction is plausibly selection-relevant
        # (e.g. `all_to_all_single` vs `all_to_allv`). The actual split
        # values flow through **kwargs to backend.execute below.
        extra_key=(in_splits is not None, out_splits is not None),
        in_splits=in_splits,
        out_splits=out_splits,
    )


@_all_to_all_op.register_fake
def _(
    x: Tensor,
    mesh_name: str,
    axis: str,
    in_splits: Optional[list[int]],
    out_splits: Optional[list[int]],
) -> Tensor:
    mesh = resolve_mesh(mesh_name)
    ws = mesh.axis_size(axis)
    if out_splits is not None:
        out_n0 = sum(out_splits)
    else:
        out_n0 = x.shape[0]
    out_shape = (out_n0,) + tuple(x.shape[1:])
    return torch.empty(out_shape, dtype=x.dtype, device=x.device)


def all_to_all(
    x: Tensor,
    *,
    axis: str,
    in_splits: Optional[list[int]] = None,
    out_splits: Optional[list[int]] = None,
    mesh: "str | Mesh" = "model",
) -> Tensor:
    return torch.ops.phyai.all_to_all.default(
        x,
        _name(mesh),
        axis,
        in_splits,
        out_splits,
    )


# =============================================================================
#  broadcast
# =============================================================================


@torch.library.custom_op("phyai::broadcast", mutates_args=())
def _broadcast_op(
    x: Tensor,
    mesh_name: str,
    axis: str,
    src: int,
) -> Tensor:
    mesh = resolve_mesh(mesh_name)
    if mesh.axis_size(axis) <= 1:
        return x.clone()
    output = torch.empty_like(x)
    return _execute(
        op=Op.BROADCAST,
        mesh_name=mesh_name,
        axis=axis,
        tensor=x,
        output=output,
        extra_key=(src,),
        src=src,
    )


@_broadcast_op.register_fake
def _(x: Tensor, mesh_name: str, axis: str, src: int) -> Tensor:
    return torch.empty_like(x)


def broadcast(
    x: Tensor,
    *,
    axis: str,
    src: int = 0,
    mesh: "str | Mesh" = "model",
) -> Tensor:
    return torch.ops.phyai.broadcast.default(x, _name(mesh), axis, src)


# =============================================================================
#  send / recv / barrier — direct calls (no custom op wrapper)
#
#  Rationale: send/recv shape allocation requires (shape, dtype) at op time
#  which makes the schema awkward; barrier has no input/output. These are
#  not on the typical compiled fast-path of a transformer block, so the
#  graph-break risk is low.
# =============================================================================


def send(
    x: Tensor,
    *,
    axis: str,
    dst: int,
    mesh: "str | Mesh" = "model",
) -> None:
    mesh_obj = resolve_mesh(mesh)
    if mesh_obj.axis_size(axis) <= 1:
        return  # ws=1: nothing to send
    backend = get_dispatcher().select(
        op=Op.SEND,
        mesh=mesh_obj,
        axis=axis,
        tensor=x,
        extra_key=(dst,),
    )
    backend.execute(
        op=Op.SEND,
        pg=mesh_obj.axis_group(axis),
        input=x,
        output=None,
        dst=dst,
        _mesh_name=mesh_obj.name,
        _axis=axis,
        _device=x.device,
    )


def recv(
    shape: tuple[int, ...] | list[int],
    dtype: torch.dtype,
    *,
    axis: str,
    src: int,
    device: torch.device | str | None = None,
    mesh: "str | Mesh" = "model",
) -> Tensor:
    mesh_obj = resolve_mesh(mesh)
    if device is None:
        device = current_device()
    output = torch.empty(tuple(shape), dtype=dtype, device=device)
    if mesh_obj.axis_size(axis) <= 1:
        # ws=1: no peer to receive from. Caller's bug if they reach this,
        # but we tolerate it by returning the zeroed buffer.
        return output
    backend = get_dispatcher().select(
        op=Op.RECV,
        mesh=mesh_obj,
        axis=axis,
        tensor=output,
        extra_key=(src,),
    )
    backend.execute(
        op=Op.RECV,
        pg=mesh_obj.axis_group(axis),
        input=None,
        output=output,
        src=src,
        _mesh_name=mesh_obj.name,
        _axis=axis,
        _device=output.device,
    )
    return output


def barrier(
    *,
    axis: str = "world",
    mesh: "str | Mesh" = "model",
) -> None:
    mesh_obj = resolve_mesh(mesh)
    if mesh_obj.axis_size(axis) <= 1:
        return
    backend = get_dispatcher().select(
        op=Op.BARRIER,
        mesh=mesh_obj,
        axis=axis,
        tensor=None,
    )
    backend.execute(
        op=Op.BARRIER,
        pg=mesh_obj.axis_group(axis),
        input=None,
        output=None,
        _mesh_name=mesh_obj.name,
        _axis=axis,
        _device=current_device(),
    )
