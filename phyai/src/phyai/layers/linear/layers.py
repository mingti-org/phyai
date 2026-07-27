"""Linear layers — named-axis TP/SP on top of WeightSpec + KernelDispatcher.

These classes only own the *parallel* aspect (mesh axis, bias placement,
input/output collectives) and delegate everything else to ``self.spec``
and the dispatcher. No fp8 / cutlass / marlin branches live here — the
decision tree is pushed into :class:`KernelDispatcher`.

Weight loading is param-attached: each layer's ``__init__`` (after
``spec.allocate``) sets ``param.hf_keys`` and ``param.weight_loader``
using the shared factories in :mod:`phyai.weights.shards`. The
top-level :func:`phyai.weights.load_pretrained` walks
``named_parameters`` and dispatches; no per-model boilerplate.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

import torch
import torch.nn as nn

import phyai.parallel as P
from phyai.engine_config import get_engine_config
from phyai.layers.linear.dispatch import get_linear_dispatcher
from phyai.layers.linear.spec import Bf16Spec
from phyai.layers.quant import AllocationRequest
from phyai.layers.quant.active import get_active_plan
from phyai.layers.quant.materialize import materialize
from phyai.parallel.state import resolve_mesh
from phyai.utils.cuda import sm_arch
from phyai.weights.shards import _Leg, fused, replicated, sharded


def _M_of(x: torch.Tensor) -> int:
    """Total token count along all batch dims."""
    M = 1
    for s in x.shape[:-1]:
        M *= int(s)
    return M


class LinearBase(nn.Module):
    """Shared state for every Linear variant.

    Subclasses must call :meth:`spec.allocate` exactly once (usually
    straight from ``__init__``) so ``spec`` gets a chance to register
    parameters on ``self``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        scheme: object | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.skip_bias_add = skip_bias_add
        if spec is not None:
            self.spec = spec
        elif scheme is not None:
            self.spec = materialize(scheme, sm_arch())
        else:
            plan = get_active_plan()
            resolved = (
                plan.resolve(prefix, type(self))
                if (plan is not None and prefix)
                else None
            )
            self.spec = (
                materialize(resolved, sm_arch()) if resolved is not None else Bf16Spec()
            )
        self.device = (
            device if device is not None else get_engine_config().device.target
        )
        self.prefix = prefix
        self._bias_requested = bias

    def post_load(self) -> None:
        """Spec-driven post-load fixup (e.g. fp8 per-tensor -> per-channel)."""
        proc = getattr(self.spec, "process_after_loading", None)
        if callable(proc):
            proc(self)

    @staticmethod
    def _attach_optional_scales(layer: nn.Module, hf_base: str) -> None:
        """Attach hf_keys/weight_loader/optional=True to spec-allocated scales.

        ``Fp8Spec`` / ``Nvfp4Spec`` (and any future quant spec) create
        scale parameters on the layer. They are absent in non-quant
        checkpoints, so they're marked optional — missing keys don't
        raise under ``strict=True`` and the spec's
        :meth:`process_after_loading` handles any shape fixup.
        """
        for name in ("weight_scale", "input_scale", "weight_global_scale"):
            p = getattr(layer, name, None)
            if isinstance(p, nn.Parameter):
                p.hf_keys = [(f"{hf_base}.{name}", None)]
                p.weight_loader = replicated()
                p.optional = True

    @staticmethod
    def _attach_weight_loader(layer: nn.Module, loader) -> None:
        load_weight = getattr(layer.spec, "load_weight", None)
        if callable(load_weight):
            layer.weight.weight_loader = lambda _param, loaded, shard_id: load_weight(
                layer,
                loaded,
                shard_id,
                loader,
            )
        else:
            layer.weight.weight_loader = loader


class ReplicatedLinear(LinearBase):
    """Every rank holds the full weight — no collectives."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        scheme: object | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            scheme=scheme,
            device=device,
            prefix=prefix,
        )
        self.spec.allocate(
            self,
            AllocationRequest(
                weight_shape=(out_features, in_features),
                logical_widths=[out_features],
                fused_dim=0,
                params_dtype=self.params_dtype,
                device=self.device,
            ),
        )
        self.input_size_per_partition = in_features
        self.output_size_per_partition = out_features
        self.input_size_global = in_features
        self.output_size_global = out_features
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=self.params_dtype, device=self.device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if prefix:
            weight_loader = replicated()
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self._attach_weight_loader(self, weight_loader)
            if self.bias is not None:
                self.bias.hf_keys = [(f"{prefix}.bias", None)]
                self.bias.weight_loader = replicated()
            self._attach_optional_scales(self, prefix)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        kernel = get_linear_dispatcher().select(
            spec_id=self.spec.spec_id,
            M=_M_of(x),
            N=self.out_features,
            K=self.in_features,
            in_dtype=x.dtype,
            out_dtype=self.params_dtype,
        )
        bias = self.bias if not self.skip_bias_add else None
        y = kernel.apply(self, x, bias)
        return y, (self.bias if self.skip_bias_add else None)


class ColumnParallelLinear(LinearBase):
    """Sharded on the output dim along ``axis`` of ``mesh``.

    With ``sp_axis`` set, the forward first all-gathers ``x`` along that
    axis (dim=0 by convention) — sequence-parallel entry path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        axis: str = "tp",
        sp_axis: str | None = None,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        scheme: object | None = None,
        output_sizes: list[int] | None = None,
        mesh: str = "model",
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            scheme=scheme,
            device=device,
            prefix=prefix,
        )
        mesh_obj = resolve_mesh(mesh)
        self.mesh_name = mesh_obj.name
        self._mesh = mesh_obj
        self.axis = axis
        self.sp_axis = sp_axis
        self.gather_output = gather_output
        self.tp_size = mesh_obj.axis_size(axis)
        self.tp_rank = mesh_obj.axis_local_rank(axis)

        global_sizes = output_sizes if output_sizes is not None else [out_features]
        if sum(global_sizes) != out_features:
            raise ValueError(
                f"output_sizes sum ({sum(global_sizes)}) != out_features ({out_features})"
            )
        per_rank_sizes = []
        for s in global_sizes:
            if s % self.tp_size != 0:
                raise ValueError(
                    f"output partition {s} not divisible by tp_size={self.tp_size}"
                )
            per_rank_sizes.append(s // self.tp_size)
        self.output_partition_sizes = per_rank_sizes
        self.output_sizes_global = list(global_sizes)

        self.spec.allocate(
            self,
            AllocationRequest(
                weight_shape=(sum(per_rank_sizes), in_features),
                logical_widths=per_rank_sizes,
                fused_dim=0,
                params_dtype=self.params_dtype,
                device=self.device,
            ),
        )
        self.input_size_per_partition = in_features
        self.output_size_per_partition = sum(per_rank_sizes)
        self.input_size_global = in_features
        self.output_size_global = out_features

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(
                    sum(per_rank_sizes), dtype=self.params_dtype, device=self.device
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        # Non-fused column-parallel: subclasses (Merged / QKV) override
        # by re-attaching after super().__init__ returns.
        if prefix and len(per_rank_sizes) == 1:
            weight_loader = sharded(dim=0, axis=axis, mesh=mesh_obj)
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self._attach_weight_loader(self, weight_loader)
            if self.bias is not None:
                self.bias.hf_keys = [(f"{prefix}.bias", None)]
                self.bias.weight_loader = sharded(dim=0, axis=axis, mesh=mesh_obj)
            self._attach_optional_scales(self, prefix)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.sp_axis is not None:
            x = P.all_gather(x, axis=self.sp_axis, dim=0)

        kernel = get_linear_dispatcher().select(
            spec_id=self.spec.spec_id,
            M=_M_of(x),
            N=self.output_size_per_partition,
            K=self.input_size_per_partition,
            in_dtype=x.dtype,
            out_dtype=self.params_dtype,
        )
        bias = self.bias if not self.skip_bias_add else None
        y = kernel.apply(self, x, bias)

        if self.gather_output and self.tp_size > 1:
            y = P.all_gather(y, axis=self.axis, dim=-1)
        return y, (self.bias if self.skip_bias_add else None)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Gate/up-style fused ColumnParallelLinear; ``output_sizes=[gate, up]``."""

    DEFAULT_HF_LEGS: tuple[str, ...] = ("gate_proj", "up_proj")

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        *,
        axis: str = "tp",
        sp_axis: str | None = None,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        scheme: object | None = None,
        hf_legs: Sequence[str] | None = None,
        mesh: str = "model",
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_features,
            sum(output_sizes),
            axis=axis,
            sp_axis=sp_axis,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            scheme=scheme,
            output_sizes=output_sizes,
            mesh=mesh,
            device=device,
            prefix=prefix,
        )
        legs = tuple(hf_legs) if hf_legs is not None else self.DEFAULT_HF_LEGS
        if len(legs) != len(self.output_partition_sizes):
            raise ValueError(
                f"hf_legs length {len(legs)} != output_partition_sizes "
                f"{len(self.output_partition_sizes)}"
            )
        self.hf_legs = legs

        if prefix:
            parent = prefix.rpartition(".")[0]
            leg_dict: dict[int, _Leg] = {}
            keys: list[tuple[str, int]] = []
            bias_keys: list[tuple[str, int]] = []
            offset = 0
            for i, (leg_name, per_rank) in enumerate(
                zip(legs, self.output_partition_sizes)
            ):
                hf_base = f"{parent}.{leg_name}" if parent else leg_name
                leg_dict[i] = _Leg(
                    offset=offset, size=per_rank, dim=0, axis=axis, replicate=1
                )
                keys.append((f"{hf_base}.weight", i))
                if self.bias is not None:
                    bias_keys.append((f"{hf_base}.bias", i))
                offset += per_rank
            self.weight.hf_keys = keys
            weight_loader = fused(fuse_dim=0, legs=leg_dict, mesh=self._mesh)
            self._attach_weight_loader(self, weight_loader)
            if self.bias is not None:
                self.bias.hf_keys = bias_keys
                self.bias.weight_loader = fused(
                    fuse_dim=0, legs=leg_dict, mesh=self._mesh
                )


class QKVParallelLinear(ColumnParallelLinear):
    """Q/K/V fused ColumnParallelLinear with GQA support.

    When ``tp_size`` exceeds ``num_kv_heads``, K and V are replicated
    ``tp_size // num_kv_heads`` times so every rank has a full set.
    The replica logic shows up in the ``fused(...)`` loader's
    ``_Leg(replicate=...)`` for the K and V legs.
    """

    DEFAULT_HF_LEGS: Mapping[str, str] = MappingProxyType(
        {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
    )

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        axis: str = "tp",
        sp_axis: str | None = None,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        scheme: object | None = None,
        hf_legs: Mapping[str, str] | None = None,
        mesh: str = "model",
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        mesh_obj = resolve_mesh(mesh)
        tp_size = mesh_obj.axis_size(axis)

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if tp_size >= num_kv_heads:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"tp_size={tp_size} not a multiple of num_kv_heads={num_kv_heads}"
                )
            num_kv_replicas = tp_size // num_kv_heads
            effective_kv_heads = tp_size
        else:
            if num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"num_kv_heads={num_kv_heads} not divisible by tp_size={tp_size}"
                )
            num_kv_replicas = 1
            effective_kv_heads = num_kv_heads

        q_size = num_heads * head_dim
        kv_size = effective_kv_heads * head_dim
        out_features = q_size + 2 * kv_size

        super().__init__(
            hidden_size,
            out_features,
            axis=axis,
            sp_axis=sp_axis,
            gather_output=gather_output,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            scheme=scheme,
            output_sizes=[q_size, kv_size, kv_size],
            mesh=mesh,
            device=device,
            prefix=prefix,
        )
        self.num_kv_replicas = num_kv_replicas
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        legs = dict(hf_legs) if hf_legs is not None else dict(self.DEFAULT_HF_LEGS)
        for required in ("q", "k", "v"):
            if required not in legs:
                raise ValueError(f"hf_legs missing key {required!r}; got {legs!r}")
        self.hf_legs = legs

        if prefix:
            parent = prefix.rpartition(".")[0]
            q_local, k_local, v_local = self.output_partition_sizes
            leg_dict: dict[str, _Leg] = {
                "q": _Leg(offset=0, size=q_local, dim=0, axis=axis, replicate=1),
                "k": _Leg(
                    offset=q_local,
                    size=k_local,
                    dim=0,
                    axis=axis,
                    replicate=num_kv_replicas,
                ),
                "v": _Leg(
                    offset=q_local + k_local,
                    size=v_local,
                    dim=0,
                    axis=axis,
                    replicate=num_kv_replicas,
                ),
            }
            keys: list[tuple[str, str]] = []
            bias_keys: list[tuple[str, str]] = []
            for kind in ("q", "k", "v"):
                hf_name = legs[kind]
                hf_base = f"{parent}.{hf_name}" if parent else hf_name
                keys.append((f"{hf_base}.weight", kind))
                if self.bias is not None:
                    bias_keys.append((f"{hf_base}.bias", kind))
            self.weight.hf_keys = keys
            weight_loader = fused(fuse_dim=0, legs=leg_dict, mesh=self._mesh)
            self._attach_weight_loader(self, weight_loader)
            if self.bias is not None:
                self.bias.hf_keys = bias_keys
                self.bias.weight_loader = fused(
                    fuse_dim=0, legs=leg_dict, mesh=self._mesh
                )


class RowParallelLinear(LinearBase):
    """Sharded on the input dim along ``axis`` of ``mesh``.

    Exit collective is ``all_reduce`` on ``axis`` unless ``sp_axis`` is
    set, in which case it becomes ``reduce_scatter`` on ``sp_axis``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        axis: str = "tp",
        sp_axis: str | None = None,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        spec: object | None = None,
        scheme: object | None = None,
        mesh: str = "model",
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            spec=spec,
            scheme=scheme,
            device=device,
            prefix=prefix,
        )
        mesh_obj = resolve_mesh(mesh)
        self.mesh_name = mesh_obj.name
        self._mesh = mesh_obj
        self.axis = axis
        self.sp_axis = sp_axis
        self.tp_size = mesh_obj.axis_size(axis)
        self.tp_rank = mesh_obj.axis_local_rank(axis)
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        if in_features % self.tp_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by tp_size={self.tp_size}"
            )
        in_per_rank = in_features // self.tp_size

        self.spec.allocate(
            self,
            AllocationRequest(
                weight_shape=(out_features, in_per_rank),
                logical_widths=[out_features],
                fused_dim=0,
                params_dtype=self.params_dtype,
                device=self.device,
            ),
        )
        self.input_size_per_partition = in_per_rank
        self.output_size_per_partition = out_features
        self.input_size_global = in_features
        self.output_size_global = out_features
        if bias:
            # RowParallel bias is global (only rank 0 adds it at forward), so
            # every rank loads the full disk tensor unsliced.
            self.bias = nn.Parameter(
                torch.zeros(out_features, dtype=self.params_dtype, device=self.device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if prefix:
            weight_loader = sharded(dim=1, axis=axis, mesh=mesh_obj)
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self._attach_weight_loader(self, weight_loader)
            if self.bias is not None:
                # Bias is replicated for row-parallel — full copy, no slice.
                self.bias.hf_keys = [(f"{prefix}.bias", None)]
                self.bias.weight_loader = replicated()
            self._attach_optional_scales(self, prefix)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.input_is_parallel and self.tp_size > 1:
            shard = x.shape[-1] // self.tp_size
            x = x.narrow(-1, self.tp_rank * shard, shard).contiguous()

        kernel = get_linear_dispatcher().select(
            spec_id=self.spec.spec_id,
            M=_M_of(x),
            N=self.output_size_per_partition,
            K=self.input_size_per_partition,
            in_dtype=x.dtype,
            out_dtype=self.params_dtype,
        )
        # Only rank 0 adds the bias to avoid double counting post-reduce.
        add_bias = (
            self.bias
            if (self.bias is not None and self.tp_rank == 0 and not self.skip_bias_add)
            else None
        )
        y = kernel.apply(self, x, add_bias)

        if self.reduce_results and self.tp_size > 1:
            if self.sp_axis is not None:
                y = P.reduce_scatter(y, axis=self.sp_axis, dim=0)
            else:
                y = P.all_reduce(y, axis=self.axis)
        return y, (self.bias if self.skip_bias_add else None)
