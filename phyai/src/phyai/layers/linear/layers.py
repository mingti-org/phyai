"""Named-group TP/SP linear layers on top of WeightSpec and kernel selection.

These classes only own the *parallel* aspect (mesh group, bias placement,
input/output collectives) and delegate everything else to ``self.spec``
and the unified kernel selector. No fp8 / cutlass / marlin branches live here;
the catalog owns that decision tree.

Weight loading is param-attached: each layer's ``__init__`` (after
``spec.allocate``) sets ``param.hf_keys`` and ``param.weight_loader``
using the shared factories in :mod:`phyai.weights.shards`. The
top-level :func:`phyai.weights.load_pretrained` walks
``named_parameters`` and dispatches; no per-model boilerplate.
"""

from __future__ import annotations

from types import MappingProxyType
from collections.abc import Mapping, Sequence

import torch
from torch import nn

import phyai.parallel as P
from phyai.kernel.call import CallSite
from phyai.layers.quant import AllocationRequest
from phyai.engine_config import get_engine_config
from phyai.parallel.state import resolve_mesh
from phyai.weights.shards import _Leg, fused, sharded, replicated, packed_sharded
from phyai.layers.linear.spec import Bf16Spec
from phyai.layers.quant.active import get_active_plan
from phyai.layers.quant.materialize import materialize


def _M_of(x: torch.Tensor) -> int:
    """Total token count along all batch dims."""
    M = 1
    for s in x.shape[:-1]:
        M *= int(s)
    return M


def _resolve_linear_kernel(
    layer: LinearBase,
    x: torch.Tensor,
    *,
    M: int,
    N: int,
    K: int,
):
    """Select the GEMM implementation for one call.

    The quantization signature is passed as structured facts rather than as an
    identifier string. Each catalog row names its exact storage format, so
    there is no lossy round trip through a ``spec_id`` and no possibility of a
    row claiming eligibility for a format its code cannot execute.
    """

    # One binding per layer, created on first use and stashed on the layer.
    # Lazy rather than in ``__init__`` because this adapter also serves the tied
    # vocabulary head, which is not a ``LinearBase``.
    site = getattr(layer, "gemm_call", None)
    if site is None or site.role != layer.kernel_role:
        site = CallSite("gemm", role=layer.kernel_role)
        layer.gemm_call = site

    signature = layer.spec.physical_signature
    return site.select(
        device=x.device,
        dtype={
            "input": x.dtype,
            "output": x.dtype
            if signature.format in {"bf16", "fp16"}
            else layer.params_dtype,
            "weight": layer.weight.dtype,
        },
        quant=signature,
        dims={"M": M, "N": N, "K": K},
    )


# Public adapter for non-LinearBase consumers such as the tied vocabulary head.
# Keeping the query construction in one place prevents those consumers from
# silently drifting back to the legacy dispatcher.
resolve_linear_kernel = _resolve_linear_kernel


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
        kernel_role: str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.skip_bias_add = skip_bias_add
        resolved_device = (
            device if device is not None else get_engine_config().device.target
        )
        if spec is not None:
            self.spec = spec
        elif scheme is not None:
            self.spec = materialize(scheme, resolved_device)
        else:
            plan = get_active_plan()
            resolved = (
                plan.resolve(prefix, type(self))
                if (plan is not None and prefix)
                else None
            )
            self.spec = (
                materialize(resolved, resolved_device)
                if resolved is not None
                else Bf16Spec()
            )
        self.device = resolved_device
        self.prefix = prefix
        self.kernel_role = (
            kernel_role or type(self).__name__.removesuffix("Linear").lower()
        )
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
        kernel_role: str | None = None,
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
            kernel_role=kernel_role,
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
        kernel = _resolve_linear_kernel(
            self,
            x,
            M=_M_of(x),
            N=self.out_features,
            K=self.in_features,
        )
        bias = self.bias if not self.skip_bias_add else None
        y = kernel.execute(self, x, bias)
        return y, (self.bias if self.skip_bias_add else None)


class ColumnParallelLinear(LinearBase):
    """Sharded on the output dim along ``group`` of ``mesh``.

    With ``sequence_parallel=True``, the forward first all-gathers tokens
    along the same TP group (dim=0 by convention).
    ``output_sizes`` describes a packed checkpoint such as ``[Q; K; V]``;
    each partition is sharded independently on load.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: str = "dense_tp",
        sequence_parallel: bool = False,
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
        kernel_role: str | None = None,
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
            kernel_role=kernel_role,
        )
        mesh_obj = resolve_mesh(mesh)
        self.mesh_name = mesh_obj.name
        self._mesh = mesh_obj
        self.group = group
        self.sequence_parallel = sequence_parallel
        if not isinstance(sequence_parallel, bool):
            raise TypeError("sequence_parallel must be a bool.")
        self.gather_output = gather_output
        self.tp_size = mesh_obj.group_size(group)
        self.tp_rank = mesh_obj.group_rank(group)

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

        # Merged / QKV subclasses replace this single-source plan with their
        # separate checkpoint legs after construction.
        if prefix:
            weight_loader = (
                sharded(dim=0, group=group, mesh=mesh_obj)
                if len(per_rank_sizes) == 1
                else packed_sharded(sizes=global_sizes, group=group, mesh=mesh_obj)
            )
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self._attach_weight_loader(self, weight_loader)
            if self.bias is not None:
                self.bias.hf_keys = [(f"{prefix}.bias", None)]
                self.bias.weight_loader = weight_loader
            self._attach_optional_scales(self, prefix)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.sequence_parallel:
            x = P.all_gather(x, group=self.group, dim=0, mesh=self.mesh_name)

        kernel = _resolve_linear_kernel(
            self,
            x,
            M=_M_of(x),
            N=self.output_size_per_partition,
            K=self.input_size_per_partition,
        )
        bias = self.bias if not self.skip_bias_add else None
        y = kernel.execute(self, x, bias)

        if self.gather_output and self.tp_size > 1:
            y = P.all_gather(y, group=self.group, dim=-1, mesh=self.mesh_name)
        return y, (self.bias if self.skip_bias_add else None)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Gate/up-style fused ColumnParallelLinear; ``output_sizes=[gate, up]``."""

    DEFAULT_HF_LEGS: tuple[str, ...] = ("gate_proj", "up_proj")

    def __init__(
        self,
        in_features: int,
        output_sizes: list[int],
        *,
        group: str = "dense_tp",
        sequence_parallel: bool = False,
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
        kernel_role: str | None = None,
    ) -> None:
        super().__init__(
            in_features,
            sum(output_sizes),
            group=group,
            sequence_parallel=sequence_parallel,
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
            kernel_role=kernel_role,
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
                    offset=offset,
                    size=per_rank,
                    dim=0,
                    group=group,
                    replication_factor=1,
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
    ``_Leg(replication_factor=...)`` for the K and V legs.
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
        group: str = "attention_tp",
        sequence_parallel: bool = False,
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
        kernel_role: str | None = None,
    ) -> None:
        mesh_obj = resolve_mesh(mesh)
        tp_size = mesh_obj.group_size(group)

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
            group=group,
            sequence_parallel=sequence_parallel,
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
            kernel_role=kernel_role,
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
                "q": _Leg(
                    offset=0, size=q_local, dim=0, group=group, replication_factor=1
                ),
                "k": _Leg(
                    offset=q_local,
                    size=k_local,
                    dim=0,
                    group=group,
                    replication_factor=num_kv_replicas,
                ),
                "v": _Leg(
                    offset=q_local + k_local,
                    size=v_local,
                    dim=0,
                    group=group,
                    replication_factor=num_kv_replicas,
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
    """Sharded on the input dim along ``group`` of ``mesh``.

    The exit collective is ``all_reduce`` on ``group``, or ``reduce_scatter``
    on the same group when ``sequence_parallel=True``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: str = "dense_tp",
        sequence_parallel: bool = False,
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
        kernel_role: str | None = None,
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
            kernel_role=kernel_role,
        )
        mesh_obj = resolve_mesh(mesh)
        self.mesh_name = mesh_obj.name
        self._mesh = mesh_obj
        self.group = group
        self.sequence_parallel = sequence_parallel
        if not isinstance(sequence_parallel, bool):
            raise TypeError("sequence_parallel must be a bool.")
        self.tp_size = mesh_obj.group_size(group)
        self.tp_rank = mesh_obj.group_rank(group)
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
            weight_loader = sharded(dim=1, group=group, mesh=mesh_obj)
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

        kernel = _resolve_linear_kernel(
            self,
            x,
            M=_M_of(x),
            N=self.output_size_per_partition,
            K=self.input_size_per_partition,
        )
        # Only rank 0 adds the bias to avoid double counting post-reduce.
        add_bias = (
            self.bias
            if (self.bias is not None and self.tp_rank == 0 and not self.skip_bias_add)
            else None
        )
        y = kernel.execute(self, x, add_bias)

        if self.reduce_results and self.tp_size > 1:
            if self.sequence_parallel:
                y = P.reduce_scatter(y, group=self.group, dim=0, mesh=self.mesh_name)
            else:
                y = P.all_reduce(y, group=self.group, mesh=self.mesh_name)
        return y, (self.bias if self.skip_bias_add else None)
