"""Dense two-layer FFN with gated and plain forms.

The gated form computes ``down(act(gate(x)) * up(x))``. A
:class:`~phyai.layers.linear.MergedColumnParallelLinear` writes gate first and
up second, which matches FlashInfer's fused ``act_and_mul`` layout. The gated
path accepts ``silu``, exact ``gelu``, or ``gelu_tanh``.

The plain form computes ``fc2(act(fc1(x)))`` and accepts exact or tanh GELU.
Non-gated SiLU is rejected. Activation names are case-insensitive, and hyphens
normalize to underscores. ``gelu_pytorch_tanh`` and ``gelu_new`` both map to
``gelu_tanh``. Exact and tanh GELU use separate kernel IDs, so a policy can pin
either implementation directly.

Each child linear owns its ``hf_keys`` and ``weight_loader``. Gated weights use
``gate_proj``, ``up_proj``, and ``down_proj`` by default; plain weights use
``fc1`` and ``fc2``. ``gated_hf_legs`` overrides the gate and up names.

Activation and quantization run as separate operations. FlashInfer allocates
the fused activation output internally and controls ``enable_pdl``.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from phyai.kernel.call import CallSite, token_shape
from phyai.engine_config import get_engine_config
from phyai.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)


_GATED_ACTS = ("silu", "gelu", "gelu_tanh")
_PLAIN_ACTS = ("gelu", "gelu_tanh")
_TANH_ALIASES = ("gelu_tanh", "gelu_pytorch_tanh", "gelu_new")


def _canonicalise_activation(name: str) -> str:
    """Normalise underscore/hyphen and tanh aliases to a canonical form."""
    canon = name.lower().replace("-", "_")
    if canon in _TANH_ALIASES:
        return "gelu_tanh"
    return canon


def _activation_fn(name: str, *, gated: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable that selects its kernel per call.

    Selection facts include shape, device and execution mode, and one module
    can see several decode/prefill buckets, so the choice cannot be frozen at
    construction. The selector caches on the normalized query, which is a
    superset of what a local memo could key on -- the two closure-local dicts
    this replaces were caching in front of that.

    Gated and plain forms share one catalog operation and differ by the
    ``gated`` fact, because that is exactly what distinguishes the contracts:
    FlashInfer only ships the fused gate-and-multiply variants.
    """

    canon = _canonicalise_activation(name)
    if gated:
        if canon not in _GATED_ACTS:
            raise ValueError(
                f"Unsupported gated activation {name!r}; expected one of "
                f"{_GATED_ACTS!r}."
            )
    else:
        if canon == "silu":
            raise ValueError(
                "non-gated SiLU is not supported (no real model uses it; "
                "did you mean gated=True?)"
            )
        if canon not in _PLAIN_ACTS:
            raise ValueError(
                f"Unsupported plain activation {name!r}; expected one of "
                f"{_PLAIN_ACTS!r}."
            )

    site = CallSite(
        "activation",
        role="mlp.activation",
        attrs={"activation": canon, "gated": gated},
    )

    def activate(value: torch.Tensor) -> torch.Tensor:
        handle = site.select(
            device=value.device,
            dtype={"input": value.dtype, "output": value.dtype},
            dims=token_shape(value, hidden=value.shape[-1]),
        )
        return handle.execute(value)

    return activate


def _resolve_gated_act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return _activation_fn(name, gated=True)


def _resolve_plain_act(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    return _activation_fn(name, gated=False)


class DenseMLP(nn.Module):
    """Generic FFN block. See module docstring for the topology matrix.

    Parameters
    ----------
    hidden_size:
        Input / output channel width. Equal to model hidden size.
    intermediate_size:
        Width of the FFN's hidden activation. For SwiGLU/GeGLU this is
        the size of *each* of ``gate``, ``up``, and the input to
        ``down``.
    activation:
        ``"silu"`` | ``"gelu"`` | ``"gelu_tanh"``. Aliases
        ``gelu_pytorch_tanh`` / ``gelu_new`` map to ``gelu_tanh``.
    gated:
        If ``True`` (default), build the gated SwiGLU/GeGLU path. If
        ``False``, build the plain ``fc1->act->fc2`` path. Cannot combine
        with ``activation="silu"``.
    bias:
        Bias on every internal linear. Llama / Gemma FFNs use
        ``bias=False``; SigLIP / BERT-style use ``bias=True``.
    group / sequence_parallel:
        TP group and optional sequence parallelism. Sequence parallelism
        gathers input tokens and reduce-scatters outputs over the same group.
        ``None`` uses the engine's dense sequence-parallel setting.
    params_dtype:
        Dtype for parameter allocation. Defaults to torch default.
    cast_before_multiply:
        Round the gated activation to the input dtype before multiplying
        the up projection. Default False keeps the fused activation kernel.
    spec_in / spec_out:
        Per-leg :class:`~phyai.layers.quant.WeightSpec`. Most configs
        use the same spec for both, but FP8 mixed-precision recipes
        commonly keep ``down_proj`` at bf16 (sensitive reduction). Two
        knobs, no wrapper class.
    mesh:
        Mesh name. Default ``"model"``.
    prefix:
        Dotted state-dict prefix for THIS module (not its parent).
        Children are constructed with ``prefix=f"{prefix}.gate_up_proj"``
        etc. Empty prefix means children skip ``hf_keys`` attachment;
        such an MLP can still run forward but will not load weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        activation: Literal["silu", "gelu", "gelu_tanh"] = "silu",
        gated: bool = True,
        bias: bool = False,
        group: str = "dense_tp",
        sequence_parallel: bool | None = None,
        params_dtype: torch.dtype | None = None,
        spec_in: object | None = None,
        spec_out: object | None = None,
        gated_hf_legs: tuple[str, str] = ("gate_proj", "up_proj"),
        mesh: str = "model",
        prefix: str = "",
        device: torch.device | str | None = None,
        cast_before_multiply: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation = _canonicalise_activation(activation)
        self.gated = gated
        self.cast_before_multiply = cast_before_multiply
        if cast_before_multiply and not gated:
            raise ValueError("cast_before_multiply requires a gated MLP")
        self.bias_enabled = bias
        self.prefix = prefix
        if sequence_parallel is None:
            sequence_parallel = get_engine_config().parallel.dense.sequence_parallel

        if gated:
            self.gate_up_proj = MergedColumnParallelLinear(
                in_features=hidden_size,
                output_sizes=[intermediate_size, intermediate_size],
                group=group,
                sequence_parallel=sequence_parallel,
                gather_output=False,
                bias=bias,
                params_dtype=params_dtype,
                spec=spec_in,
                hf_legs=gated_hf_legs,
                mesh=mesh,
                device=device,
                prefix=f"{prefix}.gate_up_proj" if prefix else "gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                in_features=intermediate_size,
                out_features=hidden_size,
                group=group,
                sequence_parallel=sequence_parallel,
                input_is_parallel=True,
                reduce_results=True,
                bias=bias,
                params_dtype=params_dtype,
                spec=spec_out,
                mesh=mesh,
                device=device,
                prefix=f"{prefix}.down_proj" if prefix else "down_proj",
            )
            # TODO(fp8/fp4 fused act-quant): wire a spec hook that fuses
            # silu_and_mul + per-token fp8/nvfp4 quant for the down_proj
            # input. Today the act and quant happen as two separate
            # kernels.
            self._act_and_mul = _resolve_gated_act(activation)
            self._act_fn = None
        else:
            self.fc1 = ColumnParallelLinear(
                in_features=hidden_size,
                out_features=intermediate_size,
                group=group,
                sequence_parallel=sequence_parallel,
                gather_output=False,
                bias=bias,
                params_dtype=params_dtype,
                spec=spec_in,
                mesh=mesh,
                device=device,
                prefix=f"{prefix}.fc1" if prefix else "fc1",
            )
            self.fc2 = RowParallelLinear(
                in_features=intermediate_size,
                out_features=hidden_size,
                group=group,
                sequence_parallel=sequence_parallel,
                input_is_parallel=True,
                reduce_results=True,
                bias=bias,
                params_dtype=params_dtype,
                spec=spec_out,
                mesh=mesh,
                device=device,
                prefix=f"{prefix}.fc2" if prefix else "fc2",
            )
            self._act_and_mul = None
            self._act_fn = _resolve_plain_act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            fused, _ = self.gate_up_proj(x)
            if self.cast_before_multiply:
                gate, up = fused.chunk(2, dim=-1)
                if self.activation == "silu":
                    activated = F.silu(gate)
                else:
                    activated = F.gelu(
                        gate,
                        approximate="tanh"
                        if self.activation == "gelu_tanh"
                        else "none",
                    )
                activated = activated * up
            else:
                activated = self._act_and_mul(fused)
            out, _ = self.down_proj(activated)
            return out
        h, _ = self.fc1(x)
        h = self._act_fn(h)
        out, _ = self.fc2(h)
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"activation={self.activation!r}, gated={self.gated}, "
            f"bias={self.bias_enabled}"
        )


__all__ = ["DenseMLP"]
