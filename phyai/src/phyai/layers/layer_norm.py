"""RMSNorm + LayerNorm + AdaRMSNorm with selectable kernel backends.

Four related modules in this file:

* :class:`RMSNorm` — standard RMSNorm; :class:`GemmaRMSNorm` for the
  ``(1 + w)`` variant. Used by RMSNorm-based text decoders.
* :class:`GatedRMSNorm` — RMSNorm fused with a SiLU gate
  (``rmsnorm(x) * silu(gate)``), the head-wise gated norm of the
  Qwen3.5 / Qwen3-Next GDN mixer family.
* :class:`LayerNorm` — standard mean/variance LayerNorm with optional
  bias. Used by ViT-style vision encoders.
* :class:`AdaRMSNorm` — adaptive RMSNorm with a learned conditioning
  projection. Replaces the ``(1 + w)`` affine with ``(1 + scale)`` and
  ``+ shift`` from a per-token ``cond`` vector, and exposes a ``gate``
  output for the surrounding gated-residual. Used by adaptive-norm
  variants of decoder layers (``use_adarms=True``).

The constructor's ``backend=`` argument is a *preference*, not a decision.
The kernel selector owns selection: it filters by what each implementation
declares it can execute, then orders by policy. ``backend=`` puts one
backend's kernels first among the eligible ones, and is validated against the
catalog — a typo raises and the error lists the names this build offers,
while naming something unavailable on this host falls back rather than
failing. An operator's policy file outranks it. FlashInfer ships no AdaRMS
kernel, so it is never among the AdaRMSNorm names.

Affine dtypes are *derived*, not guessed. FlashInfer's LayerNorm requires
fp32 gamma and beta while its RMSNorm reads gamma through the input's C type;
:class:`LayerNorm` asks the selector which dtype keeps the strongest
implementation eligible instead of testing a backend name.

Reductions and the affine multiply run in fp32 on every backend; output
is cast back to ``x.dtype``. RMSNorm's ``forward`` accepts an optional
``residual`` for the fused ``residual += x; rmsnorm(residual)`` path used
between attention and the MLP in most decoder blocks. LayerNorm has no
fused-add path today (encoder paths don't need one). AdaRMSNorm's
``forward(x, cond)`` returns a ``(out, gate)`` tuple.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from phyai.engine_config import get_engine_config
from phyai.kernel.call import (
    CallSite,
    backend_preference,
    param_dtypes,
    token_shape,
    torch_dtype,
)
from phyai.kernel.types import dtype_name
from phyai.layers.linear import ReplicatedLinear
from phyai.weights.shards import replicated


class RMSNorm(nn.Module):
    """Standard RMSNorm with selectable kernel backend.

    Computes ``y = (x * rsqrt(mean(x ** 2) + eps)) * weight``. The variance
    and the weight multiply both run in fp32; the result is cast back to
    ``x.dtype`` on the way out.

    Parameters
    ----------
    hidden_size:
        Size of the last dim of the input. Weight is ``(hidden_size,)``.
    eps:
        Added to the variance before ``rsqrt`` for numerical stability.
    backend:
        ``"flashinfer"`` (default) or ``"phyai-kernel"``. Underscore,
        hyphen, and case are normalized.
    elementwise_affine:
        Whether to register a weight parameter. If False, use a fixed identity
        weight buffer that is excluded from checkpoints.
    dtype:
        Optional weight dtype. Defaults to the global default dtype.
        **flashinfer caveat**: the CUDA RMSNorm / GemmaRMSNorm /
        FusedAddRMSNorm kernels do *not* check weight dtype — they
        ``static_cast`` the weight pointer to the input ``c_type`` (fp16
        or bf16) inside the dispatch macro. Passing an fp32 weight when
        the input is bf16 silently produces garbage. So when
        ``backend="flashinfer"``, ``dtype`` must match the dtype of the
        tensor that will be normalized (typically ``torch.bfloat16``);
        do *not* leave it as the fp32 default. The ``"phyai-kernel"``
        Triton path accepts any floating dtype.
    cast_before_affine:
        Round normalized values to the activation dtype before multiplying
        gamma. This matches checkpoints whose reference uses a separate
        low-precision affine multiply; weight and activation dtypes must match.
        Has no effect when ``elementwise_affine=False``.

    The forward signature is ``forward(x, residual=None)``:

    * with ``residual`` left as ``None``, returns the normalized tensor;
    * with a ``residual`` tensor, returns ``(y, residual)``. Both buffers
      are written in place by the kernel and the same objects come back.

    On the no-residual path, higher-rank inputs are flattened to 2-D for
    the kernel and reshaped back. The residual path expects 2-D contiguous
    inputs (the kernels themselves don't know how to reshape).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        backend: str | None = None,
        *,
        elementwise_affine: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        kernel_role: str = "norm",
        cast_before_affine: bool = False,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.prefix = prefix
        self.kernel_role = kernel_role
        self.elementwise_affine = elementwise_affine
        self.cast_before_affine = cast_before_affine and elementwise_affine
        if self.cast_before_affine and self.variant != "rms":
            raise ValueError("cast_before_affine applies to standard RMSNorm only")
        if device is None:
            device = get_engine_config().device.target
        # Validates the hint against the catalog and records the kernel ids to
        # try first. A soft preference: policy still outranks it, and a hint
        # naming something unavailable on this host is not fatal.
        self._prefer = backend_preference("rmsnorm", backend)
        self._prefer_fused = backend_preference("rmsnorm_add", backend)
        # Bound once here so the forward pass pays a tuple build and a dict
        # lookup instead of constructing a query. See kernel/call.py.
        self._plain_call = CallSite(
            "rmsnorm",
            role=kernel_role,
            prefer=self._prefer,
            dims={"hidden": hidden_size},
        )
        self._fused_call = CallSite(
            "rmsnorm_add",
            role=kernel_role,
            prefer=self._prefer_fused,
            dims={"hidden": hidden_size},
        )
        weight = self._initial_weight(hidden_size, dtype, device)
        if elementwise_affine:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.register_buffer("weight", weight, persistent=False)
        if self.cast_before_affine:
            self.register_buffer(
                "_unit_weight", torch.ones_like(self.weight), persistent=False
            )
        if prefix and elementwise_affine:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()

    @property
    def variant(self) -> str:
        """``"gemma"`` for the ``(1 + w)`` subclass, else ``"rms"``."""

        return "rms"

    @staticmethod
    def _initial_weight(
        hidden_size: int,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> torch.Tensor:
        # The kernel multiplies by ``w``, so identity is ``w == 1``.
        return torch.ones(hidden_size, dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.cast_before_affine and self.weight.dtype != x.dtype:
            raise ValueError(
                "cast_before_affine requires matching activation and weight dtypes"
            )
        weight = self._unit_weight if self.cast_before_affine else self.weight
        dtypes: dict[str, object] = {
            "input": x.dtype,
            "weight": weight.dtype,
            "output": x.dtype,
        }

        if residual is not None:
            # The fused residual add is its own operation, because its
            # eligibility depends on the residual's dtype — something the plain
            # form has no opinion about.
            dtypes["residual"] = residual.dtype
            handle = self._fused_call.select(
                device=x.device,
                dtype=dtypes,
                dims=token_shape(x),
                attrs={"variant": self.variant},
            )
            if x.dim() == 2:
                out, res = handle.execute(
                    x, residual, weight.data, self.variance_epsilon
                )
                if self.cast_before_affine:
                    out.mul_(self.weight)
                return out, res
            # Fused kernels operate on (tokens, hidden). For contiguous inputs
            # the reshape is a view, so the kernels' in-place writes still land
            # in the caller's buffers.
            orig_shape = x.shape
            out, res = handle.execute(
                x.reshape(-1, orig_shape[-1]),
                residual.reshape(-1, orig_shape[-1]),
                weight.data,
                self.variance_epsilon,
            )
            if self.cast_before_affine:
                out.mul_(self.weight)
            return out.reshape(orig_shape), res.reshape(orig_shape)

        handle = self._plain_call.select(
            device=x.device,
            dtype=dtypes,
            dims=token_shape(x),
            attrs={"variant": self.variant},
        )
        needs_reshape = x.dim() != 2
        if needs_reshape:
            orig_shape = x.shape
            x = x.contiguous().reshape(-1, orig_shape[-1])
        out = handle.execute(x, weight.data, self.variance_epsilon)
        if self.cast_before_affine:
            out = out * self.weight
        return out.reshape(orig_shape) if needs_reshape else out

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, eps={self.variance_epsilon}, backend={self.backend!r}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class GemmaRMSNorm(RMSNorm):
    """``(1 + w)`` RMSNorm variant.

    Selecting the ``(1 + w)`` kernel is now a fact about the call —
    ``attrs.variant == "gemma"`` — rather than a subclass override that
    imported a different pair of functions. The subclass therefore only has to
    say which variant it is, and start the weight at zero so a freshly
    constructed module is the identity. Matches the HF transformers convention.
    """

    @property
    def variant(self) -> str:
        return "gemma"

    @staticmethod
    def _initial_weight(
        hidden_size: int,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> torch.Tensor:
        # The ``(1 + w)`` kernel multiplies by ``(1 + w)``, so identity is ``w == 0``.
        return torch.zeros(hidden_size, dtype=dtype, device=device)


class GatedRMSNorm(nn.Module):
    """RMSNorm whose output is gated by ``silu(gate)``.

    Computes ``rmsnorm(x) * silu(gate)`` through the fused
    ``rmsnorm_silu_mul`` catalog op — the head-wise gated norm used by the
    Qwen3.5 / Qwen3-Next GDN mixer family, where ``x`` is the GDN core
    output and ``gate`` its parallel z-projection.

    Reductions and the gate run in fp32 on every backend; the result is
    cast back to ``x.dtype``. The weight defaults to fp32 (no backend of
    this op reads gamma through the input's C type, so the flashinfer
    caveat on :class:`RMSNorm` does not apply here).

    ``forward(x, gate)`` expects ``x`` and ``gate`` with matching shapes;
    both are made contiguous for the kernel.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        backend: str | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        kernel_role: str = "gated_norm",
    ) -> None:
        super().__init__()
        self.backend = backend
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.prefix = prefix
        self.kernel_role = kernel_role
        if device is None:
            device = get_engine_config().device.target
        self._prefer = backend_preference("rmsnorm_silu_mul", backend)
        self._call = CallSite(
            "rmsnorm_silu_mul",
            role=kernel_role,
            prefer=self._prefer,
            dims={"hidden": hidden_size},
        )
        self.weight = nn.Parameter(
            torch.ones(
                hidden_size,
                dtype=torch.float32 if dtype is None else dtype,
                device=device,
            ),
            requires_grad=False,
        )
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        handle = self._call.select(
            device=x.device,
            dtype={"input": x.dtype, "weight": self.weight.dtype},
            dims=token_shape(x),
        )
        return handle.execute(
            x.contiguous(),
            gate.contiguous(),
            self.weight,
            self.variance_epsilon,
        )

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, eps={self.variance_epsilon}, backend={self.backend!r}"
        )


class LayerNorm(nn.Module):
    """Standard LayerNorm with a selectable kernel backend.

    Computes ``y = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias``,
    with mean / variance / affine all in fp32 and the output cast back to
    ``x.dtype``. This is the path used by ViT-style encoder layers
    (typically two per encoder block plus a final ``post_layernorm``).

    Parameters
    ----------
    hidden_size:
        Last dim of the input. ``weight`` and (when present) ``bias`` are
        ``(hidden_size,)``.
    eps:
        Numerical-stability epsilon. Default ``1e-5`` matches
        :class:`torch.nn.LayerNorm`; ViT-style configs typically use
        ``1e-6``.
    backend:
        ``"flashinfer"`` (default) or ``"phyai-kernel"``.
    bias:
        Whether to allocate a learnable ``beta``. Defaults to ``True``
        (the typical encoder configuration). flashinfer's kernel always
        reads ``beta``; when ``bias=False`` the wrapper feeds it a zero
        buffer so the kernel's add becomes a no-op.
    elementwise_affine:
        If False, use nonpersistent identity buffers instead of checkpoint
        parameters. The same selected normalization kernel still runs.
    dtype:
        Optional weight / bias dtype. Defaults to the global default.
        flashinfer's CUDA kernel hard-checks ``gamma`` / ``beta`` in
        fp32 (``norm.cu`` aborts otherwise), so this wrapper overrides
        the caller's ``dtype`` to ``torch.float32`` when
        ``backend="flashinfer"`` — the parameters are allocated in fp32
        once at construction and the hot path can hand the buffers to
        the kernel directly, no per-forward cast. The Triton kernel
        accepts any floating dtype natively.
    prefix:
        Dotted state-dict prefix for placement loading.

    Forward
    -------
    ``forward(x) -> y`` where ``x`` is any 2-D-or-higher tensor with last
    dim ``hidden_size``. Higher-rank inputs are flattened to ``(N, D)``
    for the kernel and reshaped back.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        backend: str | None = None,
        *,
        bias: bool = True,
        elementwise_affine: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        kernel_role: str = "layernorm",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        self.backend = backend
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.elementwise_affine = elementwise_affine
        bias = bias and elementwise_affine
        self.has_bias = bias
        self.prefix = prefix
        self.kernel_role = kernel_role
        if device is None:
            device = get_engine_config().device.target
        self._prefer = backend_preference("layernorm", backend)
        self._call = CallSite(
            "layernorm",
            role=kernel_role,
            prefer=self._prefer,
            dims={"hidden": hidden_size},
            attrs={"bias": bias},
        )

        # Affine parameters have to be allocated now, before any input tensor
        # exists, so the dtype cannot come from "whatever the selector picks".
        # Instead ask which dtype keeps the strongest implementation eligible:
        # FlashInfer's CUDA kernel hard-requires fp32 gamma and beta, the
        # Triton one takes any float, and the reference path casts. On an
        # NVIDIA host with no preference that yields fp32 — the same value the
        # old code hardcoded, but now a consequence of the declared contracts
        # rather than a string test against a backend name. Adding a bf16-gamma
        # kernel later changes the answer without touching this file.
        #
        # Both forms of caller intent are honoured: an explicit ``dtype=`` is
        # tried first, and ``backend=`` ranks its own kernels highest, so we
        # allocate for the implementation the caller actually asked for.
        activation = dtype or get_engine_config().device.params_dtype
        # ``dtype=None`` has always meant "torch's default dtype", so that is
        # what gets requested when the caller does not say. The activation dtype
        # is still what the *contracts* are evaluated against, since that is
        # what will actually flow through the kernel.
        requested = dtype_name(
            dtype if dtype is not None else torch.get_default_dtype()
        )
        chosen = param_dtypes(
            "layernorm",
            activation=activation,
            known={"attrs.bias": bias},
            preferred={"weight": requested, "bias": requested},
            prefer=self._prefer,
        )
        param_dtype = torch_dtype(chosen["weight"])
        beta_dtype = torch_dtype(chosen.get("bias", chosen["weight"]))

        weight = torch.ones(hidden_size, dtype=param_dtype, device=device)
        if elementwise_affine:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.register_buffer("weight", weight, persistent=False)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(hidden_size, dtype=beta_dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
            # FlashInfer's kernel always reads beta, so a no-bias layer needs a
            # zero buffer for its add to become a no-op. Allocating it
            # unconditionally costs one vector and removes a branch that used
            # to depend on which backend had been guessed at construction.
            self.register_buffer(
                "_zero_beta",
                torch.zeros(hidden_size, dtype=beta_dtype, device=device),
                persistent=False,
            )

        if prefix and elementwise_affine:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()
            if bias:
                self.bias.hf_keys = [(f"{prefix}.bias", None)]
                self.bias.weight_loader = replicated()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.bias.data if self.has_bias else self._zero_beta
        handle = self._call.select(
            device=x.device,
            dtype={
                "input": x.dtype,
                "weight": self.weight.dtype,
                "bias": beta.dtype,
                "output": x.dtype,
            },
            dims=token_shape(x),
        )
        needs_reshape = x.dim() != 2
        if needs_reshape:
            orig_shape = x.shape
            x = x.contiguous().reshape(-1, orig_shape[-1])
        out = handle.execute(x, self.weight.data, beta, self.variance_epsilon)
        return out.reshape(orig_shape) if needs_reshape else out

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, eps={self.variance_epsilon}, "
            f"bias={self.has_bias}, elementwise_affine={self.elementwise_affine}, "
            f"backend={self.backend!r}"
        )


# --------------------------------------------------------------------------- #
# AdaRMSNorm — adaptive RMSNorm with (scale, shift, gate) conditioning.
# --------------------------------------------------------------------------- #


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm with conditional ``(scale, shift, gate)`` modulation.

    Forward signature ``forward(x, cond) -> (out, gate)``::

        modulation = self.dense(cond)         # (..., 3 * D)
        normed     = x * rsqrt(mean(x^2)+eps) # fp32 reduction
        scale, shift, gate = chunk(modulation, 3, dim=-1)
        out  = (normed * (1 + scale) + shift).to(x.dtype)
        gate = gate.to(x.dtype)

    The ``(1 + weight)`` term of the standard ``(1 + w)`` RMSNorm is
    *replaced* by ``(1 + scale)`` from the conditioning projection; there
    is no learned ``weight`` parameter on this class. ``self.dense.weight`` and
    ``self.dense.bias`` are zero-initialised so a freshly constructed
    AdaRMSNorm is the identity (``scale=0``, ``shift=0``, ``gate=0``).

    Used by adaptive-norm decoder layers (``use_adarms=True``); other
    decoder variants typically use plain :class:`GemmaRMSNorm`.

    Parameters
    ----------
    hidden_size:
        Last dim ``D`` of the input. Modulation projection produces
        ``3 * D`` channels.
    cond_dim:
        Width of the conditioning vector ``cond``. The dense projection is
        a :class:`ReplicatedLinear(cond_dim, 3 * hidden_size, bias=True)` —
        every rank holds the full weight, no collectives — so the AdaRMS
        modulation matches the (replicated) per-token ``cond`` it conditions
        on without an extra all-gather.
    eps:
        Numerical-stability epsilon for the variance reduction.
    backend:
        ``"phyai-kernel"`` (default — Triton on CUDA) or ``"torch"``
        (eager fp32 fallback for CPU / MPS / non-CUDA hosts and for
        ``torch.compile`` integration). flashinfer has no AdaRMS kernel
        and is rejected at construction time.
    dtype:
        Optional dtype for ``self.dense``'s parameters. Defaults to the
        global default dtype.
    prefix:
        Dotted state-dict prefix for placement loading.

    Forward
    -------
    * 3-D ``x`` ``(B, S, D)`` with 2-D ``cond`` ``(B, cond_dim)``: the
      modulation is unsqueezed to ``(B, 1, 3D)`` and broadcast across the
      sequence axis. ``gate`` comes back shaped ``(B, 1, D)`` so the
      caller's ``residual + out * gate`` broadcasts correctly.
    * Same-rank ``x`` and ``cond``: 1:1 per-row mapping. ``gate`` shape
      mirrors ``cond`` (``(N, D)``).

    The Triton kernel handles both shapes via a ``group_size`` derived
    from ``prod(x.shape[:-1]) / prod(modulation.shape[:-1])``; the torch
    backend just does the arithmetic broadcast.

    The op is stateless: it either projects a ``cond`` it is handed, or
    applies a ``modulation`` it is handed — exactly one per call. When the
    conditioning is drawn from a small, fixed, input-independent set, a
    caller can project them all once with :meth:`project_modulation` (a
    pure helper that stores nothing) and later pass a single
    ``(1, 3 * hidden_size)`` row back via ``forward(x, modulation=...)``;
    the kernel broadcasts that row across all rows of ``x``. This keeps the
    ``self.dense`` projection out of a captured graph that replays the same
    schedule many times, without the op holding any per-call cache.
    """

    def __init__(
        self,
        hidden_size: int,
        cond_dim: int,
        eps: float = 1e-6,
        backend: str | None = None,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
        kernel_role: str = "adarmsnorm",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}.")
        self.backend = backend
        self._prefer = backend_preference("adarmsnorm", backend)
        self._call = CallSite(
            "adarmsnorm",
            role=kernel_role,
            prefer=self._prefer,
            dims={"hidden": hidden_size, "cond_dim": cond_dim},
        )
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.variance_epsilon = eps
        self.prefix = prefix
        self.kernel_role = kernel_role
        if device is None:
            device = get_engine_config().device.target

        # ReplicatedLinear allocates ``weight`` empty (Bf16Spec) and ``bias``
        # zero; we then zero the weight as well so a freshly constructed
        # AdaRMSNorm is the identity (scale=0, shift=0, gate=0). Loaders
        # auto-attach when ``prefix`` is non-empty.
        self.dense = ReplicatedLinear(
            cond_dim,
            3 * hidden_size,
            bias=True,
            params_dtype=dtype,
            device=device,
            prefix=f"{prefix}.dense" if prefix else "",
        )
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def project_modulation(self, conds: torch.Tensor) -> torch.Tensor:
        """Project a fixed set of conditioning rows to their modulation.

        Runs ``conds`` ``(K, cond_dim)`` through ``self.dense`` once and
        returns the projected ``(K, 3 * hidden_size)`` table. This is a
        **pure** helper — it stores nothing on the module. The caller owns
        the returned table and feeds individual rows back through
        ``forward(x, modulation=row)``; that is the path used when the
        conditioning is a small, fixed, input-independent set (so the
        projections are constants) and the forward runs many times — e.g.
        inside a captured graph that would otherwise replay the projection
        on every call.

        Call after the real ``dense`` weights are loaded. Because the op
        holds no cache, there is no stale-cache hazard: re-project whenever
        the weights or the conditioning set change.
        """
        if conds.shape[-1] != self.cond_dim:
            raise ValueError(
                f"AdaRMSNorm.project_modulation: conds last dim "
                f"{conds.shape[-1]} does not match cond_dim={self.cond_dim}."
            )
        with torch.no_grad():
            return self.dense(conds)[0].contiguous()

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        modulation: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"AdaRMSNorm: x last dim {x.shape[-1]} does not match "
                f"hidden_size={self.hidden_size}."
            )
        if (cond is None) == (modulation is None):
            raise ValueError(
                "AdaRMSNorm.forward: provide exactly one of `cond` or `modulation`."
            )

        if modulation is None:
            # Project the per-token condition through ``self.dense``.
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(
                    f"AdaRMSNorm: cond last dim {cond.shape[-1]} does not match "
                    f"cond_dim={self.cond_dim}."
                )
            modulation, _ = self.dense(cond)
        else:
            # Use the caller's already-projected modulation (e.g. one row of
            # a :meth:`project_modulation` table). A single ``(1, 3D)`` row
            # broadcasts across all rows of ``x`` via the kernel's leading-dim
            # ratio; a ``(B, 3D)`` modulation maps per batch.
            if modulation.shape[-1] != 3 * self.hidden_size:
                raise ValueError(
                    f"AdaRMSNorm: modulation last dim {modulation.shape[-1]} "
                    f"does not match 3 * hidden_size={3 * self.hidden_size}."
                )

        # When ``x`` is 3-D ``(B, S, D)`` and ``modulation`` is 2-D
        # ``(B, 3D)``, broadcast the modulation across the sequence axis.
        if x.dim() == 3 and modulation.dim() == 2:
            modulation = modulation.unsqueeze(1)

        handle = self._call.select(
            device=x.device,
            dtype={
                "input": x.dtype,
                "modulation": modulation.dtype,
                "output": x.dtype,
            },
            dims=token_shape(x),
        )
        return handle.execute(x, modulation, self.variance_epsilon)

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, cond_dim={self.cond_dim}, "
            f"eps={self.variance_epsilon}, backend={self.backend!r}"
        )


__all__ = ["AdaRMSNorm", "GatedRMSNorm", "GemmaRMSNorm", "LayerNorm", "RMSNorm"]
