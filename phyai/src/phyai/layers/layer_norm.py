"""RMSNorm + LayerNorm + AdaRMSNorm with selectable kernel backends.

Three related modules in this file:

* :class:`RMSNorm` — standard RMSNorm; :class:`GemmaRMSNorm` for the
  ``(1 + w)`` variant. Used by RMSNorm-based text decoders.
* :class:`LayerNorm` — standard mean/variance LayerNorm with optional
  bias. Used by ViT-style vision encoders.
* :class:`AdaRMSNorm` — adaptive RMSNorm with a learned conditioning
  projection. Replaces the ``(1 + w)`` affine with ``(1 + scale)`` and
  ``+ shift`` from a per-token ``cond`` vector, and exposes a ``gate``
  output for the surrounding gated-residual. Used by adaptive-norm
  variants of decoder layers (``use_adarms=True``).

Backend selection (constructor ``backend=``):

* :class:`RMSNorm` / :class:`GemmaRMSNorm` / :class:`LayerNorm`:
  ``"flashinfer"`` (default) or ``"phyai-kernel"``.
* :class:`AdaRMSNorm`: ``"phyai-kernel"`` (default, Triton on CUDA) or
  ``"torch"`` (eager fallback for CPU / MPS / non-CUDA). flashinfer has
  no AdaRMS kernel, so that backend is rejected.

Reductions and the affine multiply run in fp32 on every backend; output
is cast back to ``x.dtype``. RMSNorm's ``forward`` accepts an optional
``residual`` for the fused ``residual += x; rmsnorm(residual)`` path used
between attention and the MLP in most decoder blocks. LayerNorm has no
fused-add path today (encoder paths don't need one). AdaRMSNorm's
``forward(x, cond)`` returns a ``(out, gate)`` tuple.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from phyai.engine_config import get_engine_config
from phyai.layers.linear import ReplicatedLinear
from phyai.weights.shards import replicated

_VALID_BACKENDS: tuple[str, ...] = ("flashinfer", "phyai-kernel")


def list_norm_backends() -> list[str]:
    """Return every registered :class:`RMSNorm` / :class:`LayerNorm` backend name.

    The list is what :class:`~phyai.engine_config.BackendConfig` validates
    against; ``AdaRMSNorm`` has its own narrower set, exposed via
    :func:`list_adarms_backends`.
    """
    return list(_VALID_BACKENDS)


def _resolve_backend(name: str) -> str:
    canonical = name.replace("_", "-").lower()
    if canonical not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown norm backend {name!r}; expected one of {_VALID_BACKENDS!r}."
        )
    return canonical


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
        backend: str = "flashinfer",
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.backend = _resolve_backend(backend)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.prefix = prefix
        if device is None:
            device = get_engine_config().device.target
        self.weight = nn.Parameter(
            self._initial_weight(hidden_size, dtype, device), requires_grad=False
        )
        self._rmsnorm, self._fused_add_rmsnorm = self._load_kernels(self.backend)
        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()

    @staticmethod
    def _load_kernels(backend: str) -> tuple[Callable, Callable]:
        """Return ``(rmsnorm, fused_add_rmsnorm)`` for the chosen backend.

        Both returned callables share the same return contract:
        ``fused_add_rmsnorm(x, residual, weight, eps) -> (x, residual)``.
        flashinfer's CUDA op mutates in place and returns ``None``, so we
        wrap it here once at construction time — the hot path then doesn't
        have to inspect the return value.

        The imports live inside each branch on purpose: picking one backend
        shouldn't drag in the other's package. Subclasses override this to
        swap in a different kernel pair, e.g. the ``(1 + w)`` variant.
        """
        if backend == "flashinfer":
            from flashinfer.norm import (
                fused_add_rmsnorm as _fi_fused_add_rmsnorm,
                rmsnorm,
            )

            def fused_add_rmsnorm(x, residual, weight, eps):
                _fi_fused_add_rmsnorm(x, residual, weight, eps)
                return x, residual

            return rmsnorm, fused_add_rmsnorm
        from phyai_kernel import fused_add_rmsnorm, rmsnorm

        return rmsnorm, fused_add_rmsnorm

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
        if residual is not None:
            # Fused add then norm, in place. Both backends return the
            # ``(x, residual)`` pair — flashinfer's None-return is wrapped at
            # construction time inside :meth:`_load_kernels`.
            return self._fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )

        needs_reshape = x.dim() != 2
        if needs_reshape:
            orig_shape = x.shape
            x = x.contiguous().reshape(-1, orig_shape[-1])
        out = self._rmsnorm(x, self.weight.data, self.variance_epsilon)
        if needs_reshape:
            out = out.reshape(orig_shape)
        return out

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, eps={self.variance_epsilon}, backend={self.backend!r}"
        )


class GemmaRMSNorm(RMSNorm):
    """``(1 + w)`` RMSNorm variant.

    Same wrapping as RMSNorm, just bound to the ``(1 + w)`` kernel pair:
    the multiplier is ``(1 + weight)`` and the weight starts at zero, so
    a freshly constructed module is the identity. Matches the HF
    transformers convention for the ``(1 + w)`` variant.
    """

    @staticmethod
    def _load_kernels(backend: str) -> tuple[Callable, Callable]:
        if backend == "flashinfer":
            from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

            return gemma_rmsnorm, gemma_fused_add_rmsnorm
        from phyai_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm

        return gemma_rmsnorm, gemma_fused_add_rmsnorm

    @staticmethod
    def _initial_weight(
        hidden_size: int,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
    ) -> torch.Tensor:
        # The ``(1 + w)`` kernel multiplies by ``(1 + w)``, so identity is ``w == 0``.
        return torch.zeros(hidden_size, dtype=dtype, device=device)


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
        backend: str = "flashinfer",
        *,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        self.backend = _resolve_backend(backend)
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.has_bias = bias
        self.prefix = prefix
        if device is None:
            device = get_engine_config().device.target

        # flashinfer's CUDA layernorm hard-requires fp32 gamma/beta. Pre-allocate
        # in fp32 once so the hot path skips the per-forward cast. phyai-kernel's
        # Triton path accepts any floating dtype, so honor the caller's ``dtype``.
        param_dtype = torch.float32 if self.backend == "flashinfer" else dtype

        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=param_dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(hidden_size, dtype=param_dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self._layernorm = self._load_kernel(self.backend)
        # Pre-bind the no-bias placeholder for ``beta`` so forward doesn't
        # branch on backend:
        # * flashinfer always reads ``beta`` — feed an fp32 zero buffer so
        #   the kernel's add becomes a no-op;
        # * phyai-kernel accepts ``bias=None`` directly — register ``None``
        #   so the same attribute access works on the hot path.
        if not bias:
            zero_beta = (
                torch.zeros(hidden_size, dtype=torch.float32, device=device)
                if self.backend == "flashinfer"
                else None
            )
            self.register_buffer("_zero_beta", zero_beta, persistent=False)

        if prefix:
            self.weight.hf_keys = [(f"{prefix}.weight", None)]
            self.weight.weight_loader = replicated()
            if bias:
                self.bias.hf_keys = [(f"{prefix}.bias", None)]
                self.bias.weight_loader = replicated()

    @staticmethod
    def _load_kernel(backend: str) -> Callable:
        if backend == "flashinfer":
            from flashinfer.norm import layernorm

            return layernorm
        from phyai_kernel import layernorm

        return layernorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        needs_reshape = x.dim() != 2
        if needs_reshape:
            orig_shape = x.shape
            x = x.contiguous().reshape(-1, orig_shape[-1])

        # ``beta`` is pre-resolved at construction time:
        # * has_bias=True  -> bias.data (resolved per-call so weight loading is reflected)
        # * has_bias=False -> ``_zero_beta`` (fp32 zeros for flashinfer; ``None`` for phyai-kernel)
        beta = self.bias.data if self.has_bias else self._zero_beta
        out = self._layernorm(x, self.weight.data, beta, self.variance_epsilon)

        if needs_reshape:
            out = out.reshape(orig_shape)
        return out

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, eps={self.variance_epsilon}, "
            f"bias={self.has_bias}, backend={self.backend!r}"
        )


# --------------------------------------------------------------------------- #
# AdaRMSNorm — adaptive RMSNorm with (scale, shift, gate) conditioning.
# --------------------------------------------------------------------------- #


_ADARMS_BACKENDS: tuple[str, ...] = ("phyai-kernel", "torch")


def list_adarms_backends() -> list[str]:
    """Return every registered :class:`AdaRMSNorm` backend name."""
    return list(_ADARMS_BACKENDS)


def _resolve_adarms_backend(name: str) -> str:
    canonical = name.replace("_", "-").lower()
    if canonical == "flashinfer":
        raise ValueError(
            "AdaRMSNorm has no flashinfer backend; pick 'phyai-kernel' "
            "(default, Triton CUDA) or 'torch' (eager fallback)."
        )
    if canonical not in _ADARMS_BACKENDS:
        raise ValueError(
            f"Unknown AdaRMSNorm backend {name!r}; expected one of "
            f"{_ADARMS_BACKENDS!r}."
        )
    return canonical


def _torch_adarmsnorm(
    x: torch.Tensor,
    modulation: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager torch reference path used by the ``"torch"`` backend.

    ``modulation`` must already be broadcast-shaped against ``x`` along
    the last dim (the caller's :class:`AdaRMSNorm.forward` handles the
    ``(B, 3D) -> (B, 1, 3D)`` unsqueeze for 3-D ``x``). Reductions and
    the affine run in fp32; gate is cast to ``x.dtype``.
    """
    dtype = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    out = xf * (1.0 + scale.float()) + shift.float()
    return out.to(dtype), gate.to(dtype)


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
        backend: str = "phyai-kernel",
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be positive, got {cond_dim}.")
        self.backend = _resolve_adarms_backend(backend)
        self.hidden_size = hidden_size
        self.cond_dim = cond_dim
        self.variance_epsilon = eps
        self.prefix = prefix
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

        self._adarms_kernel = self._load_kernel(self.backend)

    @staticmethod
    def _load_kernel(backend: str) -> Callable:
        if backend == "phyai-kernel":
            from phyai_kernel import adarmsnorm

            return adarmsnorm
        # ``"torch"`` backend: eager fp32 reference, signature-compatible with
        # the Triton kernel so forward can dispatch through one indirection.
        return _torch_adarmsnorm

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

        return self._adarms_kernel(x, modulation, self.variance_epsilon)

    def extra_repr(self) -> str:
        return (
            f"{self.hidden_size}, cond_dim={self.cond_dim}, "
            f"eps={self.variance_epsilon}, backend={self.backend!r}"
        )


__all__ = ["AdaRMSNorm", "GemmaRMSNorm", "LayerNorm", "RMSNorm"]
