"""TransformerBlock — unified pre-norm / sandwich-norm transformer block.

One class for three attention flavors, chosen by the ``attn_kind``
argument:

* ``attn_kind="attention"`` (default) -> :class:`Attention`. No KV
  cache. Suitable for SigLIP-style vision encoders or any prefill-only
  path. Forward takes ``cu_seqlens_q`` / ``cu_seqlens_kv`` for ragged
  input or builds a default ctx from the q/k layout.
* ``attn_kind="ar"`` (requires ``layer_idx: int``) -> :class:`ARAttention`
  bound to that ``layer_id``. LM-side paged attention. Forward expects
  an ``attn_ctx`` from the runner; K/V get scattered into
  ``attn_ctx.kv_pool`` at ``attn_ctx.write_indices``.
* ``attn_kind="diffusion"`` (requires ``layer_idx: int``) ->
  :class:`DiffusionAttention` bound to that ``layer_id``. Action-expert
  / diffusion paged attention. Same forward shape as ``"ar"``.

Two normalisation topologies, optional Q/K head-dim norm:

* **Pre-norm** (Llama / Qwen2 / Qwen2.5 / Qwen3 / Mistral / Phi3 /
  SigLIP encoder)::

      h = x + attn(input_norm(x))
      y = h + mlp(pre_ff_norm(h))

* **Sandwich norm** (``sandwich_norm=True``; Gemma2 / Gemma3)::

      h = x + post_attn_norm(attn(input_norm(x)))
      y = h + post_ff_norm(mlp(pre_ff_norm(h)))

When ``attn_qk_norm=True`` the block additionally normalises Q and K
on the per-head ``head_dim`` axis after the QKV projection and before
RoPE — used by Gemma3 (gemma-style ``(1+w)`` RMS) and Qwen3 (standard
RMS).

Branch-free forward
-------------------
Every optional knob is resolved at __init__ into a concrete module slot
(:class:`torch.nn.Identity` stand-ins for the disabled cases) plus a
bound :attr:`_attn_forward` method pointer that captures the differing
:class:`Attention` / :class:`ARAttention` / :class:`DiffusionAttention`
call signatures. ``forward()`` itself contains zero ``if`` statements.

Knob matrix
-----------

================  ========================================================
group             knobs
================  ========================================================
mode              ``attn_kind`` (``"attention"`` / ``"ar"`` /
                  ``"diffusion"``); ``layer_idx`` (optional stack-position
                  metadata for ``"attention"`` — ignored; required for the
                  other two)
norm              ``norm_type`` (rmsnorm / gemma_rmsnorm / layernorm),
                  ``norm_eps``, ``norm_bias`` (LN only), ``norm_backend``,
                  ``sandwich_norm`` (off -> 2 norms; on -> 4 norms)
attention         ``num_heads``, ``num_kv_heads``, ``head_dim``,
                  ``attn_causal``, ``attn_sliding_window``
                  (``"attention"`` only),
                  ``attn_logits_soft_cap`` (``"attention"`` only),
                  ``attn_scale``,
                  ``attn_bias`` (q/k/v), ``attn_out_bias`` (o-proj),
                  ``attn_qk_norm`` (per-head Q/K norm), ``attn_backend``
RoPE              ``rope`` (a :class:`RotaryEmbedding` instance shared
                  across layers, or ``None`` for vision encoders / when
                  positions don't apply); ``precompute_rope`` (Pattern A
                  vs Pattern B — see "RoPE patterns" below)
MLP               ``intermediate_size``, ``mlp_gated``, ``mlp_activation``,
                  ``mlp_bias``
TP                ``axis`` / ``sp_axis`` / ``mesh``
quant             ``spec_qkv`` / ``spec_o`` / ``spec_mlp_in`` /
                  ``spec_mlp_out`` (per-leg :class:`WeightSpec`)
HF naming         **required**: ``norm_hf_names`` (per-position),
                  ``attn_out_hf_name`` (Llama/Gemma ``"o_proj"`` vs SigLIP
                  ``"out_proj"``); optional: ``attn_qkv_hf_names``,
                  ``mlp_gated_hf_names``
================  ========================================================

The block is a **structural primitive** — HF naming conventions belong
to the model that uses it. The two highly-divergent knobs are required
arguments; the two truly-universal-ish ones (Q/K/V, gated MLP) keep
sensible defaults that match every modern decoder phyai targets (and
SigLIP, for the QKV side).

Forward
-------
``forward(x, *, positions=None, attn_ctx=None, cu_seqlens_q=None, cu_seqlens_kv=None, cos=None, sin=None) -> y``

* ``x``: ``(B, S, hidden_size)`` for padded batches, ``(nnz, hidden_size)``
  for ragged. Output preserves the leading shape.
* ``positions``: required in **Pattern A** (``precompute_rope=False``,
  the default) — ``(B, S)`` / ``(S,)`` for padded, ``(nnz,)`` for ragged.
  Ignored in Pattern B.
* ``cos`` / ``sin``: required in **Pattern B**
  (``precompute_rope=True``) — pre-gathered cos/sin tensors from
  :meth:`RotaryEmbedding.get_cos_sin`. Ignored in Pattern A.
* ``attn_ctx``: required for ``attn_kind="ar"`` / ``"diffusion"``
  (the runner builds the right ctx type per stack); optional for
  ``attn_kind="attention"`` (:class:`Attention` builds a default ctx
  if absent).
* ``cu_seqlens_q`` / ``cu_seqlens_kv``: int32 ``(B+1,)`` for ragged
  input in ``attn_kind="attention"``. Ignored in the paged kinds (the
  paged attention reads cu_seqlens off the runner-built ``attn_ctx``).

RoPE patterns
-------------
* **Pattern A** (``precompute_rope=False``, default): per-layer fused
  gather-and-rotate via :meth:`RotaryEmbedding.forward(positions, q, k)`.
  Argument order is ``(positions, q, k)`` to match the kernel's
  position-then-rotation contract. The flashinfer kernel does the
  position -> cos/sin lookup in-kernel.
* **Pattern B** (``precompute_rope=True``): the caller (typically the
  stack) calls :meth:`RotaryEmbedding.get_cos_sin(positions)` once,
  then threads the resulting ``(cos, sin)`` to every layer's forward as
  ``cos`` / ``sin``. Each layer skips the cache gather and
  runs only the rotation via :meth:`RotaryEmbedding.apply`.
  Helpful for cuda-graph capture (one less kernel inside the captured
  region per layer) and deep stacks where the gather appears in profiles.

The pattern is fixed at construction; :meth:`forward` dispatches via a
bound :attr:`_apply_rope` method pointer with zero ``if`` on rope
configuration.

Norm-position keys
------------------

================  ===============================  ===============================
position           pre-norm                          sandwich norm
================  ===============================  ===============================
``input_norm``     ✓                                ✓
``post_attn_norm`` —                                ✓
``pre_ff_norm``    ✓                                ✓
``post_ff_norm``   —                                ✓
================  ===============================  ===============================

``norm_hf_names`` must contain *exactly* the keys for the chosen
topology (``{input_norm, pre_ff_norm}`` for pre-norm,
``{input_norm, post_attn_norm, pre_ff_norm, post_ff_norm}`` for
sandwich) — extra or missing keys are rejected at construction. When
``attn_qk_norm=True``, the q/k norms are HF-named ``q_norm`` / ``k_norm``
inside ``self_attn`` (universal across Gemma3 / Qwen3).

The attention sub-prefix is always ``self_attn`` and the MLP sub-prefix
is always ``mlp``.

Limitations
-----------
* Paged kinds (``"ar"`` / ``"diffusion"``) do not yet accept
  ``attn_sliding_window`` / ``attn_logits_soft_cap`` (raises
  :class:`NotImplementedError` at construction); the paged attention
  classes do not surface them today.
* No append-prefill mode in no-cache mode. Q and K must share token
  count.
* No fused FP8 RoPE / Q/K quant.
* No cross-attention (decoder-only / encoder-only).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal, Mapping

import torch
import torch.nn as nn

from phyai.layers.attention.ar import ARAttention, ARAttnCtx
from phyai.layers.attention.attention import Attention, AttnCtx
from phyai.layers.attention.diffusion import DiffusionAttention, DiffusionAttnCtx
from phyai.layers.layer_norm import GemmaRMSNorm, LayerNorm, RMSNorm
from phyai.layers.linear.layers import (
    QKVParallelLinear,
    RowParallelLinear,
)
from phyai.layers.mlp.dense_mlp import DenseMLP


# HuggingFace de-facto norm naming defaults. Llama / Qwen2 / Qwen3 / Mistral /
# Phi3 / Olmo all use ``input_layernorm`` + ``post_attention_layernorm``;
# Gemma2 / Gemma3 add ``pre_feedforward_layernorm`` + ``post_feedforward_layernorm``
# for the sandwich-norm topology. The mapping is ``phyai_slot -> hf_default``.
# The user-facing override dict is keyed by ``hf_default`` (recognisable to anyone
# coming from a HuggingFace modeling file) and gets resolved to the slot internally.
_HF_NORM_NAMES_PRE: Mapping[str, str] = MappingProxyType(
    {
        "input_norm": "input_layernorm",
        "pre_ff_norm": "post_attention_layernorm",
    }
)
_HF_NORM_NAMES_SANDWICH: Mapping[str, str] = MappingProxyType(
    {
        "input_norm": "input_layernorm",
        "post_attn_norm": "post_attention_layernorm",
        "pre_ff_norm": "pre_feedforward_layernorm",
        "post_ff_norm": "post_feedforward_layernorm",
    }
)

_VALID_NORM_TYPES: tuple[str, ...] = ("rmsnorm", "gemma_rmsnorm", "layernorm")


def _make_norm(
    norm_type: str,
    hidden_size: int,
    eps: float,
    *,
    bias: bool,
    backend: str,
    dtype: torch.dtype | None,
    prefix: str,
) -> nn.Module:
    if norm_type == "rmsnorm":
        return RMSNorm(
            hidden_size, eps=eps, backend=backend, dtype=dtype, prefix=prefix
        )
    if norm_type == "gemma_rmsnorm":
        return GemmaRMSNorm(
            hidden_size, eps=eps, backend=backend, dtype=dtype, prefix=prefix
        )
    if norm_type == "layernorm":
        return LayerNorm(
            hidden_size,
            eps=eps,
            backend=backend,
            bias=bias,
            dtype=dtype,
            prefix=prefix,
        )
    raise ValueError(
        f"Unknown norm_type {norm_type!r}; expected one of {_VALID_NORM_TYPES!r}."
    )


def _validate_norm_hf_names(
    overrides: Mapping[str, str] | None, sandwich: bool
) -> dict[str, str]:
    """Resolve the actual HF source name for each norm slot.

    ``overrides`` is keyed by **HuggingFace default names**
    (``"input_layernorm"`` etc.) — the same names users see in any
    HF modeling file. A missing entry uses the HF default itself.
    ``None`` means use HF defaults for everything (covers Llama /
    Qwen / Gemma / Mistral / Phi3 — the common case).

    Returns a ``{phyai_slot -> actual_hf_source_name}`` map for
    internal use by :func:`_norm_prefix_for`.
    """
    defaults = _HF_NORM_NAMES_SANDWICH if sandwich else _HF_NORM_NAMES_PRE
    out = dict(defaults)
    if overrides is None:
        return out
    hf_to_slot = {hf_default: slot for slot, hf_default in defaults.items()}
    unknown = set(overrides) - set(hf_to_slot)
    if unknown:
        topo = "sandwich-norm" if sandwich else "pre-norm"
        raise ValueError(
            f"norm_hf_names has unknown keys {sorted(unknown)!r} for "
            f"{topo} topology; expected subset of "
            f"{sorted(hf_to_slot.keys())!r}."
        )
    for hf_default, actual in overrides.items():
        out[hf_to_slot[hf_default]] = actual
    return out


def _norm_prefix_for(prefix: str, position: str, names: Mapping[str, str]) -> str:
    own = names[position]
    return f"{prefix}.{own}" if prefix else own


def _split_qkv(
    fused: torch.Tensor,
    leading: torch.Size,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused QKV projection into ``(q, k, v)`` shaped ``(..., heads, head_dim)``."""
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    q, k, v = fused.split([q_dim, kv_dim, kv_dim], dim=-1)
    q = q.reshape(*leading, q_heads, head_dim)
    k = k.reshape(*leading, kv_heads, head_dim)
    v = v.reshape(*leading, kv_heads, head_dim)
    return q, k, v


class TransformerBlock(nn.Module):
    """Configurable transformer block with optional KV cache.

    See module docstring for the full topology and knob list. The class
    composes existing phyai primitives:
    :class:`QKVParallelLinear` ->
    (optional Q/K norm) ->
    (optional :class:`RotaryEmbedding`, two patterns) ->
    :class:`Attention` (when ``attn_kind="attention"``),
    :class:`ARAttention` (when ``attn_kind="ar"``), or
    :class:`DiffusionAttention` (when ``attn_kind="diffusion"``) ->
    :class:`RowParallelLinear` (output proj) ->
    :class:`DenseMLP`, with norms inserted per the chosen topology.

    Two RoPE patterns, both branchless in :meth:`forward`:

    * **Pattern A** (default, ``precompute_rope=False``): per-layer fused
      gather-and-rotate via :meth:`RotaryEmbedding.forward`. Forward
      caller passes ``positions`` only.
    * **Pattern B** (``precompute_rope=True``): caller pre-computes
      ``(cos, sin)`` once at the top of the layer loop via
      :meth:`RotaryEmbedding.get_cos_sin` and threads the tensors
      through every layer's forward as ``cos`` / ``sin``;
      the layer applies them via :meth:`RotaryEmbedding.apply`,
      skipping the per-layer cache gather. Useful for cuda-graph capture
      (one less kernel inside the captured region per layer) and deep
      stacks where the position lookup shows up in profiles.

    The pattern is fixed at construction; :meth:`forward` dispatches via
    a bound method pointer (zero ``if`` on configuration).
    """

    def __init__(
        self,
        # ---- Core dims -------------------------------------------------- #
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        *,
        # ---- Attention kind toggle ------------------------------------- #
        attn_kind: Literal["attention", "ar", "diffusion"] = "attention",
        layer_idx: int | None = None,
        # ---- HF naming (defaults match Llama / Qwen / Gemma conventions) ---- #
        norm_hf_names: Mapping[str, str] | None = None,
        attn_out_hf_name: str = "o_proj",
        # ---- Attention dims --------------------------------------------- #
        num_kv_heads: int | None = None,
        head_dim: int | None = None,
        # ---- Topology --------------------------------------------------- #
        sandwich_norm: bool = False,
        # ---- Attention behaviour --------------------------------------- #
        attn_causal: bool = True,
        attn_sliding_window: int | None = None,
        attn_logits_soft_cap: float | None = None,
        attn_scale: float | None = None,
        attn_bias: bool = False,
        attn_out_bias: bool | None = None,
        attn_qk_norm: bool = False,
        attn_backend: str = "flashinfer",
        attn_backend_kwargs: dict[str, Any] | None = None,
        # ---- RoPE ------------------------------------------------------- #
        rope: nn.Module | None = None,
        precompute_rope: bool = False,
        # ---- MLP -------------------------------------------------------- #
        mlp_gated: bool = True,
        mlp_activation: str = "silu",
        mlp_bias: bool = False,
        # ---- Norm ------------------------------------------------------- #
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-6,
        norm_bias: bool = False,
        norm_backend: str = "flashinfer",
        # ---- TP / mesh -------------------------------------------------- #
        axis: str = "tp",
        sp_axis: str | None = None,
        mesh: str = "model",
        # ---- Quant specs ------------------------------------------------ #
        spec_qkv: object | None = None,
        spec_o: object | None = None,
        spec_mlp_in: object | None = None,
        spec_mlp_out: object | None = None,
        # ---- Optional HF naming overrides (defaults are universal) ----- #
        attn_qkv_hf_names: Mapping[str, str] | None = None,
        mlp_gated_hf_names: tuple[str, str] = ("gate_proj", "up_proj"),
        # ---- Misc ------------------------------------------------------- #
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        if norm_type not in _VALID_NORM_TYPES:
            raise ValueError(
                f"Unknown norm_type {norm_type!r}; expected one of "
                f"{_VALID_NORM_TYPES!r}."
            )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if head_dim is None:
            if hidden_size % num_heads != 0:
                raise ValueError(
                    f"hidden_size={hidden_size} not divisible by "
                    f"num_heads={num_heads}; pass head_dim explicitly."
                )
            head_dim = hidden_size // num_heads
        if attn_out_bias is None:
            attn_out_bias = attn_bias

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.sandwich_norm = sandwich_norm
        self.norm_type = norm_type
        self.attn_qk_norm_enabled = attn_qk_norm
        self.layer_idx = layer_idx
        self.attn_kind = attn_kind
        self.prefix = prefix

        # Naming knobs. ``norm_hf_names=None`` uses HF defaults for the
        # chosen topology (covers Llama / Qwen / Gemma / Mistral / Phi3).
        # When passed, keys are HF default names (``"input_layernorm"``
        # etc.) so the override dict reads naturally; values are the
        # actual HF source names in this checkpoint.
        self.attn_out_hf_name = attn_out_hf_name
        self.attn_qkv_hf_names = (
            dict(attn_qkv_hf_names) if attn_qkv_hf_names is not None else None
        )
        self.mlp_gated_hf_names = tuple(mlp_gated_hf_names)
        self._norm_hf_names = _validate_norm_hf_names(norm_hf_names, sandwich_norm)

        # Sub-prefixes
        attn_prefix = f"{prefix}.self_attn" if prefix else "self_attn"
        mlp_prefix = f"{prefix}.mlp" if prefix else "mlp"

        norm_kwargs: dict[str, Any] = dict(
            hidden_size=hidden_size,
            eps=norm_eps,
            bias=norm_bias,
            backend=norm_backend,
            dtype=params_dtype,
        )
        self.input_norm = _make_norm(
            norm_type,
            prefix=_norm_prefix_for(prefix, "input_norm", self._norm_hf_names),
            **norm_kwargs,
        )
        self.pre_ff_norm = _make_norm(
            norm_type,
            prefix=_norm_prefix_for(prefix, "pre_ff_norm", self._norm_hf_names),
            **norm_kwargs,
        )
        if sandwich_norm:
            self.post_attn_norm = _make_norm(
                norm_type,
                prefix=_norm_prefix_for(prefix, "post_attn_norm", self._norm_hf_names),
                **norm_kwargs,
            )
            self.post_ff_norm = _make_norm(
                norm_type,
                prefix=_norm_prefix_for(prefix, "post_ff_norm", self._norm_hf_names),
                **norm_kwargs,
            )
        else:
            # Identity stand-ins so forward never branches on ``sandwich_norm``.
            self.post_attn_norm = nn.Identity()
            self.post_ff_norm = nn.Identity()

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            axis=axis,
            sp_axis=sp_axis,
            gather_output=False,
            bias=attn_bias,
            params_dtype=params_dtype,
            spec=spec_qkv,
            hf_legs=self.attn_qkv_hf_names,
            mesh=mesh,
            prefix=f"{attn_prefix}.qkv_proj",
        )
        tp_size = self.qkv_proj.tp_size
        self.q_heads_local = num_heads // tp_size
        self.kv_heads_local = max(1, num_kv_heads // tp_size)

        # Optional per-head Q/K norm (Gemma3 / Qwen3). Operates on
        # ``head_dim`` (each Q/K head normalised independently). Names
        # ``q_norm`` / ``k_norm`` are universal across the families that
        # use this hook today. Identity stand-ins when disabled.
        if attn_qk_norm:
            self.q_norm = _make_norm(
                norm_type,
                hidden_size=head_dim,
                eps=norm_eps,
                bias=norm_bias,
                backend=norm_backend,
                dtype=params_dtype,
                prefix=f"{attn_prefix}.q_norm",
            )
            self.k_norm = _make_norm(
                norm_type,
                hidden_size=head_dim,
                eps=norm_eps,
                bias=norm_bias,
                backend=norm_backend,
                dtype=params_dtype,
                prefix=f"{attn_prefix}.k_norm",
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # RoPE — three modes wired at __init__ via a bound dispatcher
        # method pointer. The hot path calls ``self._apply_rope(...)`` —
        # zero ``if`` on rope configuration. ``self._rope_input_kind``
        # picks the input-validation branch in :meth:`forward`.
        #
        # * ``rope=None``                       -> passthrough (no rotation)
        # * ``precompute_rope=False`` (default) -> Pattern A: fused per-layer
        # * ``precompute_rope=True``            -> Pattern B: caller-supplied cos/sin
        if rope is None:
            self.rope = None
            self._apply_rope = self._apply_rope_passthrough
            self._rope_input_kind = "none"
        elif precompute_rope:
            self.rope = rope
            self._apply_rope = self._apply_rope_precomputed
            self._rope_input_kind = "cos_sin"
        else:
            self.rope = rope
            self._apply_rope = self._apply_rope_per_layer
            self._rope_input_kind = "positions"

        # Attention sub-module + bound forward-method pointer. The
        # method-pointer write at __init__ replaces a runtime ``if`` in
        # the hot path with a single attribute lookup.
        if attn_kind == "attention":
            # layer_idx is optional here: the no-cache Attention has no KV-pool
            # slot to address, so it is kept only as stack-position metadata
            # (surfaced in extra_repr) and not passed on. The paged kinds below
            # require it.
            self.attn = Attention(
                num_heads=self.q_heads_local,
                head_dim=head_dim,
                num_kv_heads=self.kv_heads_local,
                scale=attn_scale,
                causal=attn_causal,
                sliding_window=attn_sliding_window,
                logits_soft_cap=attn_logits_soft_cap,
                backend=attn_backend,
                backend_kwargs=attn_backend_kwargs,
            )
            self._attn_forward = self._attn_forward_attention
        elif attn_kind == "ar":
            if layer_idx is None:
                raise ValueError("attn_kind='ar' requires layer_idx: int.")
            if attn_sliding_window is not None or attn_logits_soft_cap is not None:
                raise NotImplementedError(
                    "attn_sliding_window / attn_logits_soft_cap are not yet "
                    "supported in the AR (paged) path; ARAttention does not "
                    "accept them today."
                )
            self.attn = ARAttention(
                num_heads=self.q_heads_local,
                head_dim=head_dim,
                layer_id=layer_idx,
                num_kv_heads=self.kv_heads_local,
                scale=attn_scale,
                causal=attn_causal,
                backend=attn_backend,
                backend_kwargs=attn_backend_kwargs,
            )
            self._attn_forward = self._attn_forward_ar
        elif attn_kind == "diffusion":
            if layer_idx is None:
                raise ValueError("attn_kind='diffusion' requires layer_idx: int.")
            if attn_sliding_window is not None or attn_logits_soft_cap is not None:
                raise NotImplementedError(
                    "attn_sliding_window / attn_logits_soft_cap are not yet "
                    "supported in the diffusion (paged) path; "
                    "DiffusionAttention does not accept them today."
                )
            self.attn = DiffusionAttention(
                num_heads=self.q_heads_local,
                head_dim=head_dim,
                layer_id=layer_idx,
                num_kv_heads=self.kv_heads_local,
                scale=attn_scale,
                causal=attn_causal,
                backend=attn_backend,
                backend_kwargs=attn_backend_kwargs,
            )
            self._attn_forward = self._attn_forward_diffusion
        else:
            raise ValueError(
                f"Unknown attn_kind {attn_kind!r}; expected one of "
                f"'attention', 'ar', 'diffusion'."
            )

        self.o_proj = RowParallelLinear(
            in_features=num_heads * head_dim,
            out_features=hidden_size,
            axis=axis,
            sp_axis=sp_axis,
            input_is_parallel=True,
            reduce_results=True,
            bias=attn_out_bias,
            params_dtype=params_dtype,
            spec=spec_o,
            mesh=mesh,
            prefix=f"{attn_prefix}.{attn_out_hf_name}",
        )

        self.mlp = DenseMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=mlp_activation,
            gated=mlp_gated,
            bias=mlp_bias,
            axis=axis,
            sp_axis=sp_axis,
            params_dtype=params_dtype,
            spec_in=spec_mlp_in,
            spec_out=spec_mlp_out,
            gated_hf_legs=self.mlp_gated_hf_names,
            mesh=mesh,
            prefix=mlp_prefix,
        )

    def _attn_forward_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_ctx: Any,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_kv: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.attn(
            q,
            k,
            v,
            ctx=attn_ctx,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
        )

    def _attn_forward_ar(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_ctx: Any,
        cu_seqlens_q: torch.Tensor | None,  # noqa: ARG002 — interface match
        cu_seqlens_kv: torch.Tensor | None,  # noqa: ARG002 — interface match
    ) -> torch.Tensor:
        return self.attn(q, k, v, attn_ctx)

    def _attn_forward_diffusion(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_ctx: Any,
        cu_seqlens_q: torch.Tensor | None,  # noqa: ARG002 — interface match
        cu_seqlens_kv: torch.Tensor | None,  # noqa: ARG002 — interface match
    ) -> torch.Tensor:
        return self.attn(q, k, v, attn_ctx)

    def _apply_rope_passthrough(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor | None,  # noqa: ARG002 — interface match
        cos: torch.Tensor | None,  # noqa: ARG002 — interface match
        sin: torch.Tensor | None,  # noqa: ARG002 — interface match
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def _apply_rope_per_layer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor | None,
        cos: torch.Tensor | None,  # noqa: ARG002 — interface match
        sin: torch.Tensor | None,  # noqa: ARG002 — interface match
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pattern A: per-layer fused gather-and-rotate.
        return self.rope(positions, q, k)

    def _apply_rope_precomputed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor | None,  # noqa: ARG002 — interface match
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pattern B: caller passed pre-gathered cos/sin (one gather per
        # stack, amortised across N layers). Apply rotation only.
        return self.rope.apply(q, k, cos, sin)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        attn_ctx: AttnCtx | ARAttnCtx | DiffusionAttnCtx | None = None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_kv: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Input contract — cheap checks against `self`-stored constants.
        # NOT configuration branches (those are eliminated at __init__
        # via Identity stand-ins + the bound ``self._attn_forward`` /
        # ``self._apply_rope`` pointers).
        if x.dim() not in (2, 3):
            raise ValueError(
                f"x must be 2-D (ragged) or 3-D (padded), got shape {tuple(x.shape)}."
            )
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"x last dim {x.shape[-1]} != hidden_size={self.hidden_size}."
            )
        if self._rope_input_kind == "positions" and positions is None:
            raise ValueError(
                "rope is set (Pattern A) but no positions passed to forward."
            )
        if self._rope_input_kind == "cos_sin" and (cos is None or sin is None):
            raise ValueError(
                "precompute_rope=True (Pattern B) but cos / sin "
                "missing from forward kwargs."
            )

        residual = x
        h = self.input_norm(x)
        fused, _ = self.qkv_proj(h)
        q, k, v = _split_qkv(
            fused,
            x.shape[:-1],
            self.q_heads_local,
            self.kv_heads_local,
            self.head_dim,
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self._apply_rope(q, k, positions, cos, sin)
        attn_out = self._attn_forward(q, k, v, attn_ctx, cu_seqlens_q, cu_seqlens_kv)
        attn_flat = attn_out.reshape(
            *attn_out.shape[:-2], self.q_heads_local * self.head_dim
        )
        out, _ = self.o_proj(attn_flat)
        out = self.post_attn_norm(out)
        h = residual + out

        residual = h
        m = self.pre_ff_norm(h)
        m = self.mlp(m)
        m = self.post_ff_norm(m)
        return residual + m

    def extra_repr(self) -> str:
        s = (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"intermediate_size={self.intermediate_size}, "
            f"norm_type={self.norm_type!r}, sandwich_norm={self.sandwich_norm}, "
            f"attn_kind={self.attn_kind!r}"
        )
        if self.layer_idx is not None:
            s += f", layer_idx={self.layer_idx}"
        if self.attn_qk_norm_enabled:
            s += ", attn_qk_norm=True"
        return s


__all__ = ["TransformerBlock"]
