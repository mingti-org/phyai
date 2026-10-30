"""Define cacheless, paged, and GDN attention operations."""

from __future__ import annotations

import inspect
import importlib
from typing import Mapping

from phyai.utils import get_logger
from phyai.kernel.facts import lib, attrs, dtype, shape, device, model, role
from phyai.kernel.opspec import Impl, OpSpec, Priority, returns_instance
from phyai.kernel.policy import PolicyError
from phyai.kernel.registry import Catalog
from phyai.kernel.predicate import same, all_of, any_of, implies

logger = get_logger(__name__)

HALF_FLOATS = frozenset({"bf16", "fp16"})
NVIDIA_FLASHINFER = lib.has("flashinfer") & (device.vendor == "nvidia")


ATTENTION = OpSpec(
    name="attention",
    dims=("tokens", "kv_tokens", "heads", "kv_heads", "head_dim"),
    dtypes=("input",),
    optional_dtypes=("key", "value"),
    attributes=("layout", "causal"),
    optional_attributes=(
        "layer_id",
        "sliding_window",
        "logits_soft_cap",
        "mask_kind",
        "masked",
        "kv_tokens",
        "has_attn_bias",
    ),
    signature="(runner=None, **kwargs) -> AttentionBackend",
    returns=returns_instance(object, constructed_with=("runner",)),
    doc="Cacheless attention; the only stack supporting S_q != S_kv.",
)

ATTENTION_PAGED = OpSpec(
    name="attention_paged",
    dims=("tokens", "heads", "kv_heads", "head_dim"),
    dtypes=("input",),
    attributes=("layout", "causal"),
    optional_attributes=("layer_id", "runner", "batch_size"),
    signature="(runner) -> PagedAttentionBackend",
    returns=returns_instance(object, constructed_with=("runner",)),
    requires_reference=False,
    doc="Paged-KV attention (LM prefix, action experts, AR decode).",
)

ATTENTION_GDN = OpSpec(
    name="attention_gdn",
    dims=("tokens", "heads", "key_heads", "value_heads", "state_heads", "head_dim"),
    dtypes=("input", "key", "value", "a", "b", "a_log", "dt_bias"),
    attributes=("layout",),
    optional_attributes=("layer_id", "use_qk_l2norm", "causal"),
    signature="(runner=None, **kwargs) -> GatedDeltaNetBackend",
    returns=returns_instance(object, constructed_with=("runner",)),
    requires_reference=False,
    doc="Gated DeltaNet linear attention.",
)


# FlashInfer prefill supports a fixed set of head dimensions and half dtypes.
# Declarative masks lower to varlen packing there, which cannot express a
# block-causal `segments` structure, and a scattered `keys` mask under a
# causal layer has no single position renumbering — those combinations stay
# on the dense (sdpa/eager) path by capability, not by a runtime raise.
FLASHINFER_ATTENTION = all_of(
    NVIDIA_FLASHINFER,
    shape.head_dim.in_({64, 128, 256}),
    dtype.input.in_(HALF_FLOATS),
    dtype.key.in_(HALF_FLOATS),
    dtype.value.in_(HALF_FLOATS),
    attrs.has_attn_bias != True,
    attrs.mask_kind != "segments",
    # Spelled as an implication, not a negated conjunction: mask_kind is an
    # optional fact, and a missing optional evaluates vacuously true — under
    # a negation that would flip into a spurious rejection.
    implies(
        attrs.causal,
        any_of(attrs.mask_kind.is_none(), attrs.mask_kind == "lengths"),
    ),
)

# The paged wrapper shares FlashInfer's prefill contract. Gating here keeps an
# unsupported head_dim from selecting fine and dying inside ``wrapper.plan`` —
# a raise cannot drive a fallback.
FLASHINFER_PAGED = all_of(
    NVIDIA_FLASHINFER,
    shape.head_dim.in_({64, 128, 256}),
    dtype.input.in_(HALF_FLOATS),
)

# SDPA supports padded layouts.
SDPA_ATTENTION = attrs.layout == "padded"

# FlashInfer GDN supports the declared NVIDIA generations and dtype contract.
# Activations are bf16-only: the fp16 decode kernel writes a wrong recurrent
# state in flashinfer 0.6.12-0.6.17 (~30% of entries off, while bf16 matches
# the sequential reference exactly), so fp16 calls fall through to FLA.
FLASHINFER_GDN = all_of(
    NVIDIA_FLASHINFER,
    device.arch.family_in({"sm9", "sm10"}),
    dtype.input == "bf16",
    dtype.key == "bf16",
    dtype.value == "bf16",
    dtype.a.in_(HALF_FLOATS),
    dtype.b.in_(HALF_FLOATS),
    dtype.a_log == "fp32",
    dtype.dt_bias.in_({"bf16", "fp32"}),
)


# FlashQLA runs GDN chunked prefill through TileLang fused kernels, measured
# 2-3x over the FLA Triton path on Hopper/Blackwell. It ships no decode
# kernel -- the backend rides FLA's recurrent op for that -- so the row
# needs both libraries installed.
FLASH_QLA_GDN = all_of(
    lib.has("flash_qla"),
    lib.has("fla"),
    device.vendor == "nvidia",
    device.arch.family_in({"sm9", "sm10", "sm12"}),
    # The TileLang kernels are specialized for K == V == 128.
    shape.head_dim == 128,
    dtype.input.in_(HALF_FLOATS),
    dtype.key.in_(HALF_FLOATS),
    dtype.value.in_(HALF_FLOATS),
    dtype.a_log == "fp32",
    dtype.dt_bias.in_({"bf16", "fp32"}),
    # Require matching query, key, and value dtypes.
    same(dtype.input, dtype.key, dtype.value),
)


#: Backend classes constructed by the rows below, as (module, class) pairs.
#: Imports happen inside ``prepare`` so building the catalog stays light.
ATTENTION_FLASHINFER_CLS = (
    "phyai.layers.attention.nocache.backends.flashinfer",
    "FlashInferAttentionBackend",
)
ATTENTION_SDPA_CLS = (
    "phyai.layers.attention.nocache.backends.sdpa",
    "SdpaAttentionBackend",
)
ATTENTION_EAGER_CLS = (
    "phyai.layers.attention.nocache.backends.eager",
    "EagerAttentionBackend",
)
PAGED_FLASHINFER_CLS = (
    "phyai.layers.attention.paged.backends.flashinfer",
    "FlashInferPagedBackend",
)
GDN_FLASHINFER_CLS = (
    "phyai.layers.attention.gdn.backends.flashinfer",
    "FlashInferGatedDeltaNetBackend",
)
GDN_FLA_CLS = (
    "phyai.layers.attention.gdn.backends.fla",
    "FlaGatedDeltaNetBackend",
)
GDN_FLASH_QLA_CLS = (
    "phyai.layers.attention.gdn.backends.flash_qla",
    "FlashQlaGatedDeltaNetBackend",
)

#: Captured ragged wrappers use fixed indptr buffers.
RAGGED_CAPTURE_DEFAULTS: Mapping[str, object] = {"use_cuda_graph": True}


def _backend_prepare(
    backend: tuple[str, str],
    *,
    graph_defaults: Mapping[str, object] | None = None,
    **fixed_params: object,
):
    """Build a prepare function that constructs one backend class."""

    module_path, class_name = backend

    def prepare(facts, params):
        backend_cls = getattr(importlib.import_module(module_path), class_name)

        policy_kwargs = dict(fixed_params)
        policy_kwargs.update(params)
        if graph_defaults is not None and facts.lookup("mode") == "capture":
            for key, value in graph_defaults.items():
                policy_kwargs.setdefault(key, value)

        # The constructor signature is the contract: ``runner`` plus keywords.
        accepted = set(inspect.signature(backend_cls.__init__).parameters)
        accepted -= {"self", "runner"}
        unknown = sorted(set(policy_kwargs) - accepted)
        if unknown:
            # A policy rule targeted this kernel by name, so a parameter the
            # backend cannot accept is a configuration error, not a
            # capability miss to fall back from.
            raise PolicyError(
                f"{class_name} does not accept parameter(s) {unknown}; "
                f"accepted: {sorted(accepted)}"
            )

        def build(runner=None, **overrides):
            # Call-site keywords ride a soft backend preference and may name
            # options of a backend that lost selection; keep only the ones
            # this backend knows, but say so — a silently vanishing kwarg
            # reads as "applied" when it was not. Policy parameters outrank
            # call-site overrides.
            dropped = sorted(set(overrides) - accepted)
            if dropped:
                logger.warning_once(
                    "kernel: %s ignores call-site backend kwargs %s "
                    "(accepted: %s); they likely target a backend that lost "
                    "selection.",
                    class_name,
                    ", ".join(dropped),
                    ", ".join(sorted(accepted)),
                )
            merged = {key: value for key, value in overrides.items() if key in accepted}
            merged.update(policy_kwargs)
            return backend_cls(runner, **merged)

        return build

    prepare.__name__ = f"prepare_{class_name}"
    return prepare


# FlashInfer prefill backends are registered as separate selectable rows.

#: Head dimensions supported by FA3 prefill.
FA3_HEAD_DIMS = frozenset({64, 128, 256})

HOPPER = device.arch.family_in({"sm9"})
BLACKWELL = device.arch.family_in({"sm10", "sm11"})

#: Paged prefill backends and their additional capability predicates.
PAGED_PREFILL_BACKENDS: tuple[tuple[str, object | None], ...] = (
    ("fa2", None),
    ("fa3", all_of(HOPPER, shape.head_dim.in_(FA3_HEAD_DIMS))),
    ("cudnn", None),
    ("trtllm-gen", BLACKWELL),
)

RAGGED_PREFILL_BACKENDS: tuple[tuple[str, object | None], ...] = (
    ("fa2", None),
    ("fa3", all_of(HOPPER, shape.head_dim.in_(FA3_HEAD_DIMS))),
    ("cudnn", None),
    ("cutlass", BLACKWELL),
    ("cute-dsl", None),
)


def _prefill_variants(
    op: str,
    backend: tuple[str, str],
    base: object,
    backends: tuple[tuple[str, object | None], ...],
    *,
    graph_defaults: Mapping[str, object] | None = None,
) -> tuple[Impl, ...]:
    """Build one implementation row per FlashInfer prefill backend."""
    rows = []
    for backend_name, extra in backends:
        when = base if extra is None else all_of(base, extra)
        rows.append(
            Impl(
                kernel_id=f"flashinfer.{op}.{backend_name}",
                op=op,
                priority=Priority.OPTIMIZED + 1,
                when=when,
                prepare=_backend_prepare(
                    backend,
                    graph_defaults=graph_defaults,
                    prefill_backend=backend_name,
                ),
                metadata={"package": "flashinfer", "prefill_backend": backend_name},
            )
        )
    return tuple(rows)


def register(catalog: Catalog) -> None:
    for spec in (ATTENTION, ATTENTION_PAGED, ATTENTION_GDN):
        catalog.register_op(spec)

    catalog.register_many(
        (
            *_prefill_variants(
                "attention",
                ATTENTION_FLASHINFER_CLS,
                FLASHINFER_ATTENTION,
                RAGGED_PREFILL_BACKENDS,
                graph_defaults=RAGGED_CAPTURE_DEFAULTS,
            ),
            *_prefill_variants(
                "attention_paged",
                PAGED_FLASHINFER_CLS,
                FLASHINFER_PAGED,
                PAGED_PREFILL_BACKENDS,
            ),
            # Cacheless attention.
            Impl(
                kernel_id="flashinfer.attention",
                op="attention",
                priority=Priority.OPTIMIZED + 2,
                when=FLASHINFER_ATTENTION,
                prepare=_backend_prepare(
                    ATTENTION_FLASHINFER_CLS, graph_defaults=RAGGED_CAPTURE_DEFAULTS
                ),
                metadata={"package": "flashinfer"},
            ),
            Impl(
                kernel_id="sdpa.attention",
                op="attention",
                priority=Priority.OPTIMIZED,
                when=SDPA_ATTENTION,
                prepare=_backend_prepare(ATTENTION_SDPA_CLS),
                metadata={"package": "torch"},
            ),
            Impl(
                kernel_id="eager.attention",
                op="attention",
                priority=Priority.REFERENCE,
                reference=True,
                when=attrs.layout.is_set(),
                prepare=_backend_prepare(ATTENTION_EAGER_CLS),
                metadata={"package": "torch", "note": "debug reference"},
            ),
            # Paged-KV attention.
            Impl(
                kernel_id="phyai_kernel.attention_paged.sm110_expert",
                op="attention_paged",
                priority=Priority.GENERAL,
                when=all_of(
                    NVIDIA_FLASHINFER,
                    lib.has("phyai_kernel"),
                    lib.has("cutlass"),
                    model.family == "pi05",
                    role == "expert",
                    device.arch == "sm110",
                    dtype.input == "bf16",
                    attrs.layout == "paged",
                    attrs.causal == False,
                    attrs.runner.is_set(),
                    attrs.runner == "pi05.expert",
                    attrs.batch_size.is_set(),
                    attrs.batch_size == 1,
                    shape.tokens == 50,
                    shape.heads == 8,
                    shape.kv_heads == 1,
                    shape.head_dim == 256,
                ),
                prepare=_backend_prepare(
                    (
                        "phyai.layers.attention.paged.backends.sm110",
                        "SM110ExpertBackend",
                    )
                ),
                metadata={"package": "phyai-kernel", "note": "opt-in Pi05 expert"},
            ),
            Impl(
                kernel_id="flashinfer.attention_paged",
                op="attention_paged",
                priority=Priority.OPTIMIZED + 2,
                when=FLASHINFER_PAGED,
                prepare=_backend_prepare(PAGED_FLASHINFER_CLS),
                metadata={"package": "flashinfer"},
            ),
            # Gated DeltaNet attention.
            Impl(
                kernel_id="flash_qla.attention_gdn",
                op="attention_gdn",
                priority=Priority.OPTIMIZED + 3,
                # TileLang JIT-compiles on first call; not validated inside
                # captured graphs yet, so capture mode falls back to FLA.
                capture_safe=False,
                when=FLASH_QLA_GDN,
                prepare=_backend_prepare(GDN_FLASH_QLA_CLS),
                metadata={"package": "flash-qla", "note": "prefill; decode via fla"},
            ),
            Impl(
                kernel_id="flashinfer.attention_gdn",
                op="attention_gdn",
                priority=Priority.OPTIMIZED + 2,
                # The FlashInfer GDN backend does not support graph capture.
                capture_safe=False,
                when=FLASHINFER_GDN,
                prepare=_backend_prepare(GDN_FLASHINFER_CLS),
                metadata={"package": "flashinfer"},
            ),
            Impl(
                kernel_id="fla.attention_gdn",
                op="attention_gdn",
                priority=Priority.OPTIMIZED,
                when=all_of(
                    lib.has("fla"),
                    device.vendor == "nvidia",
                    dtype.input.in_(HALF_FLOATS),
                ),
                prepare=_backend_prepare(GDN_FLA_CLS),
                metadata={"package": "flash-linear-attention"},
            ),
        )
    )


__all__ = [
    "ATTENTION",
    "ATTENTION_GDN",
    "ATTENTION_PAGED",
    "register",
]
