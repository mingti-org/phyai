"""Optional compound operations with ordinary layer implementations as defaults."""

from phyai.kernel.facts import attrs, device, dtype, lib, model, shape
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.predicate import all_of, any_of
from phyai.kernel.registry import Catalog


KV_SCATTER = OpSpec(
    name="kv_scatter",
    dims=("tokens", "heads", "head_dim"),
    dtypes=("input", "value", "cache", "indices"),
    attributes=("compatible",),
    signature="(indices, k, v, k_cache, v_cache) -> None",
)
DENSE_MLP = OpSpec(
    name="dense_mlp",
    dims=("tokens", "hidden", "intermediate"),
    dtypes=("input", "weight"),
    optional_dtypes=("bias",),
    attributes=("activation", "gated", "bias", "compatible"),
    signature="(mlp, x) -> Tensor",
)
TENSOR_LAYOUT = OpSpec(
    name="tensor_layout",
    dims=("tokens", "hidden"),
    dtypes=("input",),
    attributes=(),
    signature="(x) -> Tensor",
    doc="Preserve values while optionally materializing row-major storage.",
)


def _scatter_reference(facts, params):
    def execute(indices, k, v, key_cache, value_cache):
        key_cache[indices, 0] = k
        value_cache[indices, 0] = v

    return execute


def _mlp_reference(facts, params):
    return lambda mlp, x: mlp.forward_unfused(x)


def _identity(facts, params):
    return lambda x: x


def _contiguous(facts, params):
    return lambda x: x.contiguous()


def _sm110(name):
    def prepare(facts, params):
        from phyai.kernel.policy import PolicyError

        if params:
            raise PolicyError(
                f"SM110 fusion does not accept parameters: {sorted(params)}"
            )
        from phyai_kernel.cuda.sm110 import fusion

        return getattr(fusion, name)(facts)

    return prepare


def register(catalog: Catalog) -> None:
    common = all_of(
        model.family == "pi05",
        device.vendor == "nvidia",
        device.arch == "sm110",
        lib.has("phyai_kernel"),
        lib.has("flashinfer"),
        lib.has("cutlass"),
        dtype.input == "bf16",
        attrs.compatible,
    )
    for spec, reference in (
        (KV_SCATTER, _scatter_reference),
        (DENSE_MLP, _mlp_reference),
        (TENSOR_LAYOUT, _identity),
    ):
        catalog.register_op(spec)
        catalog.register(
            Impl(
                kernel_id=f"torch.{spec.name}",
                op=spec.name,
                priority=Priority.REFERENCE,
                reference=True,
                when=dtype.input.is_set(),
                prepare=reference,
                metadata={"package": "torch", "note": "default layer composition"},
            )
        )
    catalog.register_many(
        (
            Impl(
                kernel_id="phyai_kernel.tensor_layout.sm110",
                op="tensor_layout",
                priority=Priority.GENERAL,
                when=all_of(
                    model.family == "pi05",
                    device.arch == "sm110",
                    dtype.input == "bf16",
                    shape.tokens == 512,
                    shape.hidden == 1152,
                ),
                prepare=_contiguous,
                metadata={"package": "phyai-kernel"},
            ),
            Impl(
                kernel_id="phyai_kernel.kv_scatter.sm110",
                op="kv_scatter",
                priority=Priority.GENERAL,
                when=all_of(
                    common,
                    dtype.value == "bf16",
                    dtype.cache == "bf16",
                    dtype.indices == "int64",
                ),
                prepare=_sm110("prepare_scatter"),
                metadata={"package": "phyai-kernel"},
            ),
            Impl(
                kernel_id="phyai_kernel.dense_mlp.sm110",
                op="dense_mlp",
                priority=Priority.GENERAL,
                when=all_of(
                    common,
                    dtype.weight == "bf16",
                    attrs.activation == "gelu",
                    any_of(
                        all_of(
                            shape.tokens == 512,
                            shape.hidden == 1152,
                            shape.intermediate == 4304,
                            attrs.gated == False,
                            attrs.bias,
                            dtype.bias == "bf16",
                        ),
                        all_of(
                            shape.tokens.in_({528, 560, 624, 712}),
                            shape.hidden == 2048,
                            shape.intermediate == 16384,
                            attrs.gated,
                            attrs.bias == False,
                        ),
                    ),
                ),
                prepare=_sm110("DenseMLPFusion"),
                metadata={"package": "phyai-kernel"},
            ),
        )
    )
