"""Scatter valid K/V rows while leaving the unread padding slot untouched."""

from __future__ import annotations

from phyai.kernel.facts import device, dtype, lib, shape
from phyai.kernel.opspec import Impl, OpSpec, Priority
from phyai.kernel.predicate import all_of
from phyai.kernel.registry import Catalog

KV_SCATTER = OpSpec(
    name="kv_scatter",
    dims=("tokens", "heads_k", "head_dim", "slots"),
    dtypes=("input",),
    signature="(k, v, k_cache, v_cache, slots) -> None",
    doc="Unique positive slot writes; zero is an unread sentinel. Caller validates slot values before staging.",
)


def _cuda(facts, params):
    from phyai_kernel.cuda.jit.kv_vector_scatter import prepare, run

    call = prepare()

    def scatter(k, v, k_cache, v_cache, slots):
        run(k, v, k_cache, v_cache, slots, call)

    return scatter


def _torch(facts, params):
    def scatter(k, v, k_cache, v_cache, slots):
        valid = slots > 0
        k_cache[slots[valid], 0] = k[valid]
        v_cache[slots[valid], 0] = v[valid]

    return scatter


def register(catalog: Catalog) -> None:
    catalog.register_op(KV_SCATTER)
    catalog.register_many(
        (
            Impl(
                kernel_id="phyai_kernel.kv_scatter.sm110",
                op="kv_scatter",
                when=all_of(
                    device.vendor == "nvidia",
                    device.arch == "sm110",
                    lib.has("phyai_kernel"),
                    dtype.input == "bf16",
                    shape.heads_k == 1,
                    shape.head_dim == 256,
                    shape.slots == 1019,
                    shape.tokens >= 1,
                    shape.tokens <= 4096,
                ),
                prepare=_cuda,
                priority=Priority.OPTIMIZED,
            ),
            Impl(
                kernel_id="torch.kv_scatter",
                op="kv_scatter",
                when=dtype.input.is_set(),
                prepare=_torch,
                priority=Priority.REFERENCE,
                reference=True,
                capture_safe=False,
            ),
        )
    )
