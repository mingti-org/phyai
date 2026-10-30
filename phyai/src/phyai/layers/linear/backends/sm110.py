"""Prepare the opt-in BF16 kernels; selection stays in the catalog."""

from __future__ import annotations

from phyai.kernel.policy import PolicyError


def prepare(facts, params):
    if params:
        raise PolicyError(f"SM110 GEMM does not accept parameters: {sorted(params)}")
    from phyai_kernel.cuda.sm110.gemm import Gemm, configurations

    signature = tuple(facts.lookup(f"shape.{name}") for name in ("M", "N", "K"))
    for row in configurations():
        if (
            signature == (row["m"], row["n"], row["k"])
            and facts.lookup("attrs.bias") == row["bias"]
        ):
            return Gemm(row["target"], row["m"])
    raise ValueError("SM110 GEMM capability and prepared shape disagree")
