"""Constraints that used to be hand-rolled per model, now declared on the rows.

Six functions across `models/pi0`, `models/pi05` and `models/minicpm_gr00t`
rewrote a backend *name* at construction time — "flashinfer has no AdaRMS kernel,
use Triton", "the vision tower runs fp32, flashinfer's norm needs bf16",
"head_dim is 72, flashinfer's prefill wants 64/128/256". They existed because the
backend arrived as one global name for the whole model, so a model that could not
use it locally had to rename it.

The names are gone and so are the rewrites. These tests assert the catalog
reaches the same conclusion on its own — they are the standing evidence for the
three models that have no checkpoint here (`minicpm_gr00t`, `qwen3_5`,
`gr00t_n17`), where a bit-exact before/after is not available.
"""

from __future__ import annotations

import pytest

from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


@pytest.fixture
def selector():
    return Selector(build_catalog(), device="nvidia:SM90")


def chosen(selector, op: str, **kwargs) -> str:
    return selector.select(KernelQuery.build(op, **kwargs)).kernel_id


def norm_query(dtype: str) -> dict:
    return dict(
        dtype={"input": dtype, "weight": dtype},
        shape={"tokens": 8, "hidden": 4096},
        attrs={"variant": "rms"},
    )


@pytest.mark.parametrize(
    "dtype, expected",
    [("bf16", "flashinfer.rmsnorm"), ("fp32", "phyai_kernel.rmsnorm")],
)
def test_an_fp32_norm_lands_on_triton_without_a_rewrite(
    selector, dtype, expected
) -> None:
    """Replaces ``_fp32_norm_backend`` / ``_vision_norm_backend``.

    ``flashinfer.rmsnorm`` declares ``dtype.input == bf16``, so an fp32 vision
    tower makes it ineligible and Triton wins on priority. Three models carried
    a copy of this rename.
    """

    assert chosen(selector, "rmsnorm", **norm_query(dtype)) == expected


@pytest.mark.parametrize(
    "head_dim, expected",
    [(128, "flashinfer.attention"), (72, "sdpa.attention")],
)
def test_siglip_head_dim_lands_on_sdpa_without_a_rewrite(
    selector, head_dim, expected
) -> None:
    """Replaces ``_attention_backend_for_head_dim``.

    ``flashinfer.attention`` declares ``shape.head_dim in {64, 128, 256}``;
    SigLIP-So400m is 1152/16 = 72. Deleting the pi0.5 copy of this rewrite is
    what caused a real regression once — the construction call went with it — so
    this test also guards the *reason* the rewrite is safe to remove.
    """

    got = chosen(
        selector,
        "attention",
        dtype={"input": "bf16", "key": "bf16", "value": "bf16"},
        shape={"tokens": 64, "head_dim": head_dim, "num_heads": 16},
        attrs={"layout": "padded", "causal": False},
    )
    assert got == expected


def test_adarmsnorm_lands_on_triton_because_no_flashinfer_row_exists(
    selector,
) -> None:
    """Replaces ``_adarms_backend``.

    Nothing declares anything here — there simply is no FlashInfer AdaRMS row,
    so priority order answers it. The rename existed only because passing
    ``"flashinfer"`` to an op that has no such backend raises.
    """

    got = chosen(
        selector,
        "adarmsnorm",
        dtype={"input": "bf16", "modulation": "bf16"},
        shape={"tokens": 8, "hidden": 4096, "cond_dim": 1024},
    )
    assert got == "phyai_kernel.adarmsnorm"
    assert "flashinfer" not in {
        row.kernel_id.split(".")[0] for row in build_catalog().impls("adarmsnorm")
    }


def test_paged_stack_keeps_flashinfer_as_default(selector) -> None:
    """Specialized opt-in Expert rows preserve the ordinary paged backend."""
    rows = build_catalog().impls("attention_paged")
    assert [row.kernel_id for row in rows] == [
        "flashinfer.attention_paged",
        "flashinfer.attention_paged.cudnn",
        "flashinfer.attention_paged.fa2",
        "flashinfer.attention_paged.fa3",
        "flashinfer.attention_paged.trtllm-gen",
        "phyai_kernel.attention_paged.sm110_expert",
    ]
    assert (
        chosen(
            selector,
            "attention_paged",
            dtype={"input": "bf16", "output": "bf16"},
            shape={"tokens": 50, "heads": 8, "kv_heads": 1, "head_dim": 256},
            attrs={"causal": False, "layout": "paged"},
        )
        == "flashinfer.attention_paged"
    )


def test_rope_falls_back_to_eager_on_its_own(selector) -> None:
    """Replaces ``rope_backend = "flashinfer" if attn_backend == ... else "eager"``.

    ``flashinfer.rope`` declares ``shape.rotary_dim == shape.head_dim``; a
    partial-rotary call fails that leaf and lands on eager.
    """

    full = chosen(
        selector,
        "rope",
        dtype={"input": "bf16"},
        shape={"tokens": 8, "head_dim": 128, "rotary_dim": 128},
        attrs={"rope_type": "default", "interleave": False},
    )
    partial = chosen(
        selector,
        "rope",
        dtype={"input": "bf16"},
        shape={"tokens": 8, "head_dim": 128, "rotary_dim": 64},
        attrs={"rope_type": "default", "interleave": False},
    )
    assert full == "flashinfer.rope"
    assert partial == "eager.rope"
