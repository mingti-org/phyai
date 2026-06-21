"""Tests for paged-KV attention layers — ``ARAttention`` and ``DiffusionAttention``.

Both stacks are **flashinfer-only** (GPU): ``"flashinfer"`` is the only
backend registered in either subpackage. Layer construction is a pure
registry lookup (it does not instantiate the backend, so no flashinfer
import is triggered), which keeps these construction / validation tests
runnable without CUDA. Forward-numerical coverage lives in the
CUDA-gated ``test_flashinfer_paged.py``.
"""

from __future__ import annotations

import pytest

from phyai.layers.attention import (
    ARAttention,
    ARAttentionBackend,
    DiffusionAttention,
    DiffusionAttentionBackend,
    get_ar_backend_factory,
    get_diffusion_backend_factory,
)


# --------------------------------------------------------------------- #
# Parametrization helpers                                               #
# --------------------------------------------------------------------- #


_PAGED_FLAVORS = ("ar", "diffusion")


def _layer_cls(flavor: str):
    return ARAttention if flavor == "ar" else DiffusionAttention


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_flashinfer_backend_constructs(flavor: str):
    """Construction resolves the flashinfer factory by name without
    instantiating it (no flashinfer import on CPU)."""
    cls = _layer_cls(flavor)
    attn = cls(
        num_heads=4,
        head_dim=8,
        layer_id=0,
        num_kv_heads=4,
        backend="flashinfer",
    )
    assert attn.backend == "flashinfer"
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 4
    assert attn.head_dim == 8
    assert attn.layer_id == 0


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_sdpa_backend(flavor: str):
    """SDPA cannot serve the paged space — only registered in the
    no-cache stack, so layer construction must reject it."""
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="not registered"):
        cls(num_heads=4, head_dim=8, layer_id=0, backend="sdpa")


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_eager_backend(flavor: str):
    """The paged stacks are flashinfer-only — ``"eager"`` is no longer
    registered for AR / Diffusion (it remains only in the no-cache stack)."""
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="not registered"):
        cls(num_heads=4, head_dim=8, layer_id=0, backend="eager")


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_invalid_backend(flavor: str):
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="not registered"):
        cls(
            num_heads=4,
            head_dim=8,
            layer_id=0,
            backend="not-a-backend",
        )


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_bad_gqa(flavor: str):
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="must be a positive multiple"):
        cls(
            num_heads=4,
            head_dim=8,
            layer_id=0,
            num_kv_heads=3,
            backend="flashinfer",
        )


@pytest.mark.parametrize("flavor", _PAGED_FLAVORS)
def test_paged_attention_rejects_negative_layer_id(flavor: str):
    cls = _layer_cls(flavor)
    with pytest.raises(ValueError, match="layer_id must be non-negative"):
        cls(num_heads=4, head_dim=8, layer_id=-1, backend="flashinfer")


# --------------------------------------------------------------------- #
# Sanity: ar and diffusion backend classes are independent              #
# --------------------------------------------------------------------- #


def test_ar_and_diffusion_backends_are_independent_classes():
    """The two stacks have separate registries and separate backend
    classes — neither should accidentally alias the other.

    Checks factory identity and ABC subclassing *without instantiating*
    (instantiation imports flashinfer, which is unavailable on CPU)."""
    ar_fi = get_ar_backend_factory("flashinfer")
    diff_fi = get_diffusion_backend_factory("flashinfer")
    assert ar_fi is not diff_fi
    # ``@register_backend`` stores the backend class itself as the factory.
    assert isinstance(ar_fi, type) and issubclass(ar_fi, ARAttentionBackend)
    assert isinstance(diff_fi, type) and issubclass(diff_fi, DiffusionAttentionBackend)
