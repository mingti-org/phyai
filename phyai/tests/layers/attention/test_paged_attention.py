"""Construction contract of the unified paged-KV attention layer.

Construction validates backend names without instantiating them. Numerical
coverage lives in the backend-specific test modules.
"""

from __future__ import annotations

import pytest

from phyai.layers.attention import PagedAttention


def test_construction_records_traits_and_causality_per_layer():
    """Causality is a layer trait, not a subsystem: the old design shipped two
    identical stacks whose only executable difference was the ``causal`` default."""
    attn = PagedAttention(
        num_heads=4,
        head_dim=8,
        layer_id=0,
        causal=True,
        num_kv_heads=4,
        backend="flashinfer",
    )
    assert (attn.backend, attn.num_heads, attn.num_kv_heads) == ("flashinfer", 4, 4)
    assert (attn.head_dim, attn.layer_id, attn.causal) == (8, 0, True)
    prefix = PagedAttention(
        num_heads=4, head_dim=8, layer_id=0, causal=False, kernel_role="prefix"
    )
    assert prefix.causal is False and prefix.kernel_role == "prefix"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "eager"}, "unknown backend"),
        ({"backend": "not-a-backend"}, "unknown backend"),
        ({"num_kv_heads": 3}, "must be a positive multiple"),
        ({"layer_id": -1}, "layer_id must be non-negative"),
    ],
)
def test_construction_rejects_unserviceable_configs(kwargs, message):
    base = dict(num_heads=4, head_dim=8, layer_id=0, causal=True, backend="flashinfer")
    with pytest.raises(ValueError, match=message):
        PagedAttention(**{**base, **kwargs})
