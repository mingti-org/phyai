"""Per-layer K/V tensor pool used by every cache subsystem in phyai.

The pool owns one ``(num_slots, page_size, num_kv_heads, head_dim)`` tensor
for K and one for V at every transformer layer. Higher-level allocators
(:class:`~phyai.cache.static_cache.StaticCache`, the radix-cache facades
under :mod:`phyai.cache.radix_cache`) hand out slot indices into this
pool; callers write K/V via :meth:`KVCachePool.write_kv` and read back
via the raw buffers (paged-attention kernels) or
:meth:`KVCachePool.gather_kv` (sdpa fallback path).

The buffer shape mirrors flashinfer's paged-KV-cache convention exactly,
so the buffers can be passed to
``BatchPrefillWithPagedKVCacheWrapper.run`` without an extra view. With
``page_size=1`` (the simplest per-token paging) every slot holds one
token; the leading ``page_size`` dimension is kept so the same code path
extends to larger pages without an interface change.

The pool is intentionally a plain Python object (not an ``nn.Module``):
it is runtime state, not part of the model parameters, and lives
alongside the :class:`~phyai.runtime.model_runner.ModelRunner` that uses
it. Buffers are pre-allocated at construction; pointers are stable for
CUDA-graph capture (every ``write_kv`` becomes a captureable
``index_put_``).
"""

from __future__ import annotations

import torch

from phyai.kernel.call import CallSite


class KVCachePool:
    """Layer-stacked K/V tensor pool with paged-attention-compatible layout.

    Parameters
    ----------
    num_layers:
        Number of transformer layers; one ``(K, V)`` pair per layer.
    num_slots:
        Number of slots (= pages, since ``page_size=1`` by default) per
        layer. The total tensor row count per layer equals ``num_slots``.
    num_kv_heads:
        K/V head count. For grouped-query attention this is smaller than
        the query head count; the pool is layout-agnostic about that.
    head_dim:
        Per-head dimension.
    page_size:
        Tokens per page. Defaults to 1 (per-token paging). The dimension
        is always present in the buffer shape.
    dtype:
        Storage dtype for K and V (typically the model's
        ``params_dtype``).
    device:
        Device the buffers live on.

    Buffer layout (per layer)::

        k_buffer: (num_slots, page_size, num_kv_heads, head_dim)
        v_buffer: same shape

    The pool itself does NOT manage allocation policy — it is a passive
    storage container. Allocation policy (one-shot, radix, etc.) lives in
    the :class:`StaticCache` / :class:`RadixCache` wrappers.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        num_slots: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        page_size: int = 1,
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}.")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}.")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}.")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}.")

        self.num_layers = int(num_layers)
        self.num_slots = int(num_slots)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.dtype = dtype
        self.device = torch.device(device)
        self._write_call = CallSite(
            "kv_scatter",
            role="cache.write",
            prefer=("torch.kv_scatter",),
            dims={"heads": self.num_kv_heads, "head_dim": self.head_dim},
        )

        shape = (
            self.num_slots,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
        )
        self.k_buffers: list[torch.Tensor] = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.v_buffers: list[torch.Tensor] = [
            torch.zeros(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

    # ------------------------------------------------------------------ #
    # Buffer access                                                      #
    # ------------------------------------------------------------------ #

    def k_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the layer's K buffer (no copy).

        Shape ``(num_slots, page_size, num_kv_heads, head_dim)``. The
        returned tensor's storage is the pool's; mutations via this
        reference (e.g. ``buf[i] = ...``) are visible to the pool.
        """
        return self.k_buffers[layer_id]

    def v_buffer(self, layer_id: int) -> torch.Tensor:
        """Return the layer's V buffer (no copy)."""
        return self.v_buffers[layer_id]

    def kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(K, V)`` for a layer.

        Convenience for flashinfer's
        ``BatchPrefillWithPagedKVCacheWrapper.run`` which takes a tuple.
        """
        return self.k_buffers[layer_id], self.v_buffers[layer_id]

    # ------------------------------------------------------------------ #
    # Writes                                                             #
    # ------------------------------------------------------------------ #

    def write_kv(
        self,
        layer_id: int,
        slot_indices: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Scatter K and V at ``slot_indices`` for a single layer.

        Implementation is two ``index_put_`` calls — both captureable in
        a CUDA graph. ``slot_indices`` must be a 1-D ``int64`` tensor on
        the same device as the pool. ``k`` / ``v`` must be 3-D
        ``(N, num_kv_heads, head_dim)`` matching ``slot_indices.shape[0]``;
        the leading page_size dimension is broadcast to (write goes to
        page slot 0 — the only slot in a ``page_size=1`` page).

        Padding tokens are handled by the caller passing a sentinel slot
        index (e.g. 0) whose write is harmless because the slot is never
        in any read-side ``paged_kv_indices``.
        """
        if slot_indices.ndim != 1:
            raise ValueError(
                f"slot_indices must be 1-D, got shape {tuple(slot_indices.shape)}."
            )
        if slot_indices.dtype != torch.int64:
            raise ValueError(f"slot_indices must be int64, got {slot_indices.dtype}.")
        n = slot_indices.shape[0]
        expected = (n, self.num_kv_heads, self.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(
                f"k/v must have shape (N, num_kv_heads, head_dim)="
                f"{expected}; got k={tuple(k.shape)}, v={tuple(v.shape)}."
            )
        if self.page_size != 1:
            raise NotImplementedError(
                f"write_kv currently supports page_size=1 only; pool was "
                f"constructed with page_size={self.page_size}."
            )
        # Index assignment lowers to ``aten::index_put_`` which is a
        # captureable CUDA op. ``[indices, 0]`` selects all (kv_heads,
        # head_dim) at the first (and only) page slot.
        # TODO: The index put can be accelerate using 2d index put triton kernel.
        key_cache, value_cache = self.kv_buffer(layer_id)
        compatible = (
            k.device
            == v.device
            == slot_indices.device
            == key_cache.device
            == value_cache.device
            and not (
                torch.is_grad_enabled()
                and any(t.requires_grad for t in (k, v, key_cache, value_cache))
            )
            and slot_indices.is_contiguous()
            and key_cache.is_contiguous()
            and value_cache.is_contiguous()
            and k.numel() > 0
        )
        selection = self._write_call.select(
            device=k.device,
            dtype={
                "input": k.dtype,
                "value": v.dtype,
                "cache": key_cache.dtype,
                "indices": slot_indices.dtype,
            },
            dims={"tokens": n},
            attrs={"compatible": compatible},
        )
        selection.execute(slot_indices, k, v, key_cache, value_cache)

    # ------------------------------------------------------------------ #
    # Reads                                                              #
    # ------------------------------------------------------------------ #

    def gather_kv(
        self,
        layer_id: int,
        slot_indices: torch.Tensor,
        *,
        out_k: torch.Tensor | None = None,
        out_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K and V at ``slot_indices`` into contiguous buffers.

        Used by the sdpa / eager fallback attention path. The flashinfer
        paged path reads :meth:`kv_buffer` directly and does not invoke
        this helper.

        ``out_k`` / ``out_v`` may be pre-allocated scratch tensors for
        reuse across layers; if given they must have shape
        ``(N, num_kv_heads, head_dim)`` where ``N == slot_indices.shape[0]``.
        Otherwise fresh tensors are allocated.
        """
        if slot_indices.ndim != 1:
            raise ValueError(
                f"slot_indices must be 1-D, got shape {tuple(slot_indices.shape)}."
            )
        n = slot_indices.shape[0]
        # ``index_select`` along dim 0 picks the requested slots; the
        # ``squeeze(1)`` collapses the page_size=1 dim so callers get a
        # 3-D ``(N, H_kv, D)`` view directly.
        k_view = self.k_buffers[layer_id].index_select(0, slot_indices).squeeze(1)
        v_view = self.v_buffers[layer_id].index_select(0, slot_indices).squeeze(1)
        if out_k is not None:
            expected = (n, self.num_kv_heads, self.head_dim)
            if tuple(out_k.shape) != expected or tuple(out_v.shape) != expected:
                raise ValueError(
                    f"out_k/out_v must match (N, H_kv, D)={expected}; "
                    f"got out_k={tuple(out_k.shape)}, out_v={tuple(out_v.shape)}."
                )
            out_k.copy_(k_view)
            out_v.copy_(v_view)
            return out_k, out_v
        return k_view.contiguous(), v_view.contiguous()

    # ------------------------------------------------------------------ #
    # Misc                                                               #
    # ------------------------------------------------------------------ #

    def zero_(self) -> None:
        """Reset every K/V buffer to zero. Pool tensor pointers unchanged.

        Useful between unrelated inferences when a deterministic clean
        slate matters (e.g. tests). Inference correctness does not
        require zeroing because attention only reads slots whose indices
        appear in the paged_kv_indices buffer — slots not on that list
        are never touched.
        """
        for buf in self.k_buffers:
            buf.zero_()
        for buf in self.v_buffers:
            buf.zero_()

    def total_bytes(self) -> int:
        """Return the total bytes allocated across every K and V buffer."""
        per_buffer = (
            self.num_slots
            * self.page_size
            * self.num_kv_heads
            * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )
        return per_buffer * 2 * self.num_layers

    def __repr__(self) -> str:
        return (
            f"KVCachePool(num_layers={self.num_layers}, "
            f"num_slots={self.num_slots}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"page_size={self.page_size}, "
            f"dtype={self.dtype}, device={self.device})"
        )


__all__ = ["KVCachePool"]
