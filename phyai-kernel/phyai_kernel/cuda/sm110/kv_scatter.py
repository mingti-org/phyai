import triton
import triton.language as language


@triton.jit
def _fused_kv_scatter(
    indices,
    keys,
    values,
    key_cache,
    value_cache,
    ROWS: language.constexpr,
    HEADS: language.constexpr,
    DIMENSION: language.constexpr,
    SLOTS: language.constexpr,
    KEY_ROW: language.constexpr,
    KEY_HEAD: language.constexpr,
    KEY_DIM: language.constexpr,
    VALUE_ROW: language.constexpr,
    VALUE_HEAD: language.constexpr,
    VALUE_DIM: language.constexpr,
    CACHE_ROW: language.constexpr,
    BLOCK: language.constexpr,
):
    offsets = language.program_id(0) * BLOCK + language.arange(0, BLOCK)
    width = HEADS * DIMENSION
    rows = offsets // width
    heads = offsets % width // DIMENSION
    components = offsets % DIMENSION
    active = offsets < ROWS * width
    slots = language.load(indices + rows, active, other=0)
    slots = language.where(slots < 0, slots + SLOTS, slots)
    bounded = (slots >= 0) & (slots < SLOTS)
    language.device_assert((~active) | bounded, "KV slot index out of bounds")
    key_values = language.load(
        keys + rows * KEY_ROW + heads * KEY_HEAD + components * KEY_DIM, active, other=0
    )
    value_values = language.load(
        values + rows * VALUE_ROW + heads * VALUE_HEAD + components * VALUE_DIM,
        active,
        other=0,
    )
    destination = slots * CACHE_ROW + heads * DIMENSION + components
    language.store(key_cache + destination, key_values, active & bounded)
    language.store(value_cache + destination, value_values, active & bounded)


def scatter(indices, keys, values, key_cache, value_cache):
    block = 256 if keys.shape[0] <= 64 else 512
    _fused_kv_scatter[(triton.cdiv(keys.numel(), block),)](
        indices,
        keys,
        values,
        key_cache,
        value_cache,
        keys.shape[0],
        keys.shape[1],
        keys.shape[2],
        key_cache.shape[0],
        *keys.stride(),
        *values.stride(),
        key_cache.stride(0),
        block,
        num_warps=4,
        debug=True,
    )
