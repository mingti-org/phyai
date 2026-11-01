"""Small LingBot regressions: real loaders and attention, no model checkpoint."""

from dataclasses import replace

import torch
import pytest
from phyai.cache import KVCachePool
from phyai.weights import load_pretrained
from phyai.utils.cuda import init_cublas
from safetensors.torch import save_file
from phyai.engine_config import (
    DeviceConfig,
    EngineConfig,
    get_engine_config,
    set_engine_config,
)
from phyai.layers.attention import (
    AttnMode,
    AttnLayout,
    PagedAttnCtx,
    PagedAttention,
    PagedAttnMetadata,
)
from phyai.models.lingbot_v2.modeling_lingbotv2 import (
    LingBotV2TokenMoE,
    Qwen3VLVisionAttention,
)
from phyai.models.lingbot_v2.model_runner_lingbotv2 import _LingBotV2OfficialARBackend
from phyai.models.lingbot_v2.configuration_lingbotv2 import (
    LingBotV2MoEConfig,
    Qwen3VLVisionConfig,
    LingBotV2ExpertConfig,
)


@pytest.fixture(autouse=True)
def model_config(fake_mesh):
    previous = get_engine_config()
    set_engine_config(EngineConfig(device=DeviceConfig(target="cpu")))
    fake_mesh(tp_size=1)
    try:
        yield
    finally:
        set_engine_config(previous)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        ("weight", (48, 1)),
        ("weight", (49, 16)),
        ("weight", (48, 16, 1)),
        ("bias", (48, 1)),
        ("bias", (1,)),
        ("bias", (49,)),
    ],
)
def test_vision_qkv_checkpoint_shape_and_rank_order(
    tmp_path, fake_mesh, world, invalid
):
    """Exercise the actual constructor, safetensors dispatch and packed loader."""
    config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        num_heads=4,
        intermediate_size=32,
        out_hidden_size=16,
        num_position_embeddings=4,
        deepstack_visual_indexes=(),
    )
    weight = (torch.arange(48 * 16).reshape(48, 16) % 127).to(torch.bfloat16)
    bias = torch.arange(48).to(torch.bfloat16)
    tensors = {"visual.qkv.weight": weight, "visual.qkv.bias": bias}
    if invalid is not None:
        name, shape = invalid
        tensors[f"visual.qkv.{name}"] = torch.ones(shape, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    for rank in range(world):
        fake_mesh(tp_size=world, rank=rank)
        layer = Qwen3VLVisionAttention(
            config, params_dtype=torch.bfloat16, prefix="visual"
        ).qkv
        if invalid is not None:
            with pytest.raises(ValueError, match="shape"):
                load_pretrained(layer, tmp_path, strict=True)
        else:
            load_pretrained(layer, tmp_path, strict=True)
            local = 16 // world
            rows = torch.cat(
                [
                    torch.arange(leg * 16 + rank * local, leg * 16 + (rank + 1) * local)
                    for leg in range(3)
                ]
            )
            torch.testing.assert_close(layer.weight, weight[rows], rtol=0, atol=0)
            torch.testing.assert_close(layer.bias, bias[rows], rtol=0, atol=0)


@pytest.mark.parametrize(
    "device,allow_tf32,causal,amp,dtype,prefix",
    [
        ("cpu", False, True, True, torch.float32, 3),
        ("cuda", False, False, True, torch.bfloat16, 0),
        ("cuda", True, True, True, torch.float32, 3),
        ("cuda", True, False, False, torch.bfloat16, 3),
    ],
)
def test_official_attention_fp32_under_amp(
    device, allow_tf32, causal, amp, dtype, prefix
):
    """FP32 cache, noncontiguous Q and GQA; independent sequence-order oracle."""
    previous = torch.get_float32_matmul_precision()
    try:
        init_cublas(allow_tf32=allow_tf32)
        _check_attention(device, causal, amp, dtype, prefix)
    finally:
        torch.set_float32_matmul_precision(previous)


def _check_attention(device, causal, amp, dtype, prefix):
    generator = torch.Generator().manual_seed(103 + prefix)
    length, total = 2, prefix + 2
    layer = PagedAttention(
        num_heads=2, num_kv_heads=1, head_dim=8, layer_id=0, causal=causal
    )
    backend = _LingBotV2OfficialARBackend()
    pool = KVCachePool(
        num_layers=1,
        num_slots=total,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
        device=device,
    )
    slots = torch.arange(total, device=device).flip(0)
    q = torch.randn(length, 2, 16, generator=generator).to(device=device, dtype=dtype)[
        ..., ::2
    ]
    k = torch.randn(total, 1, 8, generator=generator).to(device=device, dtype=dtype)
    v = torch.randn(total, 1, 8, generator=generator).to(device=device, dtype=dtype)
    if prefix:
        pool.write_kv(0, slots[:prefix], k[:prefix].float(), v[:prefix].float())
    meta = PagedAttnMetadata(
        mode=AttnMode.PREFILL,
        layout=AttnLayout.RAGGED_3D,
        batch_size=1,
        num_query_tokens=length,
        cu_seqlens_q=torch.tensor([0, length], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, total], dtype=torch.int32),
        paged_kv_indices=slots,
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
        write_indices=slots[prefix:],
    )
    ctx = PagedAttnCtx(
        backend=backend,
        plan=backend.init_forward_metadata(meta),
        mode=meta.mode,
        layout=meta.layout,
        kv_pool=pool,
        write_indices=meta.write_indices,
    )
    # The oracle uses logical K/V directly, without the candidate's paged gather.
    scores = (
        q.float().transpose(0, 1)
        @ k.float().repeat_interleave(2, 1).transpose(0, 1).transpose(-2, -1)
    ) * (8**-0.5)
    if causal:
        mask = (
            torch.arange(total, device=device)[None, :]
            <= (torch.arange(length, device=device) + prefix)[:, None]
        )
        scores = scores.masked_fill(~mask, -2.3819763e38)
    expected = (
        scores.softmax(-1) @ v.float().repeat_interleave(2, 1).transpose(0, 1)
    ).transpose(0, 1)
    before = torch.is_autocast_enabled(device)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=amp):
        actual = backend.forward(layer, q, k[prefix:], v[prefix:], ctx)
        assert torch.is_autocast_enabled(device) == amp
        probe = torch.ones(2, 2, device=device) @ torch.ones(2, 2, device=device)
        assert probe.dtype == (torch.bfloat16 if amp else torch.float32)
    assert torch.is_autocast_enabled(device) == before
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    cached_k, cached_v = pool.gather_kv(0, slots)
    assert cached_k.dtype == cached_v.dtype == torch.float32
    torch.testing.assert_close(cached_k, k.float(), rtol=0, atol=0)
    torch.testing.assert_close(cached_v, v.float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_adarms", False),
        ("final_norm_adarms", True),
        ("hidden_act", "gelu"),
        ("mlp_bias", True),
    ],
)
def test_unsupported_expert_options_name_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        LingBotV2ExpertConfig(**{field: value})


def test_supported_defaults_and_router_correction_contract():
    expert = LingBotV2ExpertConfig()
    assert expert.use_adarms and not expert.final_norm_adarms
    assert expert.hidden_act == "silu" and not expert.mlp_bias
    config = LingBotV2MoEConfig(
        layer_indices=(0,),
        num_experts=4,
        top_k=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
    )
    model = LingBotV2TokenMoE(
        16, config, params_dtype=torch.bfloat16, device="cpu", prefix="moe"
    )
    assert model.top_k == 2
    with pytest.raises(NotImplementedError, match="use_router_correction_bias=False"):
        LingBotV2TokenMoE(
            16,
            replace(config, use_router_correction_bias=False),
            params_dtype=torch.bfloat16,
            device="cpu",
            prefix="moe",
        )
