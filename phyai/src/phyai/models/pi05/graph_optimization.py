"""Experimental, instance-owned Pi0.5 Graph dataflow specialization.

All five mechanisms are opt-in. The original model and full-hidden prefix
route remain available; the plan never patches a shared module or cache class.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import torch

from phyai.env import envs
from phyai.kernel.call import select
from phyai.kernel.ops.rope import strided_input_supported
from phyai.kernel.selector import NoKernelError
from phyai.utils import get_logger

logger = get_logger(__name__)


def supports_configuration(scheduler) -> bool:
    """Bound the migration to the historical geometry, without probing hardware."""
    cfg = scheduler.cfg
    return (
        scheduler.device.type == "cuda"
        and scheduler.max_batch_size == 1
        and scheduler.num_images == 3
        and scheduler.params_dtype == torch.bfloat16
        and scheduler.model.vision_params_dtype == torch.bfloat16
        and scheduler.vision_runner.use_cuda_graph
        and scheduler.llm_runner.use_cuda_graph
        and scheduler.expert_runner.use_cuda_graph
        and not scheduler.expert_runner.capture_rollout
        and (
            cfg.text.num_hidden_layers,
            cfg.expert.num_hidden_layers,
            cfg.text.hidden_size,
            cfg.expert.hidden_size,
            cfg.text.head_dim,
            cfg.text.num_attention_heads,
            cfg.text.num_key_value_heads,
            cfg.expert.head_dim,
            cfg.expert.num_attention_heads,
            cfg.expert.num_key_value_heads,
            cfg.tokenizer_max_length,
            cfg.chunk_size,
            cfg.max_action_dim,
            cfg.num_inference_steps,
            cfg.vision.image_size,
            cfg.vision.num_patches,
        )
        == (18, 18, 2048, 1024, 256, 8, 1, 256, 8, 1, 200, 50, 32, 10, 224, 256)
        and cfg.expert.use_adarms
        and cfg.expert.adarms_cond_dim == 1024
        and (scheduler.sentinel_slot, scheduler.prefix_base, scheduler.suffix_base)
        == (0, 1, 969)
        and scheduler.kv_pool.num_slots == 1019
        and scheduler.kv_pool.page_size == 1
    )


class StridedRope:
    def __init__(self, original, function):
        self.original, self.function = original, function

    def __call__(self, positions, q, k):
        original = self.original
        original._check_shapes(positions, q, k)
        table = original.cos_sin_cache
        if table.device != q.device:
            raise RuntimeError("RoPE cache and input must share a device")
        if (
            strided_input_supported(q, 8)
            and strided_input_supported(k, 1)
            and positions.dtype == torch.int32
            and positions.is_contiguous()
        ):
            return self.function(positions, q, k, table)
        # contiguous() preserves an already-contiguous misaligned offset view.
        # Clone such inputs before calling the original vectorized backend.
        q = q.clone(memory_format=torch.contiguous_format) if q.data_ptr() % 16 else q
        k = k.clone(memory_format=torch.contiguous_format) if k.data_ptr() % 16 else k
        return original(positions, q, k)


class VectorKVView:
    """Specialize scheduler-owned writes; retain the original pool's identity."""

    def __init__(self, pool, function):
        self.pool, self.function = pool, function

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def write_kv(self, layer_id, slot_indices, k, v):
        self.function(
            k,
            v,
            self.pool.k_buffers[layer_id],
            self.pool.v_buffers[layer_id],
            slot_indices,
        )


class CanonicalVision:
    """Materialize token rows after the original FP32 patch embedding boundary."""

    def __init__(self, original):
        self.original = original

    def __call__(self, pixel_values):
        tower = self.original
        model = tower.vision_tower.vision_model
        h = model.embeddings(pixel_values).contiguous()
        h = model.encoder(h)
        h = model.post_layernorm(h)
        return tower.multi_modal_projector(h).to(tower.io_dtype)


class PI05GraphOptimization:
    """Own prepared calls and 300 KiB of expert scratch for one scheduler."""

    def __init__(self, scheduler):
        device = scheduler.device
        common = {"device": device, "mode": "capture"}
        fusion = select(
            "gated_residual_adarmsnorm",
            role="pi05.expert.residual_norm",
            dtype={"input": "bf16", "modulation": "bf16"},
            shape={"tokens": 50, "hidden": 1024},
            **common,
        )
        scatter = select(
            "kv_scatter",
            role="pi05.kv_write",
            dtype={"input": "bf16"},
            shape={"tokens": 968, "heads_k": 1, "head_dim": 256, "slots": 1019},
            **common,
        )
        rope = select(
            "rope_strided",
            role="pi05.rope",
            dtype={"input": "bf16"},
            shape={
                "tokens": 968,
                "head_dim": 256,
                "rotary_dim": 256,
                "heads_q": 8,
                "heads_k": 1,
            },
            attrs={
                "rope_type": scheduler.model.rope.rope_type,
                "interleave": False,
                "rank": 3,
            },
            **common,
        )
        expected = (
            "phyai_kernel.gated_residual_adarmsnorm.sm110",
            "phyai_kernel.kv_scatter.sm110",
            "flashinfer.rope_strided.sm110",
        )
        if (fusion.kernel_id, scatter.kernel_id, rope.kernel_id) != expected:
            raise ValueError(
                "catalog or kernel policy did not select the specialized implementations"
            )
        self.kernel_ids = expected
        self.fusion = fusion.implementation
        self.pool = VectorKVView(scheduler.kv_pool, scatter.implementation)
        self.rope = StridedRope(scheduler.model.rope, rope.implementation)
        self.vision = CanonicalVision(scheduler.vision_runner.vision_tower)
        self.h = [
            torch.empty((50, 1024), dtype=torch.bfloat16, device=device)
            for _ in range(2)
        ]
        self.y = torch.empty_like(self.h[0])
        # Initialize FlashInfer outside capture on the owning device/stream.
        fused = torch.zeros((2, 2560), dtype=torch.bfloat16, device=device)
        q, k, _ = fused.split((2048, 256, 256), dim=-1)
        positions = torch.arange(2, dtype=torch.int32, device=device)
        self.rope(positions, q.view(2, 8, 256), k.view(2, 1, 256))
        torch.cuda.current_stream(device).synchronize()

    def prefix(self, model, h, positions, ctx) -> None:
        """Populate every layer's effective KV; action inference consumes no hidden."""
        for layer in model.layers[:-1]:
            h = layer(h, positions, self.rope, ctx)
        last = model.layers[-1]
        n = last.input_layernorm(h)
        fused, _ = last.qkv_proj(n)
        q, k, v = last._split_qkv(fused, h.shape[:-1])
        _, k = self.rope(positions, q, k)
        ctx.kv_pool.write_kv(last.attn.layer_id, ctx.write_indices, k, v)

    def expert(self, model, h, positions, ctx, modulation):
        cursor = 0

        def pair(norm, a, b, gate, mod):
            nonlocal cursor
            output = self.h[cursor % 2]
            cursor += 1
            return self.fusion(a, b, gate, mod, output, self.y, norm.variance_epsilon)

        pending = None
        for layer, mod in zip(model.layers, modulation.layers, strict=True):
            if pending is None:
                n, gate = layer.input_layernorm(h, modulation=mod.input_ln)
            else:
                h, n = pair(layer.input_layernorm, *pending, mod.input_ln)
                gate = mod.input_ln[:, 2048:3072]
            fused, _ = layer.qkv_proj(n)
            q, k, v = layer._split_qkv(fused, h.shape[:-1])
            q, k = self.rope(positions, q, k)
            attn = layer.attn(q, k, v, ctx)
            out, _ = layer.o_proj(attn.reshape(*attn.shape[:-2], -1))
            h, n = pair(
                layer.post_attention_layernorm, h, out, gate, mod.post_attention_ln
            )
            pending = (h, layer.mlp(n), mod.post_attention_ln[:, 2048:3072])
        _, result = pair(model.norm, *pending, modulation.final)
        return result


def attach(scheduler) -> PI05GraphOptimization | None:
    """Prepare atomically before capture; keep unsupported instances native."""
    if not envs.PHYAI_PI05_THOR_OPTIMIZATIONS.get():
        return None
    if not supports_configuration(scheduler):
        logger.warning_once(
            "Pi0.5 Thor optimizations requested: unsupported configuration; using original path."
        )
        return None
    try:
        if version("flashinfer-python") != "0.6.17":
            raise ValueError("this migration targets the pinned FlashInfer 0.6.17")
        from phyai.layers.linear.layers import LinearBase
        from phyai.layers.quant import Bf16Spec

        if any(
            m._forward_hooks
            or m._forward_pre_hooks
            or (isinstance(m, LinearBase) and type(m.spec) is not Bf16Spec)
            for m in scheduler.model.modules()
        ):
            raise ValueError("hooks or non-BF16 linears require the original path")
        if not all(
            type(r.attn_backend).__module__.endswith(".flashinfer")
            for r in (scheduler.llm_runner, scheduler.expert_runner)
        ):
            raise ValueError("the specialization requires FlashInfer paged attention")
        rope = scheduler.model.rope
        if (
            rope.rotary_dim != 256
            or rope.interleave
            or rope.cos_sin_cache.shape[0] < 1018
        ):
            raise ValueError("unsupported RoPE table or convention")
        # Honor explicit kernel policy on the original numerical boundaries.
        with torch.cuda.device(scheduler.device):
            original_rope = rope._call_site.select(
                device=scheduler.device,
                dtype={"input": "bf16"},
                dims={"tokens": 50, "heads_q": 8, "heads_k": 1},
                attrs={"rank": 3},
            )
            norms = [scheduler.model.expert_stack.norm]
            for layer in scheduler.model.expert_stack.layers:
                norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
            if original_rope.kernel_id != "flashinfer.rope" or any(
                norm._call.select(
                    device=scheduler.device,
                    dtype={"input": "bf16", "modulation": "bf16", "output": "bf16"},
                    dims={"tokens": 50, "hidden": 1024},
                ).kernel_id
                != "phyai_kernel.adarmsnorm"
                for norm in norms
            ):
                raise ValueError("original kernel policy selects different numerics")
            plan = PI05GraphOptimization(scheduler)
    except (ImportError, PackageNotFoundError, ValueError, NoKernelError) as exc:
        logger.warning_once(
            "Pi0.5 Thor optimizations unavailable (%s); using original path.", exc
        )
        return None
    scheduler.llm_runner.optimization = plan
    scheduler.expert_runner.optimization = plan
    scheduler.vision_runner.optimization = plan
    logger.info(
        "Enabled experimental Pi0.5 Thor Graph optimizations: %s.", plan.kernel_ids
    )
    return plan
