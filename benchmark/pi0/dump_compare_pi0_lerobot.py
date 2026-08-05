"""Dump and compare PI0 intermediate tensors from LeRobot and PhyAI.

This is the unified PhyAI-side entry point for layer alignment. It consumes a
payload produced by ``benchmark/pi0/compare_pi0_lerobot_live.py --save-output`` so
both runtimes see identical inputs, prompt tokens, and sampling noise.

Example:

    CUDA_VISIBLE_DEVICES=7 uv run python benchmark/pi0/dump_compare_pi0_lerobot.py \
        --checkpoint /data/share/pi0_base \
        --pt pt/pi0_bf16_payload.pt \
        --lerobot-root /path/to/lerobot \
        --device cuda \
        --dtype bfloat16 \
        --vision-dtype float32 \
        --phyai-attn-backend flashinfer \
        --num-steps 10 \
        --detail-layers 0 1 \
        --out pt/pi0_bf16_combined_dump.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from compare_pi0_lerobot_live import (
        add_lerobot_to_path,
        choose_attn_backend,
        cuda_device_from_name,
        dtype_from_name,
        import_lerobot_symbols,
        is_empty_camera_key,
        load_lerobot_policy,
        OBS_IMAGES_PREFIX,
        trim_lerobot_image_features,
        resolve_lerobot_image_keys,
    )
except ModuleNotFoundError:
    from benchmark.pi0.compare_pi0_lerobot_live import (
        add_lerobot_to_path,
        choose_attn_backend,
        cuda_device_from_name,
        dtype_from_name,
        import_lerobot_symbols,
        is_empty_camera_key,
        load_lerobot_policy,
        OBS_IMAGES_PREFIX,
        trim_lerobot_image_features,
        resolve_lerobot_image_keys,
    )
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import BackendConfig, DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi0.configuration_pi0 import PI0Config
from phyai.models.pi0.main_pi0 import PI0Args
from phyai.models.pi0.scheduler_ws1_pi0 import PI0Request
from phyai.utils import load_config as load_phyai_config


OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"

TOKEN_ALIASES = (
    OBS_LANGUAGE_TOKENS,
    "observation.language_tokens",
    "language_tokens",
    "input_ids",
)

MASK_ALIASES = (
    OBS_LANGUAGE_ATTENTION_MASK,
    "observation.language_attention_mask",
    "language_attention_mask",
    "attention_mask",
)

COARSE_CUTS = [
    ("img_emb_pre_io_cast", "image tokens before model IO dtype cast"),
    ("img_emb", "image tokens (SigLIP + projector)"),
    ("lang_emb", "language token embeddings"),
    ("prefix_embs", "packed prefix (LLM input)"),
    ("prefix_hidden", "LLM prefix output (post norm)"),
    ("x_t_step0", "noise fed to expert at step 0"),
    ("state_emb", "state token embedding"),
    ("action_time_emb", "action+time token embeddings"),
]

DETAIL_CUT_SUFFIXES = [
    ("input_norm", "input RMSNorm output"),
    ("qkv", "QKV projection output"),
    ("q_rope", "query after RoPE"),
    ("k_rope", "key after RoPE"),
    ("v", "value projection"),
    ("attn", "attention output before o_proj"),
    ("o_proj", "attention output projection"),
    ("attn_residual", "first residual output"),
    ("post_norm", "post-attention RMSNorm output"),
    ("mlp", "MLP output"),
]


def detail_cuts(layer_idx: int) -> list[tuple[str, str]]:
    cuts = [
        (
            f"prefix_layer{layer_idx}_{suffix}",
            f"prefix layer {layer_idx} {description}",
        )
        for suffix, description in DETAIL_CUT_SUFFIXES
    ]
    cuts.append(
        (
            f"prefix_layer{layer_idx}",
            f"prefix layer {layer_idx} second residual output",
        )
    )
    return cuts


def validate_detail_layers(detail_layers: tuple[int, ...], num_layers: int) -> None:
    invalid = [layer_idx for layer_idx in detail_layers if layer_idx >= num_layers]
    if invalid:
        raise ValueError(
            f"Detail layer indices {invalid} are out of range for {num_layers} layers."
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Dump LeRobot and PhyAI PI0 intermediates and compare them."
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--pt", type=Path, required=True, help="Payload .pt from live compare."
    )
    ap.add_argument(
        "--lerobot-root",
        type=Path,
        default=None,
        help="Optional LeRobot checkout root. Its src/ directory is prepended to sys.path.",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="Use cuda/cuda:0 after binding the physical GPU with CUDA_VISIBLE_DEVICES.",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=("bf16", "bfloat16"))
    ap.add_argument(
        "--vision-dtype",
        default="float32",
        choices=("float32", "bf16", "bfloat16"),
        help="PhyAI PI0 vision tower dtype. LeRobot path is unchanged.",
    )
    ap.add_argument(
        "--phyai-attn-backend",
        default="flashinfer",
        choices=("auto", "flashinfer"),
        help="PhyAI attention backend. PI0 AR attention currently supports flashinfer here.",
    )
    ap.add_argument(
        "--tokenizer-name",
        default=None,
        help="Recorded in metadata only; tokens are read from the payload.",
    )
    ap.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Denoising steps. Defaults to payload meta/config num_steps.",
    )
    ap.add_argument(
        "--include-layers",
        action="store_true",
        help="Also dump per-layer prefix and expert outputs.",
    )
    ap.add_argument(
        "--layer0-detail",
        "--layer0detail",
        action="store_true",
        help=(
            "Dump detailed prefix layer-0 intermediates without requiring all "
            "prefix/expert layer outputs."
        ),
    )
    ap.add_argument(
        "--layer1-detail",
        "--layer1detail",
        action="store_true",
        help=(
            "Dump detailed prefix layer-1 intermediates without requiring all "
            "prefix/expert layer outputs."
        ),
    )
    ap.add_argument(
        "--detail-layers",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Prefix layer indices whose detailed intermediates should be dumped, "
            "for example: --detail-layers 0 1."
        ),
    )
    ap.add_argument(
        "--vision-detail",
        action="store_true",
        help=(
            "Dump SigLIP input, patch embedding, embedding output, every vision "
            "encoder layer, post-layernorm, and multimodal projector output."
        ),
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.999,
        help="Cosine below this value marks the first divergence.",
    )
    ap.add_argument(
        "--first-bad-only",
        action="store_true",
        help="Print only rows up to and including the first divergence.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path for a combined dump containing lerobot, phyai, and compare rows.",
    )
    args = ap.parse_args()
    detail_layers = set(args.detail_layers or ())
    if args.layer0_detail:
        detail_layers.add(0)
    if args.layer1_detail:
        detail_layers.add(1)
    if any(layer_idx < 0 for layer_idx in detail_layers):
        ap.error("--detail-layers values must be non-negative")
    args.detail_layers = tuple(sorted(detail_layers))
    return args


def load_payload(path: Path) -> dict[str, Any]:
    """Load a tensor-only payload produced by the live comparison tool."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected payload to be a dict, got {type(payload)!r}.")
    return payload


def cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    if hasattr(value, "last_hidden_state") and torch.is_tensor(value.last_hidden_state):
        return value.last_hidden_state
    return None


def patch_method(
    patches: list[tuple[Any, str, Any]], obj: Any, name: str, new_value: Any
) -> None:
    old_value = getattr(obj, name)
    setattr(obj, name, new_value)
    patches.append((obj, name, old_value))


def restore_patches(patches: list[tuple[Any, str, Any]]) -> None:
    for obj, name, old_value in reversed(patches):
        setattr(obj, name, old_value)


def payload_num_steps(payload: dict[str, Any], fallback: int) -> int:
    meta = payload.get("meta")
    if isinstance(meta, dict):
        value = meta.get("num_steps")
        if value is not None:
            return int(value)
    return fallback


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Expected payload[{key!r}] to be a dict.")
    return value


def pick_tensor(
    mapping: dict[str, Any], aliases: tuple[str, ...], label: str
) -> torch.Tensor:
    for key in aliases:
        value = mapping.get(key)
        if torch.is_tensor(value):
            return value

    available = ", ".join(sorted(mapping))
    raise KeyError(f"Cannot find {label}. Tried {aliases}. Available keys: {available}")


def payload_image_keys(payload: dict[str, Any], fallback_num_images: int) -> list[str]:
    meta = payload.get("meta")
    if isinstance(meta, dict):
        keys = meta.get("image_keys_order")
        if isinstance(keys, (list, tuple)) and all(isinstance(k, str) for k in keys):
            return list(keys)

    raw_batch = require_mapping(payload, "raw_batch")
    keys = [
        key
        for key in raw_batch
        if key.startswith(OBS_IMAGES_PREFIX) and not is_empty_camera_key(key)
    ]
    if len(keys) < fallback_num_images:
        raise KeyError(
            f"Payload has {len(keys)} real image tensors, but config expects "
            f"{fallback_num_images}. Image keys found: {keys}"
        )
    return keys[:fallback_num_images]


def build_pixel_values(
    raw_batch: dict[str, Any], image_keys: list[str]
) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for key in image_keys:
        image = raw_batch.get(key)
        if not torch.is_tensor(image):
            raise KeyError(f"Expected raw_batch[{key!r}] to be a tensor.")

        image = image.detach().to(dtype=torch.float32)
        if image.ndim != 4:
            raise ValueError(
                f"{key} must have shape (B, C, H, W), got {tuple(image.shape)}."
            )
        if image.shape[1] != 3:
            raise ValueError(
                f"{key} must be channels-first RGB, got {tuple(image.shape)}."
            )

        images.append(image.mul(2.0).sub(1.0))

    return torch.stack(images, dim=1)


def build_request(
    payload: dict[str, Any], device: torch.device, image_keys: list[str]
) -> tuple[PI0Request, torch.Tensor]:
    raw_batch = require_mapping(payload, "raw_batch")
    processed_batch = require_mapping(payload, "processed_batch")

    state = raw_batch.get("observation.state")
    if not torch.is_tensor(state):
        raise KeyError("Expected raw_batch['observation.state'] to be a tensor.")

    sample_noise = payload.get("sample_noise")
    if not torch.is_tensor(sample_noise):
        raise KeyError("Expected payload['sample_noise'] to be a tensor.")

    reference_actions = payload.get("actions")
    if not torch.is_tensor(reference_actions):
        raise KeyError("Expected payload['actions'] to be a tensor.")

    input_ids = pick_tensor(processed_batch, TOKEN_ALIASES, "language token ids")
    attention_mask = pick_tensor(
        processed_batch, MASK_ALIASES, "language attention mask"
    )
    lang_lens = attention_mask.to(dtype=torch.long).sum(dim=-1)

    request = PI0Request(
        pixel_values=build_pixel_values(raw_batch, image_keys).to(device=device),
        input_ids=input_ids.to(device=device, dtype=torch.long),
        lang_lens=lang_lens.to(device=device, dtype=torch.long),
        state=state.to(device=device, dtype=torch.float32),
        noise=sample_noise.to(device=device, dtype=torch.float32),
    )
    return request, reference_actions.detach().cpu()


def load_config(checkpoint: Path, payload: dict[str, Any]) -> PI0Config:
    config = load_phyai_config(checkpoint, PI0Config)
    meta = payload.get("meta")
    if isinstance(meta, dict) and "num_steps" in meta:
        config = replace(config, num_inference_steps=int(meta["num_steps"]))
    return config


def tensor_shapes(dump: dict[str, Any]) -> dict[str, list[int]]:
    return {
        key: list(value.shape) for key, value in dump.items() if torch.is_tensor(value)
    }


def dump_lerobot(
    *,
    payload: dict[str, Any],
    checkpoint: Path,
    lerobot_root: Path | None,
    device: torch.device,
    dtype_name: str,
    num_steps: int,
    num_images: int,
    image_keys: list[str],
    include_layers: bool,
    detail_layers: tuple[int, ...],
    vision_detail: bool,
) -> dict[str, Any]:
    add_lerobot_to_path(lerobot_root)
    symbols = import_lerobot_symbols()

    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

    policy = load_lerobot_policy(
        symbols=symbols,
        checkpoint=checkpoint,
        device=device,
        dtype_name=dtype_name,
        num_steps=num_steps,
    )
    expected_image_keys = resolve_lerobot_image_keys(policy, num_images)
    if image_keys != expected_image_keys:
        raise ValueError(
            "Payload image keys do not match checkpoint image features. "
            f"payload={image_keys}, checkpoint={expected_image_keys}"
        )
    trim_lerobot_image_features(policy, image_keys)
    model = policy.model

    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in payload["processed_batch"].items()
    }
    noise = payload["sample_noise"].to(device=device, dtype=torch.float32)
    dump: dict[str, Any] = {}
    patches: list[tuple[Any, str, Any]] = []
    current_step: dict[str, int | None] = {"value": None}
    prefix_real_len: dict[str, int | None] = {"value": None}
    prefix_input_dtype: dict[str, str | None] = {"value": None}
    prefix_pre_cast_dtype: dict[str, str | None] = {"value": None}
    image_pre_io_cast_dtype: dict[str, str | None] = {"value": None}
    vision_capture_enabled = {"value": False}
    vision_captures: dict[str, list[torch.Tensor]] = {}
    vision_input_dtype: dict[str, str | None] = {"value": None}

    paligemma = model.paligemma_with_expert.paligemma.model
    vision_root = paligemma.vision_tower
    vision_model = getattr(vision_root, "vision_model", vision_root)

    def capture_vision_output(module: Any, key: str) -> None:
        orig = module.forward

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            if vision_capture_enabled["value"]:
                hidden = first_tensor(out)
                if hidden is not None:
                    vision_captures.setdefault(key, []).append(cpu(hidden))
            return out

        patch_method(patches, module, "forward", wrapped)

    if vision_detail:
        patch_embedding = vision_model.embeddings.patch_embedding
        orig_patch_embedding_forward = patch_embedding.forward

        def patch_embedding_forward(*args, **kwargs):
            if vision_capture_enabled["value"]:
                vision_input_dtype["value"] = str(args[0].dtype)
                vision_captures.setdefault("vision_input", []).append(cpu(args[0]))
            out = orig_patch_embedding_forward(*args, **kwargs)
            if vision_capture_enabled["value"]:
                vision_captures.setdefault("vision_patch_emb", []).append(cpu(out))
            return out

        patch_method(patches, patch_embedding, "forward", patch_embedding_forward)
        capture_vision_output(vision_model.embeddings, "vision_emb")
        for layer_idx, layer in enumerate(vision_model.encoder.layers):
            capture_vision_output(layer, f"vision_layer{layer_idx}")
        capture_vision_output(vision_model.post_layernorm, "vision_post_norm")
        capture_vision_output(paligemma.multi_modal_projector, "vision_projector")

    def prefix_cpu(tensor: torch.Tensor) -> torch.Tensor:
        real_len = prefix_real_len["value"]
        if real_len is not None and tensor.ndim >= 2:
            tensor = tensor[:, :real_len]
        return cpu(tensor)

    prefix_layers = model.paligemma_with_expert.paligemma.model.language_model.layers
    validate_detail_layers(detail_layers, len(prefix_layers))
    detail_layer_set = set(detail_layers)
    layer0 = prefix_layers[0]
    orig_layer0_forward = layer0.forward

    def layer0_forward(*args, **kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        prefix_input_dtype["value"] = str(hidden.dtype)
        dump["prefix_embs"] = prefix_cpu(hidden)
        out = orig_layer0_forward(*args, **kwargs)
        if include_layers or 0 in detail_layer_set:
            layer_out = first_tensor(out)
            if layer_out is not None:
                dump["prefix_layer0"] = prefix_cpu(layer_out)
        return out

    patch_method(patches, layer0, "forward", layer0_forward)

    for layer_idx, layer in enumerate(prefix_layers[1:], start=1):
        if not include_layers and layer_idx not in detail_layer_set:
            continue
        orig_forward = layer.forward

        def make_prefix_forward(idx, orig):
            def wrapped(*args, **kwargs):
                out = orig(*args, **kwargs)
                hidden = first_tensor(out)
                if hidden is not None:
                    dump[f"prefix_layer{idx}"] = prefix_cpu(hidden)
                return out

            return wrapped

        patch_method(
            patches, layer, "forward", make_prefix_forward(layer_idx, orig_forward)
        )

    if detail_layers:
        from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

        def install_layer_detail(layer_idx: int, layer: Any) -> None:
            key_prefix = f"prefix_layer{layer_idx}"
            projected: dict[str, torch.Tensor] = {}

            def wrap_norm(
                module: Any, key: str, *, input_key: str | None = None
            ) -> None:
                orig = module.forward

                def wrapped(*args, **kwargs):
                    if input_key is not None:
                        dump[input_key] = prefix_cpu(args[0])
                    out = orig(*args, **kwargs)
                    hidden = first_tensor(out)
                    if hidden is not None:
                        dump[key] = prefix_cpu(hidden)
                    return out

                patch_method(patches, module, "forward", wrapped)

            wrap_norm(layer.input_layernorm, f"{key_prefix}_input_norm")
            wrap_norm(
                layer.post_attention_layernorm,
                f"{key_prefix}_post_norm",
                input_key=f"{key_prefix}_attn_residual",
            )

            for part, projection in (
                ("q", layer.self_attn.q_proj),
                ("k", layer.self_attn.k_proj),
                ("v", layer.self_attn.v_proj),
            ):
                orig_projection = projection.forward

                def make_projection_forward(name, orig):
                    def wrapped(*args, **kwargs):
                        out = orig(*args, **kwargs)
                        projected[name] = out
                        if all(item in projected for item in ("q", "k", "v")):
                            dump[f"{key_prefix}_qkv"] = prefix_cpu(
                                torch.cat(
                                    [
                                        projected["q"],
                                        projected["k"],
                                        projected["v"],
                                    ],
                                    dim=-1,
                                )
                            )
                        return out

                    return wrapped

                patch_method(
                    patches,
                    projection,
                    "forward",
                    make_projection_forward(part, orig_projection),
                )

            orig_attention_forward = layer.self_attn.forward

            def attention_forward(*args, **kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                out = orig_attention_forward(*args, **kwargs)
                q_raw = projected.get("q")
                k_raw = projected.get("k")
                v_raw = projected.get("v")
                position_embeddings = kwargs.get("position_embeddings")
                if (
                    q_raw is not None
                    and k_raw is not None
                    and v_raw is not None
                    and position_embeddings is not None
                ):
                    hidden_shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
                    q = q_raw.view(hidden_shape).transpose(1, 2)
                    k = k_raw.view(hidden_shape).transpose(1, 2)
                    v = v_raw.view(hidden_shape).transpose(1, 2)
                    cos, sin = position_embeddings
                    q, k = apply_rotary_pos_emb(q, k, cos, sin)
                    dump[f"{key_prefix}_q_rope"] = prefix_cpu(
                        q.transpose(1, 2).contiguous()
                    )
                    dump[f"{key_prefix}_k_rope"] = prefix_cpu(
                        k.transpose(1, 2).contiguous()
                    )
                    dump[f"{key_prefix}_v"] = prefix_cpu(v.transpose(1, 2).contiguous())
                return out

            patch_method(patches, layer.self_attn, "forward", attention_forward)

            orig_o_proj_forward = layer.self_attn.o_proj.forward

            def o_proj_forward(*args, **kwargs):
                dump[f"{key_prefix}_attn"] = prefix_cpu(args[0])
                out = orig_o_proj_forward(*args, **kwargs)
                dump[f"{key_prefix}_o_proj"] = prefix_cpu(out)
                return out

            patch_method(patches, layer.self_attn.o_proj, "forward", o_proj_forward)

            orig_mlp_forward = layer.mlp.forward

            def mlp_forward(*args, **kwargs):
                out = orig_mlp_forward(*args, **kwargs)
                dump[f"{key_prefix}_mlp"] = prefix_cpu(out)
                return out

            patch_method(patches, layer.mlp, "forward", mlp_forward)

        for layer_idx in detail_layers:
            install_layer_detail(layer_idx, prefix_layers[layer_idx])

    if include_layers:
        expert_layers = model.paligemma_with_expert.gemma_expert.model.layers
        for layer_idx, layer in enumerate(expert_layers):
            orig_forward = layer.forward

            def make_expert_forward(idx, orig):
                def wrapped(*args, **kwargs):
                    out = orig(*args, **kwargs)
                    step = current_step["value"]
                    hidden = first_tensor(out)
                    if step is not None and hidden is not None:
                        dump[f"expert_step{step}_layer{idx}"] = cpu(hidden)
                    return out

                return wrapped

            patch_method(
                patches, layer, "forward", make_expert_forward(layer_idx, orig_forward)
            )

        expert_norm = model.paligemma_with_expert.gemma_expert.model.norm
        orig_norm_forward = expert_norm.forward

        def expert_norm_forward(*args, **kwargs):
            out = orig_norm_forward(*args, **kwargs)
            step = current_step["value"]
            hidden = first_tensor(out)
            if step is not None and hidden is not None:
                dump[f"expert_step{step}_norm"] = cpu(
                    hidden[:, -policy.config.chunk_size :]
                )
            return out

        patch_method(patches, expert_norm, "forward", expert_norm_forward)

    try:
        with torch.no_grad():
            images, img_masks = policy._preprocess_images(batch)
            lang_tokens = batch[OBS_LANGUAGE_TOKENS]
            lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            state = policy.prepare_state(batch)
            present_img_keys = [
                key for key in policy.config.image_features if key in batch
            ]

            vision_capture_enabled["value"] = vision_detail
            try:
                img_embs = [
                    model.paligemma_with_expert.embed_image(img) for img in images
                ]
            finally:
                vision_capture_enabled["value"] = False
            for key, tensors in vision_captures.items():
                dump[key] = torch.stack(tensors, dim=1)
            dump["img_emb"] = torch.stack([cpu(emb) for emb in img_embs], dim=1)
            dump["img_emb_pre_io_cast"] = dump["img_emb"]
            if img_embs:
                image_pre_io_cast_dtype["value"] = str(img_embs[0].dtype)
            dump["lang_emb"] = cpu(
                model.paligemma_with_expert.embed_language_tokens(lang_tokens)
            )

            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
            )
            prefix_real_len["value"] = int(
                prefix_pad_masks.to(dtype=torch.int64).sum(dim=1).max().item()
            )
            prefix_pre_cast_dtype["value"] = str(prefix_embs.dtype)
            dump["prefix_embs_pre_cast"] = prefix_cpu(prefix_embs)

            prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            mask4d = model._prepare_attention_masks_4d(prefix_att_2d)  # noqa: SLF001
            lm = model.paligemma_with_expert.paligemma.model.language_model
            lm.config._attn_implementation = "eager"  # noqa: SLF001

            (prefix_out, _), past_key_values = model.paligemma_with_expert.forward(
                attention_mask=mask4d,
                position_ids=prefix_pos_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            dump["prefix_hidden"] = prefix_cpu(prefix_out)
            if include_layers:
                dump["prefix_norm"] = prefix_cpu(prefix_out)

            bsize = int(state.shape[0])
            time = torch.ones(bsize, dtype=torch.float32, device=device)
            suffix_embs, _, _, _ = model.embed_suffix(state, noise, time)
            dump["x_t_step0"] = cpu(noise)
            dump["state_emb"] = cpu(suffix_embs[:, 0])
            dump["action_time_emb"] = cpu(suffix_embs[:, 1:])

            dt = -1.0 / num_steps
            x_t = noise
            for step in range(num_steps):
                if include_layers and step != 0:
                    dump[f"x_t_step{step}"] = cpu(x_t)
                current_step["value"] = step
                t = 1.0 + step * dt
                time = torch.full((bsize,), t, dtype=torch.float32, device=device)
                v_t = model.denoise_step(
                    state=state,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=time,
                )
                current_step["value"] = None
                dump[f"v_t_step{step}"] = cpu(v_t)
                x_t = x_t + dt * v_t
                if include_layers:
                    dump[f"x_t_after_step{step}"] = cpu(x_t)

            dump["actions"] = cpu(x_t)
    finally:
        current_step["value"] = None
        restore_patches(patches)

    dump["meta"] = {
        "side": "lerobot",
        "model_id": str(checkpoint),
        "dtype": dtype_name,
        "image_keys_order": present_img_keys,
        "tokenizer_len": int(lang_tokens.shape[1]),
        "chunk_size": int(policy.config.chunk_size),
        "num_steps": num_steps,
        "include_layers": include_layers,
        "detail_layers": list(detail_layers),
        "layer0_detail": 0 in detail_layer_set,
        "layer1_detail": 1 in detail_layer_set,
        "img_emb_pre_io_cast_dtype": image_pre_io_cast_dtype["value"],
        "num_vision_layers": len(vision_model.encoder.layers),
        "vision_input_dtype": vision_input_dtype["value"],
        "prefix_embs_pre_cast_dtype": prefix_pre_cast_dtype["value"],
        "prefix_embs_dtype": prefix_input_dtype["value"],
        "num_prefix_layers": len(
            model.paligemma_with_expert.paligemma.model.language_model.layers
        ),
        "num_expert_layers": len(model.paligemma_with_expert.gemma_expert.model.layers),
    }
    return dump


def dump_phyai(
    *,
    payload: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    device_target: str,
    dtype: torch.dtype,
    dtype_name: str,
    vision_dtype: torch.dtype,
    vision_dtype_name: str,
    attn_backend: str,
    num_steps: int,
    image_keys: list[str],
    include_layers: bool,
    detail_layers: tuple[int, ...],
    vision_detail: bool,
) -> dict[str, Any]:
    request, reference_actions = build_request(payload, device, image_keys)
    batch_size = int(reference_actions.shape[0])
    config = replace(load_config(checkpoint, payload), num_inference_steps=num_steps)

    engine = Engine(
        EngineArgs(
            plugin="pi0",
            plugin_args=PI0Args(
                checkpoint_dir=checkpoint,
                config=config,
                max_batch_size=batch_size,
                weight_strict=True,
                vision_params_dtype=vision_dtype,
            ),
            config=EngineConfig(
                backends=BackendConfig(attn=attn_backend),
                device=DeviceConfig(target=device_target, params_dtype=dtype),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )

    sched = engine.entry.scheduler
    if (
        sched.llm_runner.graph is not None
        or sched.expert_runner.state_graph is not None
        or sched.expert_runner.action_graph is not None
    ):
        raise RuntimeError(
            "CUDA graph is active; wrappers cannot capture. Disable cuda graph."
        )

    dump: dict[str, Any] = {}
    patches: list[tuple[Any, str, Any]] = []
    vis_outs: list[torch.Tensor] = []
    vis_pre_io_cast_outs: list[torch.Tensor] = []
    layer_patches: list[tuple[Any, str, Any]] = []

    import phyai.models.pi0.scheduler_ws1_pi0 as sched_mod

    orig_pack = sched_mod.pack_prefix_per_sample_padded
    current_step: dict[str, int | None] = {"value": None}
    cached_state: dict[int, torch.Tensor] = {}
    prefix_input_dtype: dict[str, str | None] = {"value": None}
    prefix_pre_cast_dtype: dict[str, str | None] = {"value": None}
    image_pre_io_cast_dtype: dict[str, str | None] = {"value": None}
    vision_captures: dict[str, list[torch.Tensor]] = {}
    vision_input_dtype: dict[str, str | None] = {"value": None}

    vision_model = sched.model.vision.vision_tower.vision_model

    def capture_vision_output(module: Any, key: str) -> None:
        orig = module.forward

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            hidden = first_tensor(out)
            if hidden is not None:
                vision_captures.setdefault(key, []).append(cpu(hidden))
            return out

        patch_method(patches, module, "forward", wrapped)

    if vision_detail:
        patch_embedding = vision_model.embeddings.patch_embedding
        orig_patch_embedding_forward = patch_embedding.forward

        def patch_embedding_forward(*args, **kwargs):
            vision_input_dtype["value"] = str(args[0].dtype)
            vision_captures.setdefault("vision_input", []).append(cpu(args[0]))
            out = orig_patch_embedding_forward(*args, **kwargs)
            vision_captures.setdefault("vision_patch_emb", []).append(cpu(out))
            return out

        patch_method(patches, patch_embedding, "forward", patch_embedding_forward)
        capture_vision_output(vision_model.embeddings, "vision_emb")
        for layer_idx, layer in enumerate(vision_model.encoder.layers):
            capture_vision_output(layer, f"vision_layer{layer_idx}")
        capture_vision_output(vision_model.post_layernorm, "vision_post_norm")

    projector = sched.model.vision.multi_modal_projector
    orig_projector_fwd = projector.forward

    def projector_fwd(*args, **kwargs):
        out = orig_projector_fwd(*args, **kwargs)
        image_pre_io_cast_dtype["value"] = str(out.dtype)
        vis_pre_io_cast_outs.append(cpu(out))
        if vision_detail:
            vision_captures.setdefault("vision_projector", []).append(cpu(out))
        return out

    patch_method(patches, projector, "forward", projector_fwd)

    def vis_fwd(batch):
        out = orig_vis_fwd(batch)
        vis_outs.append(cpu(out))
        return out

    orig_vis_fwd = sched.vision_runner.forward
    sched.vision_runner.forward = vis_fwd

    lm = sched.model.paligemma_lm
    orig_embed_lang = lm.embed_lang

    def embed_lang(input_ids):
        out = orig_embed_lang(input_ids)
        dump["lang_emb"] = cpu(out)
        return out

    lm.embed_lang = embed_lang

    def pack(*args, **kwargs):
        out = orig_pack(*args, **kwargs)
        prefix_pre_cast_dtype["value"] = str(out.dtype)
        dump["packed_flat"] = cpu(out)
        return out

    sched_mod.pack_prefix_per_sample_padded = pack

    validate_detail_layers(detail_layers, len(lm.layers))
    detail_layer_set = set(detail_layers)
    layer0 = lm.layers[0]
    orig_layer0_forward = layer0.forward

    def layer0_forward(h, *args, **kwargs):
        prefix_input_dtype["value"] = str(h.dtype)
        dump["prefix_input_flat"] = cpu(h)
        out = orig_layer0_forward(h, *args, **kwargs)
        if include_layers or 0 in detail_layer_set:
            dump["prefix_layer0_flat"] = cpu(out)
        return out

    patch_method(layer_patches, layer0, "forward", layer0_forward)

    for layer_idx, layer in enumerate(lm.layers[1:], start=1):
        if not include_layers and layer_idx not in detail_layer_set:
            continue
        orig_forward = layer.forward

        def make_prefix_forward(idx, orig):
            def wrapped(*args, **kwargs):
                out = orig(*args, **kwargs)
                dump[f"prefix_layer{idx}_flat"] = cpu(out)
                return out

            return wrapped

        patch_method(
            layer_patches,
            layer,
            "forward",
            make_prefix_forward(layer_idx, orig_forward),
        )

    if detail_layers:

        def wrap_tensor_output(
            module: Any, key: str, *, input_key: str | None = None
        ) -> None:
            orig = module.forward

            def wrapped(*args, **kwargs):
                if input_key is not None:
                    dump[input_key] = cpu(args[0])
                out = orig(*args, **kwargs)
                hidden = first_tensor(out)
                if hidden is not None:
                    dump[key] = cpu(hidden)
                return out

            patch_method(layer_patches, module, "forward", wrapped)

        def install_layer_detail(layer_idx: int, layer: Any) -> None:
            key_prefix = f"prefix_layer{layer_idx}"
            wrap_tensor_output(
                layer.input_layernorm,
                f"{key_prefix}_input_norm_flat",
            )
            wrap_tensor_output(
                layer.qkv_proj,
                f"{key_prefix}_qkv_flat",
            )

            orig_attention_forward = layer.attn.forward

            def attention_forward(q, k, v, *args, **kwargs):
                dump[f"{key_prefix}_q_rope_flat"] = cpu(q)
                dump[f"{key_prefix}_k_rope_flat"] = cpu(k)
                dump[f"{key_prefix}_v_flat"] = cpu(v)
                out = orig_attention_forward(q, k, v, *args, **kwargs)
                dump[f"{key_prefix}_attn_flat"] = cpu(out.reshape(*out.shape[:-2], -1))
                return out

            patch_method(layer_patches, layer.attn, "forward", attention_forward)

            wrap_tensor_output(
                layer.o_proj,
                f"{key_prefix}_o_proj_flat",
            )
            wrap_tensor_output(
                layer.post_attention_layernorm,
                f"{key_prefix}_post_norm_flat",
                input_key=f"{key_prefix}_attn_residual_flat",
            )
            wrap_tensor_output(
                layer.mlp,
                f"{key_prefix}_mlp_flat",
            )

        for layer_idx in detail_layers:
            install_layer_detail(layer_idx, lm.layers[layer_idx])

    if include_layers:
        runner = sched.expert_runner
        for layer_idx, layer in enumerate(runner.expert_stack.layers):
            orig_forward = layer.forward

            def make_expert_forward(idx, orig):
                def wrapped(h, *args, **kwargs):
                    out = orig(h, *args, **kwargs)
                    step = current_step["value"]
                    if out.shape[0] == runner.batch_size:
                        cached_state[idx] = cpu(out.view(runner.batch_size, 1, -1))
                    elif (
                        step is not None
                        and out.shape[0] == runner.batch_size * runner.chunk_size
                    ):
                        state_out = cached_state.get(idx)
                        action_out = cpu(
                            out.view(runner.batch_size, runner.chunk_size, -1)
                        )
                        if state_out is not None:
                            dump[f"expert_step{step}_layer{idx}"] = torch.cat(
                                [state_out, action_out],
                                dim=1,
                            )
                    return out

                return wrapped

            patch_method(
                layer_patches,
                layer,
                "forward",
                make_expert_forward(layer_idx, orig_forward),
            )

        orig_norm_forward = runner.expert_stack.norm.forward

        def expert_norm_forward(h, *args, **kwargs):
            out = orig_norm_forward(h, *args, **kwargs)
            step = current_step["value"]
            if step is not None:
                dump[f"expert_step{step}_norm"] = cpu(
                    out.view(runner.batch_size, runner.chunk_size, -1)
                )
            return out

        patch_method(
            layer_patches, runner.expert_stack.norm, "forward", expert_norm_forward
        )

    orig_llm_fwd = sched.llm_runner._fwd

    def llm_fwd(**kwargs):
        out = orig_llm_fwd(**kwargs)
        dump["prefix_hidden_flat"] = cpu(out)
        if include_layers:
            dump["prefix_norm_flat"] = cpu(out)
        return out

    sched.llm_runner._fwd = llm_fwd

    heads = sched.expert_runner.heads
    orig_embed_state = heads.embed_state
    orig_fuse_action_time = heads.fuse_action_time

    def embed_state(state):
        out = orig_embed_state(state)
        dump.setdefault("state_emb", cpu(out))
        return out

    def fuse_action_time(x_t, time_emb):
        out = orig_fuse_action_time(x_t, time_emb)
        if "action_time_emb" not in dump:
            dump["action_time_emb"] = cpu(out)
        return out

    heads.embed_state = embed_state
    heads.fuse_action_time = fuse_action_time

    orig_one_step = sched.expert_runner._one_step
    dt = -1.0 / num_steps

    def one_step(x_t, step):
        if include_layers or step == 0:
            dump[f"x_t_step{step}"] = cpu(x_t[:batch_size])
        current_step["value"] = step
        out = orig_one_step(x_t, step)
        current_step["value"] = None
        dump[f"v_t_step{step}"] = cpu(out[:batch_size])
        if include_layers:
            x_after = x_t[:batch_size] + dt * out[:batch_size].to(x_t.dtype)
            dump[f"x_t_after_step{step}"] = cpu(x_after)
        return out

    sched.expert_runner._one_step = one_step

    try:
        with torch.inference_mode():
            actions = engine.step(request).detach().float().cpu()
    finally:
        current_step["value"] = None
        sched_mod.pack_prefix_per_sample_padded = orig_pack
        sched.vision_runner.forward = orig_vis_fwd
        lm.embed_lang = orig_embed_lang
        sched.llm_runner._fwd = orig_llm_fwd
        heads.embed_state = orig_embed_state
        heads.fuse_action_time = orig_fuse_action_time
        sched.expert_runner._one_step = orig_one_step
        restore_patches(layer_patches)
        restore_patches(patches)
        close = getattr(engine, "close", None)
        if close is not None:
            close()

    B = batch_size
    n_per_sample = sched.n_per_sample
    n_img = sched.image_token_count
    lang_len = int(request.lang_lens.max())
    real_len = n_img + lang_len

    dump["img_emb"] = torch.stack(vis_outs, dim=0)[:B]
    dump["img_emb_pre_io_cast"] = torch.stack(vis_pre_io_cast_outs, dim=0)[:B]
    for key, tensors in vision_captures.items():
        dump[key] = torch.stack(tensors, dim=0)[:B]
    dump["lang_emb"] = dump["lang_emb"][:B]
    packed = dump.pop("packed_flat").view(-1, n_per_sample, dump["lang_emb"].shape[-1])
    prefix_input = dump.pop("prefix_input_flat").view(
        -1, n_per_sample, dump["lang_emb"].shape[-1]
    )
    dump["prefix_embs_pre_cast"] = packed[:B, :real_len]
    dump["prefix_embs"] = prefix_input[:B, :real_len]
    hidden = dump.pop("prefix_hidden_flat").view(-1, n_per_sample, packed.shape[-1])
    dump["prefix_hidden"] = hidden[:B, :real_len]
    if include_layers:
        prefix_norm = dump.pop("prefix_norm_flat").view(
            -1, n_per_sample, packed.shape[-1]
        )
        dump["prefix_norm"] = prefix_norm[:B, :real_len]
        for key in list(dump):
            layer_index = key.removeprefix("prefix_layer").removesuffix("_flat")
            if (
                key.startswith("prefix_layer")
                and key.endswith("_flat")
                and layer_index.isdigit()
            ):
                layer_name = key.removesuffix("_flat")
                layer = dump.pop(key).view(-1, n_per_sample, packed.shape[-1])
                dump[layer_name] = layer[:B, :real_len]
        for key in list(dump):
            if key.startswith("expert_step") and "_layer" in key:
                dump[key] = dump[key][:B]
            elif key.startswith("expert_step") and key.endswith("_norm"):
                dump[key] = dump[key][:B]
    else:
        for layer_idx in detail_layers:
            key = f"prefix_layer{layer_idx}"
            layer = dump.pop(f"{key}_flat").view(-1, n_per_sample, packed.shape[-1])
            dump[key] = layer[:B, :real_len]

    for layer_idx in detail_layers:
        key_prefix = f"prefix_layer{layer_idx}"
        for suffix, _ in DETAIL_CUT_SUFFIXES:
            flat_key = f"{key_prefix}_{suffix}_flat"
            tensor = dump.pop(flat_key)
            tensor = tensor.view(-1, n_per_sample, *tensor.shape[1:])
            dump[f"{key_prefix}_{suffix}"] = tensor[:B, :real_len]
    dump["state_emb"] = dump["state_emb"][:B]
    dump["x_t_step0"] = dump["x_t_step0"][:B]
    dump["action_time_emb"] = dump["action_time_emb"][:B]
    for key in list(dump):
        if key.startswith("v_t_step") or key.startswith("x_t_after_step"):
            dump[key] = dump[key][:B]
    dump["actions"] = actions

    dump["meta"] = {
        "side": "phyai",
        "dtype": dtype_name,
        "vision_dtype": vision_dtype_name,
        "attn_backend": attn_backend,
        "image_keys_order": image_keys,
        "n_per_sample": n_per_sample,
        "n_img": n_img,
        "lang_len": lang_len,
        "num_steps": config.num_inference_steps,
        "include_layers": include_layers,
        "detail_layers": list(detail_layers),
        "layer0_detail": 0 in detail_layer_set,
        "layer1_detail": 1 in detail_layer_set,
        "img_emb_pre_io_cast_dtype": image_pre_io_cast_dtype["value"],
        "num_vision_layers": len(vision_model.encoder.layers),
        "vision_input_dtype": vision_input_dtype["value"],
        "prefix_embs_pre_cast_dtype": prefix_pre_cast_dtype["value"],
        "prefix_embs_dtype": prefix_input_dtype["value"],
        "num_prefix_layers": len(sched.model.paligemma_lm.layers),
        "num_expert_layers": len(sched.expert_runner.expert_stack.layers),
    }
    return dump


def align_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if left.shape == right.shape:
        return left, right, False
    if left.ndim != right.ndim:
        raise ValueError(f"rank mismatch {tuple(left.shape)} vs {tuple(right.shape)}")
    slices = tuple(slice(0, min(da, db)) for da, db in zip(left.shape, right.shape))
    return left[slices], right[slices], True


def compare_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left, right, clipped = align_tensors(left.float(), right.float())
    diff = left - right
    abs_diff = diff.abs()
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    return {
        "shape": list(left.shape),
        "clipped": clipped,
        "cosine": float(F.cosine_similarity(left_flat, right_flat, dim=0)),
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "l2_lerobot": float(left_flat.norm()),
        "l2_phyai": float(right_flat.norm()),
    }


def build_compare_cuts(
    lerobot_dump: dict[str, Any],
    phyai_dump: dict[str, Any],
    *,
    include_layers: bool,
    detail_layers: tuple[int, ...],
    vision_detail: bool,
) -> list[tuple[str, str]]:
    num_steps = int(
        min(
            lerobot_dump.get("meta", {}).get("num_steps", 1),
            phyai_dump.get("meta", {}).get("num_steps", 1),
        )
    )
    prefix_layers = int(
        min(
            lerobot_dump.get("meta", {}).get("num_prefix_layers", 0),
            phyai_dump.get("meta", {}).get("num_prefix_layers", 0),
        )
    )
    expert_layers = int(
        min(
            lerobot_dump.get("meta", {}).get("num_expert_layers", 0),
            phyai_dump.get("meta", {}).get("num_expert_layers", 0),
        )
    )
    vision_layers = int(
        min(
            lerobot_dump.get("meta", {}).get("num_vision_layers", 0),
            phyai_dump.get("meta", {}).get("num_vision_layers", 0),
        )
    )

    cuts: list[tuple[str, str]] = []
    if vision_detail:
        cuts.extend(
            [
                ("vision_input", "SigLIP pixel input after compute-dtype cast"),
                ("vision_patch_emb", "SigLIP patch convolution output"),
                ("vision_emb", "SigLIP patch + position embeddings"),
            ]
        )
        cuts.extend(
            (f"vision_layer{i}", f"SigLIP encoder layer {i} output")
            for i in range(vision_layers)
        )
        cuts.extend(
            [
                ("vision_post_norm", "SigLIP post-layernorm output"),
                ("vision_projector", "PaliGemma multimodal projector output"),
            ]
        )
    cuts.extend(
        [
            ("img_emb_pre_io_cast", "image tokens before model IO dtype cast"),
            ("img_emb", "image tokens (SigLIP + projector)"),
            ("lang_emb", "language token embeddings"),
            ("prefix_embs", "packed prefix after Gemma input dtype cast"),
        ]
    )
    for layer_idx in detail_layers:
        cuts.extend(detail_cuts(layer_idx))
    if include_layers:
        cuts.extend(
            (f"prefix_layer{i}", f"prefix layer {i} output")
            for i in range(prefix_layers)
        )
        cuts.append(("prefix_norm", "prefix final norm output"))
    cuts.extend(
        [
            ("prefix_hidden", "LLM prefix output (post norm)"),
            ("x_t_step0", "noise fed to expert at step 0"),
            ("state_emb", "state token embedding"),
            ("action_time_emb", "action+time token embeddings"),
        ]
    )
    for step in range(num_steps):
        if include_layers and step != 0:
            cuts.append((f"x_t_step{step}", f"denoise step {step} input x_t"))
        if include_layers:
            cuts.extend(
                (
                    f"expert_step{step}_layer{i}",
                    f"denoise step {step} expert layer {i} output",
                )
                for i in range(expert_layers)
            )
            cuts.append(
                (f"expert_step{step}_norm", f"denoise step {step} expert final norm")
            )
        cuts.append((f"v_t_step{step}", f"denoise step {step} velocity"))
        if include_layers:
            cuts.append(
                (f"x_t_after_step{step}", f"denoise step {step} Euler-updated x_t")
            )
    cuts.append(("actions", "final action chunk"))

    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for key, desc in cuts:
        if key not in seen:
            seen.add(key)
            deduped.append((key, desc))
    return deduped


def compare_dumps(
    lerobot_dump: dict[str, Any],
    phyai_dump: dict[str, Any],
    *,
    include_layers: bool,
    detail_layers: tuple[int, ...],
    vision_detail: bool,
    threshold: float,
) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    first_divergence: str | None = None
    for key, desc in build_compare_cuts(
        lerobot_dump,
        phyai_dump,
        include_layers=include_layers,
        detail_layers=detail_layers,
        vision_detail=vision_detail,
    ):
        row: dict[str, Any] = {
            "key": key,
            "description": desc,
            "present_lerobot": key in lerobot_dump,
            "present_phyai": key in phyai_dump,
        }
        if key in lerobot_dump and key in phyai_dump:
            row.update(compare_metrics(lerobot_dump[key], phyai_dump[key]))
            if row["cosine"] < threshold and first_divergence is None:
                first_divergence = key
                row["first_divergence"] = True
            else:
                row["first_divergence"] = False
        rows.append(row)
    return rows, first_divergence


def print_compare_table(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
    first_bad_only: bool,
) -> None:
    printable_rows = rows
    if first_bad_only:
        first_index = next(
            (idx for idx, row in enumerate(rows) if row.get("first_divergence")),
            None,
        )
        if first_index is not None:
            printable_rows = rows[: first_index + 1]

    print(
        f"{'cut':32s} {'cosine':>9s} {'max_abs':>10s} {'mean_abs':>10s} "
        f"{'rmse':>10s} {'l2(lerobot)':>12s} {'l2(phyai)':>12s}"
    )
    print("-" * 104)
    for row in printable_rows:
        key = row["key"]
        if not row.get("present_lerobot") or not row.get("present_phyai"):
            print(
                f"{key:32s} {'missing':>9s} "
                f"(lerobot={row.get('present_lerobot')}, phyai={row.get('present_phyai')})"
            )
            continue
        flag = ""
        if row.get("first_divergence"):
            flag = f"  <-- FIRST DIVERGENCE (< {threshold})"
        elif row["cosine"] < threshold:
            flag = "  (bad)"
        clip = " [shape-clipped]" if row.get("clipped") else ""
        print(
            f"{key:32s} {row['cosine']:>9.5f} {row['max_abs']:>10.4f} "
            f"{row['mean_abs']:>10.4f} {row['rmse']:>10.4f} "
            f"{row['l2_lerobot']:>12.3f} {row['l2_phyai']:>12.3f}{flag}{clip}"
        )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_dir():
        raise NotADirectoryError(
            f"--checkpoint must be a directory, got {args.checkpoint}"
        )
    if not args.pt.is_file():
        raise FileNotFoundError(f"--pt must be a file, got {args.pt}")

    payload = load_payload(args.pt)

    device = cuda_device_from_name(args.device)
    torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)
    vision_dtype = dtype_from_name(args.vision_dtype)
    attn_backend = choose_attn_backend(args.phyai_attn_backend)
    config = load_config(args.checkpoint, payload)
    num_steps = (
        args.num_steps
        if args.num_steps is not None
        else payload_num_steps(payload, config.num_inference_steps)
    )
    if num_steps <= 0:
        raise ValueError(f"--num-steps must be positive, got {num_steps}")
    config = replace(config, num_inference_steps=num_steps)
    image_keys = payload_image_keys(payload, config.num_images)

    print("dumping LeRobot...")
    lerobot_dump = dump_lerobot(
        payload=payload,
        checkpoint=args.checkpoint,
        lerobot_root=args.lerobot_root,
        device=device,
        dtype_name=args.dtype,
        num_steps=num_steps,
        num_images=config.num_images,
        image_keys=image_keys,
        include_layers=args.include_layers,
        detail_layers=args.detail_layers,
        vision_detail=args.vision_detail,
    )
    print("dumping PhyAI...")
    phyai_dump = dump_phyai(
        payload=payload,
        checkpoint=args.checkpoint,
        device=device,
        device_target=str(device),
        dtype=dtype,
        dtype_name=args.dtype,
        vision_dtype=vision_dtype,
        vision_dtype_name=args.vision_dtype,
        attn_backend=attn_backend,
        num_steps=num_steps,
        image_keys=image_keys,
        include_layers=args.include_layers,
        detail_layers=args.detail_layers,
        vision_detail=args.vision_detail,
    )

    rows, first_divergence = compare_dumps(
        lerobot_dump,
        phyai_dump,
        include_layers=args.include_layers,
        detail_layers=args.detail_layers,
        vision_detail=args.vision_detail,
        threshold=args.threshold,
    )

    print(
        f"lerobot image order: {lerobot_dump.get('meta', {}).get('image_keys_order')}"
    )
    print(f"phyai   image order: {phyai_dump.get('meta', {}).get('image_keys_order')}")
    if args.vision_detail:
        print(
            "vision input dtype: "
            f"lerobot={lerobot_dump.get('meta', {}).get('vision_input_dtype')}, "
            f"phyai={phyai_dump.get('meta', {}).get('vision_input_dtype')}"
        )
    print(
        "image pre-IO-cast dtype: "
        f"lerobot={lerobot_dump.get('meta', {}).get('img_emb_pre_io_cast_dtype')}, "
        f"phyai={phyai_dump.get('meta', {}).get('img_emb_pre_io_cast_dtype')}"
    )
    print(
        "lerobot prefix dtype: "
        f"{lerobot_dump.get('meta', {}).get('prefix_embs_pre_cast_dtype')} -> "
        f"{lerobot_dump.get('meta', {}).get('prefix_embs_dtype')}"
    )
    print(
        "phyai   prefix dtype: "
        f"{phyai_dump.get('meta', {}).get('prefix_embs_pre_cast_dtype')} -> "
        f"{phyai_dump.get('meta', {}).get('prefix_embs_dtype')}"
    )
    print_compare_table(
        rows, threshold=args.threshold, first_bad_only=args.first_bad_only
    )
    if first_divergence is None:
        print("\nAll requested cuts match.")
    else:
        print(f"\nfirst divergence: {first_divergence}")

    combined = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "pt": str(args.pt),
            "lerobot_root": str(args.lerobot_root) if args.lerobot_root else None,
            "device": args.device,
            "dtype": args.dtype,
            "vision_dtype": args.vision_dtype,
            "phyai_attn_backend": attn_backend,
            "num_steps": num_steps,
            "image_keys_order": image_keys,
            "include_layers": args.include_layers,
            "detail_layers": list(args.detail_layers),
            "layer0_detail": 0 in args.detail_layers,
            "layer1_detail": 1 in args.detail_layers,
            "vision_detail": args.vision_detail,
            "threshold": args.threshold,
            "tokenizer_name": args.tokenizer_name,
        },
        "payload_meta": payload.get("meta", {}),
        "lerobot": lerobot_dump,
        "phyai": phyai_dump,
        "compare": {
            "rows": rows,
            "first_divergence": first_divergence,
        },
    }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(combined, args.out)
        print(f"saved: {args.out}")
    else:
        print(json.dumps({"first_divergence": first_divergence}, indent=2))


if __name__ == "__main__":
    main()
