"""Compare Official and PHYAI preprocessing for one captured RoboTwin request."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from torchvision.transforms.v2 import Resize

from benchmark.lingbot_v2.robotwin.phyai_policy_server import (
    gather_robotwin_action_from_model_slots,
    scatter_robotwin_state_to_model_slots,
)
from benchmark.lingbot_v2.robotwin.replay_first_request import (
    array_metrics,
    array_sha256,
    load_captured_request,
)
from phyai.models.lingbot_v2 import LingBotVLA2Config
from phyai.utils import load_config
from phyai_utils_tools.models.lingbot_v2 import (
    ROBOTWIN_CAMERA_KEYS,
    RoboTwinLingBotV2Adapter,
    canonical_robotwin_stats,
    load_robotwin_stats,
)
from phyai_utils_tools.models.lingbot_v2.processor_lingbotv2 import (
    LingBotV2Processor,
)


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def compare_array(name: str, official: Any, phyai: Any) -> dict[str, Any]:
    left = np.ascontiguousarray(as_numpy(official))
    right = np.ascontiguousarray(as_numpy(phyai))
    result: dict[str, Any] = {
        "official_shape": list(left.shape),
        "phyai_shape": list(right.shape),
        "official_dtype": str(left.dtype),
        "phyai_dtype": str(right.dtype),
        "official_sha256": array_sha256(left),
        "phyai_sha256": array_sha256(right),
    }
    if left.shape != right.shape:
        result["shape_equal"] = False
    elif left.dtype.kind in "biu" and right.dtype.kind in "biu":
        result.update(
            {
                "shape_equal": True,
                "exact": bool(np.array_equal(left, right)),
                "mismatch_count": int(np.count_nonzero(left != right)),
            }
        )
    else:
        result.update({"shape_equal": True, **array_metrics(left, right)})
    print(f"{name:18}: {result}")
    return result


def load_official_preprocessor(
    *,
    official_root: Path,
    checkpoint: Path,
    processor_path: Path,
    stats_path: Path,
) -> tuple[Any, Any, Any]:
    sys.path.insert(0, str(official_root))
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.models import build_processor
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )

    training_config_path = checkpoint.parents[2] / "lingbotvla_cli.yaml"
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    model_config = dict(training_config["model"])
    model_config.update(training_config["train"])
    config = LingbotVLAV2Config(**model_config)
    for key, value in model_config.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    config.tokenizer_path = str(processor_path)

    processor = build_processor(str(processor_path))
    data_config = SimpleNamespace(**training_config["data"])
    transform = FeatureTransform(
        str(official_root / "configs/robot_configs/robotwin.yaml"),
        data_config,
        config,
        processor,
        chunk_size=config.chunk_size,
        norm_stats_path=str(stats_path),
    )
    return transform, data_config, config


def official_inputs(
    observation: dict[str, Any],
    *,
    transform: Any,
    data_config: Any,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prepared = dict(observation)
    resize = Resize((int(data_config.img_size), int(data_config.img_size)))
    resized = []
    for key in ROBOTWIN_CAMERA_KEYS:
        image = torch.as_tensor(prepared[key]).permute(2, 0, 1).contiguous()
        image = resize(image.to(dtype=torch.float32))
        prepared[key] = image
        resized.append(image)
    for key, value in list(prepared.items()):
        if isinstance(value, np.ndarray):
            prepared[key] = torch.from_numpy(value)
    transformed = transform.apply(prepared, policy_eval=True)
    return transformed, torch.stack(resized)


def phyai_inputs(
    observation: dict[str, Any],
    *,
    checkpoint: Path,
    processor_path: Path,
    stats_path: Path,
) -> tuple[Any, LingBotV2Processor, RoboTwinLingBotV2Adapter, torch.Tensor]:
    config = load_config(checkpoint, LingBotVLA2Config)
    stats = canonical_robotwin_stats(load_robotwin_stats(stats_path))
    adapter = RoboTwinLingBotV2Adapter(use_length=50)
    processor = LingBotV2Processor(
        processor_name=str(processor_path),
        num_images=3,
        num_channels=config.vision.in_channels,
        patch_vector_dim=config.vision.patch_vector_dim,
        max_patches_per_image=256,
        tokenizer_max_length=config.tokenizer_max_length,
        max_state_dim=config.max_state_dim,
        action_dim=14,
        dataset_stats=stats,
        normalization_eps=1e-6,
        device="cpu",
        params_dtype=torch.bfloat16,
    )
    prepared = adapter.prepare_observation(observation)
    resize = Resize((256, 256))
    resized = []
    for image in prepared["images"]:
        tensor = torch.as_tensor(image).permute(2, 0, 1).contiguous()
        resized.append(resize(tensor.to(dtype=torch.float32)))
    prepared["images"] = resized
    prepared["state"] = torch.as_tensor(prepared["state"], dtype=torch.float32)
    processed = processor.preprocess(prepared)
    processed.state = scatter_robotwin_state_to_model_slots(
        processed.state[..., :14],
        max_state_dim=config.max_state_dim,
    )
    return processed, processor, adapter, torch.stack(resized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observation = load_captured_request(args.request)
    transform, data_config, _ = load_official_preprocessor(
        official_root=args.official_root,
        checkpoint=args.checkpoint,
        processor_path=args.processor,
        stats_path=args.stats_json,
    )
    official, official_resized = official_inputs(
        observation,
        transform=transform,
        data_config=data_config,
    )
    phyai, phyai_processor, phyai_adapter, phyai_resized = phyai_inputs(
        observation,
        checkpoint=args.checkpoint,
        processor_path=args.processor,
        stats_path=args.stats_json,
    )

    official_images = official["images"].to(torch.bfloat16)
    official_state = official["state"].to(torch.bfloat16)
    official_tokens = official["lang_tokens"]
    official_language_mask = official["lang_masks"]
    phyai_language_mask = (
        torch.arange(phyai.input_ids.shape[-1]).unsqueeze(0)
        < phyai.lang_lens.unsqueeze(-1).cpu()
    )

    print("===== Model-ready preprocessing =====")
    report = {
        "resized_images": compare_array(
            "resized_images", official_resized, phyai_resized
        ),
        "pixel_values": compare_array(
            "pixel_values", official_images, phyai.pixel_values[0]
        ),
        "image_grid_thw": compare_array(
            "image_grid_thw", official["image_grid_thw"], phyai.image_grid_thw[0]
        ),
        "image_masks": compare_array(
            "image_masks", official["img_masks"], phyai.image_masks[0]
        ),
        "input_ids": compare_array("input_ids", official_tokens, phyai.input_ids[0]),
        "language_mask": compare_array(
            "language_mask", official_language_mask, phyai_language_mask[0]
        ),
        "normalized_state": compare_array(
            "normalized_state", official_state, phyai.state[0]
        ),
    }

    model_action = torch.linspace(-1.0, 1.0, 50 * 55).reshape(50, 55)
    official_postprocess_input = dict(official)
    official_postprocess_input["actions"] = model_action
    official_postprocess_input["state"] = official["state"].to(torch.float32)
    official_action = transform.unapply(official_postprocess_input)["action"]
    phyai_canonical_action = phyai_processor.postprocess(
        gather_robotwin_action_from_model_slots(model_action.unsqueeze(0))
    )
    phyai_action = phyai_adapter.format_action_chunk(phyai_canonical_action)["action"]
    report["postprocessed_action"] = compare_array(
        "postprocessed_action", official_action, phyai_action
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
