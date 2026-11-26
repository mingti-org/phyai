"""Run LingBot V2 once with synthetic camera images and a zero state."""

from pathlib import Path

import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.kernel.config import KernelConfig
from phyai.models.lingbot_v2 import (
    LingBotV2Args,
    LingBotV2Request,
    LingBotVLA2Config,
)
from phyai.utils import load_config
from phyai_utils_tools.models.lingbot_v2 import LingBotV2Processor


CHECKPOINT_DIR = Path("/path/to/lingbot-vla-v2")
PROCESSOR_DIR = "/path/to/Qwen3-VL-4B-Instruct"


def main() -> None:
    config = load_config(CHECKPOINT_DIR, LingBotVLA2Config)
    processor = LingBotV2Processor(
        processor_name=PROCESSOR_DIR,
        num_channels=config.vision.in_channels,
        patch_vector_dim=config.vision.patch_vector_dim,
        tokenizer_max_length=config.tokenizer_max_length,
        max_state_dim=config.max_state_dim,
        action_dim=config.action_dim,
        device="cuda",
        params_dtype=torch.bfloat16,
    )
    processed = processor.preprocess(
        {
            "images": [
                torch.zeros(1, config.vision.in_channels, 224, 224, dtype=torch.uint8)
                for _ in range(3)
            ],
            "task": ["pick up the cup"],
            "state": torch.zeros(1, config.max_state_dim),
        }
    )
    request = LingBotV2Request(**vars(processed))
    max_vision_tokens = (
        processed.pixel_values.shape[2] // config.vision.spatial_merge_unit
    )
    engine = Engine(
        EngineArgs(
            plugin="lingbot_v2",
            plugin_args=LingBotV2Args(
                checkpoint_dir=CHECKPOINT_DIR,
                config=config,
                max_vision_tokens_per_image=max_vision_tokens,
            ),
            config=EngineConfig(
                kernel=KernelConfig(
                    config_path=str(Path(__file__).with_name("kernel_policy.yaml")),
                ),
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(allow_tf32=True, use_cuda_graph=True),
            ),
        )
    )
    try:
        with torch.inference_mode():
            actions = processor.postprocess(engine.step(request))
        print(f"Action shape: {tuple(actions.shape)}")
        print(f"First action: {actions[0, 0].tolist()}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
