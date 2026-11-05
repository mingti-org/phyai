"""Run HumanWAM-Goal from RGB observations and normalized model-coordinate actions.

The JSON sample contains instruction, images, goal_images, observation_timestamp,
history, normalized_state and domain_id. Image paths are relative to the sample.
An optional normalized_prefix supplies RTC conditioning; outputs stay normalized.
"""

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from phyai.engine_config import (
    EngineConfig,
    RuntimeConfig,
    ParallelConfig,
    DenseParallelConfig,
    AttentionParallelConfig,
)
from phyai.models.humanwam_goal import (
    HumanWAMGoalTextRequest,
    HumanWAMGoalObservationRequest,
)
from phyai_utils_tools.models.humanwam_goal import (
    HumanWAMGoalProcessor,
    HumanWAMGoalProcessorConfig,
)
from phyai.models.humanwam_goal.main_humanwam_goal import HumanWAMGoalArgs

from phyai import Engine, EngineArgs


def read_sample(path: Path) -> dict:
    sample = json.loads(path.read_text())

    def images(views):
        result = {}
        for camera, filename in views.items():
            with Image.open(path.parent / filename) as source:
                result[camera] = np.array(source.convert("RGB"), copy=True)
        return result

    sample["images"] = images(sample["images"])
    sample["goal_images"] = images(sample["goal_images"])
    sample["history"] = [
        {**capture, "images": images(capture["images"])}
        for capture in sample.get("history", [])
    ]
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()
    config_path = args.checkpoint / "processor.json"
    processor_config = HumanWAMGoalProcessorConfig(
        **json.loads(config_path.read_text())
    )
    processor = HumanWAMGoalProcessor(
        str(args.checkpoint / "text_tokenizer"), processor_config
    )
    sample = read_sample(args.sample)
    engine = Engine(
        EngineArgs(
            plugin="humanwam_goal",
            plugin_args=HumanWAMGoalArgs(checkpoint_dir=args.checkpoint),
            config=EngineConfig(
                parallel=ParallelConfig(
                    dense=DenseParallelConfig(tp_size=args.tp),
                    attention=AttentionParallelConfig(tp_size=args.tp),
                ),
                runtime=RuntimeConfig(use_cuda_graph=not args.no_cuda_graph),
            ),
        )
    )
    result = None
    try:
        tokens = processor.tokenize(sample["instruction"])
        engine.step(HumanWAMGoalTextRequest(tokens.text_ids, tokens.text_mask))
        result = engine.step(
            HumanWAMGoalObservationRequest(**processor.preprocess(sample))
        )
        actions = processor.postprocess(result)["normalized_actions"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, actions.numpy())
    finally:
        # Drop borrowed CUDA IPC tensors before managed workers exit.
        result = None
        engine.close()


if __name__ == "__main__":
    main()
