"""Run VPP2 video prediction or a normalized-observation action request.

Use torchrun with --tp 2 for two-way tensor parallelism. Video CFG branches
can also run on separate ranks with --cfg-parallel 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

import phyai.parallel as parallel
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import (
    AttentionParallelConfig,
    DenseParallelConfig,
    DeviceConfig,
    EngineConfig,
    OuterParallelConfig,
    ParallelConfig,
    RuntimeConfig,
)
from phyai.kernel.config import KernelConfig
from phyai.models.vpp2 import VPP2Args, VPP2ObservationRequest
from phyai.server.deployment import DeploymentConfig
from phyai.utils import get_logger
from phyai_utils_tools.models.vpp2 import VPP2Processor


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", default="robodojo_joint2b_100k")
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        required=True,
        help="One image, RoboDojo head/left/right views, or two LIBERO views.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--proprio", type=float, nargs="+")
    parser.add_argument(
        "--condition-history",
        type=Path,
        help="Torch tensor [1,3,4n+1,H,W] in [-1,1], already composed and resized.",
    )
    parser.add_argument(
        "--condition-observation",
        type=Path,
        help="Independent current-frame tensor [1,3,1,H,W] in [-1,1].",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cfg-parallel", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--cuda-graph", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--kernel-config",
        type=Path,
        help="Kernel policy YAML; defaults to the action or video policy beside this script.",
    )
    parser.add_argument("--return-video", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/vpp2/weights"))
    parser.add_argument("--output", type=Path, default=Path("vpp2-output.pt"))
    args = parser.parse_args()
    processor = VPP2Processor.from_pretrained(args.checkpoint, variant=args.variant)
    images = []
    for path in args.image:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    raw = {"images": images, "task": args.task, "negative_prompt": args.negative_prompt}
    if args.proprio is not None:
        raw["proprio"] = args.proprio
    for path, key in (
        (args.condition_history, "condition_video"),
        (args.condition_observation, "condition_observation_video"),
    ):
        if path is not None:
            raw[key] = torch.load(path, map_location="cpu", weights_only=True)
    options = {}
    for value, name in (
        (args.steps, "num_inference_steps"),
        (args.seed, "seed"),
        (args.guidance, "text_cfg_scale"),
    ):
        if value is not None:
            options[name] = value
    if processor.mode == "action" and args.return_video:
        options["return_full_video"] = True
    raw["inference"] = options
    request = VPP2ObservationRequest(**processor.preprocess(raw))
    kernel_config = args.kernel_config or Path(__file__).with_name(
        f"kernels_{processor.mode}.yaml"
    )
    config = EngineConfig(
        device=DeviceConfig(params_dtype=torch.bfloat16),
        runtime=RuntimeConfig(use_cuda_graph=args.cuda_graph),
        kernel=KernelConfig(config_path=str(kernel_config)),
        parallel=ParallelConfig(
            outer=OuterParallelConfig(cfg_size=args.cfg_parallel),
            dense=DenseParallelConfig(tp_size=args.tp),
            attention=AttentionParallelConfig(tp_size=args.tp),
        ),
    )
    engine = Engine(
        EngineArgs(
            plugin="vpp2",
            config=config,
            plugin_args=VPP2Args(
                checkpoint_dir=args.checkpoint,
                variant=args.variant,
                cache_dir=args.cache_dir,
                torch_compile=args.compile,
                use_cuda_graph=args.cuda_graph,
            ),
        ),
        deployment=DeploymentConfig.external(),
    )
    try:
        output = engine.step(request)
        if parallel.default_mesh().group_rank("world") == 0:
            output = processor.postprocess(output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in output.items()
                },
                args.output,
            )
            logger.info_rank0("Saved VPP2 output to %s", args.output)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
