"""Run the RoboTwin evaluator against the LingBot direct websocket protocol.

This is an isolation harness: it keeps the current RoboTwin evaluator and
observation/action conversion, but bypasses the XPolicyLab bridge process.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=9330)
    parser.add_argument("--task-name", default="hanging_mug")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument(
        "--instruction-type", choices=("seen", "unseen"), default="unseen"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


class DirectModelClient:
    _robotwin_protocol = "legacy_direct"

    def __init__(self, *, host: str, port: int, capture_path: Path) -> None:
        from benchmark.lingbot_v2.robotwin.legacy_policy_bridge import (
            DirectWebSocketPolicyClient,
            LingBotV2LegacyBridge,
        )

        self.bridge = LingBotV2LegacyBridge(
            DirectWebSocketPolicyClient(host=host, port=port),
            capture_first_request=capture_path,
        )

    def call(self, *, func_name: str, **kwargs: Any) -> Any:
        if func_name == "reset":
            return self.bridge.reset()
        if func_name == "update_obs":
            return self.bridge.update_obs(kwargs["obs"])
        if func_name == "get_action":
            return self.bridge.get_action()
        raise NotImplementedError(func_name)

    def close(self) -> None:
        self.bridge.client.close()


def main() -> None:
    args = parse_args()
    robotwin_root = Path(args.robotwin_root).resolve()
    scripts_root = robotwin_root / "scripts"
    sys.path.insert(0, str(scripts_root))
    sys.path.insert(0, str(robotwin_root))

    evaluator = importlib.import_module("eval_policy_xpolicylab")
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Keep RoboTwin as cwd: its evaluator resolves assets/configs relatively.
    os.chdir(robotwin_root)

    capture_path = output_root / "first-direct-request.npz"
    client = DirectModelClient(
        host=args.backend_host,
        port=args.backend_port,
        capture_path=capture_path,
    )

    original_builder = evaluator.build_policy_client
    evaluator.build_policy_client = lambda _usr_args: client
    try:
        evaluator.main(
            {
                "bench_name": "RoboTwin",
                "task_name": args.task_name,
                "policy_name": "OfficialDirect",
                "host": args.backend_host,
                "port": str(args.backend_port),
                "protocol": "legacy_direct",
                "root_dir": str(robotwin_root),
                "seed": args.seed,
                "task_config": args.task_config,
                "instruction_type": args.instruction_type,
                "expert_check": True,
                "frequency": 30,
                "eval_batch": False,
                "ckpt_setting": "direct-websocket",
                "xpolicylab_root": str(robotwin_root / "XPolicyLab"),
                "test_num": args.test_num,
                "action_type": "joint",
            }
        )
    finally:
        evaluator.build_policy_client = original_builder
        client.close()


if __name__ == "__main__":
    main()
