"""Explicit debug entry points: run enables native control; demo/view never do."""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "phyai-robot/src"))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check"):
        child = sub.add_parser(
            name,
            help="Enable native robot control and record"
            if name == "run"
            else "Validate config/probe compatibility without ROS or RPC",
        )
        child.add_argument("--client-config", type=Path, required=True)
        child.add_argument("--trace-config", type=Path)
        child.add_argument("--output", type=Path)
        child.add_argument("overrides", nargs="*")
    child = sub.add_parser(
        "demo", help="Create synthetic data without ROS or model access"
    )
    child.add_argument("--output", type=Path, required=True)
    child.add_argument("--robot-assets", type=Path)
    child = sub.add_parser("view", help="Serve a saved trace, read-only")
    child.add_argument("trace", type=Path)
    child.add_argument("--host", default="127.0.0.1")
    child.add_argument("--port", type=int, default=8770)
    return parser.parse_args()


def configuration(args):
    import yaml
    from examples.deployment.configuration import parse_config
    from examples.deployment.client import validate_client
    from examples.deployment.tianji import TianjiClientConfig, DEFAULT_CONFIG

    options = {
        "output_root": "./traces",
        "queue_capacity": 4096,
        "image_capacity": 64,
        "flush_timeout_s": 15,
        "robot_assets": None,
        "model_bundle": None,
        "capture_video": True,
        "passive_ros": True,
    }
    if args.trace_config:
        supplied = yaml.safe_load(args.trace_config.read_text()) or {}
        if not isinstance(supplied, dict) or set(supplied) - set(options):
            raise ValueError("Unknown trace config fields")
        options.update(supplied)
    for key in ("output_root", "robot_assets", "model_bundle"):
        if options[key]:
            base = (
                args.trace_config.resolve().parent if args.trace_config else Path.cwd()
            )
            options[key] = str((base / Path(options[key]).expanduser()).resolve())
    for key in ("queue_capacity", "image_capacity"):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("capture_video", "passive_ros"):
        if type(options[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not 0 < float(options["flush_timeout_s"]) <= 120:
        raise ValueError("flush_timeout_s must be in (0,120]")
    config = parse_config(
        TianjiClientConfig,
        default_config=DEFAULT_CONFIG,
        argv=["--config", str(args.client_config), *args.overrides],
        validate=validate_client,
    )
    return config, options


def run(args):
    from phyai_robot.remote import run_remote, Embodiment
    from examples.deployment.robots.tianji import build
    from .storage import TraceRecorder
    from .capture import (
        Session,
        TracedRobot,
        TracedAdapter,
        TracedPolicy,
        TracedScheduler,
    )
    from .compat import Probes, verify_contract
    from .assets import snapshot_robot
    from .ros_monitor import RosMonitor

    config, options = configuration(args)
    hashes = verify_contract()
    if args.command == "check":
        print(
            json.dumps(
                {
                    "status": "configuration_and_probe_signatures_valid",
                    "client": asdict(config),
                    "trace": options,
                    "probe_source_hashes": hashes,
                    "robot_started": False,
                },
                indent=2,
            )
        )
        return
    output = args.output or Path(options["output_root"]) / datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    recorder = TraceRecorder(
        output,
        {
            "client": asdict(config),
            "trace": options,
            "code_commit": revision,
            "probe_source_hashes": hashes,
        },
    )
    trace = Session(recorder, config.control.max_sample_age_s)
    monitor = None
    try:
        if options["robot_assets"]:
            snapshot_robot(options["robot_assets"], recorder.output)
        if options["model_bundle"]:
            source = Path(options["model_bundle"])
            target = recorder.output / "model_metadata"
            target.mkdir()
            for name in (
                "config.json",
                "server.yaml",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "provenance.json",
                "kernel_policy.yaml",
            ):
                p = source / name
                if (
                    p.is_file()
                    and not p.is_symlink()
                    and p.stat().st_size < 4 * 1024**2
                ):
                    (target / name).write_bytes(p.read_bytes())
        if options["passive_ros"]:
            monitor = RosMonitor(trace, config.robot.domain_id)
            monitor.start()
        with Probes(trace, raw_video=options["capture_video"]) as probes:
            policy = config.remote.create()
            probes.policy(policy)

            def traced_build(spec):
                native = build(config.robot, spec)
                for backend in native.robot.backends:
                    probes.backend(backend)
                return Embodiment(
                    TracedRobot(native.robot, trace),
                    TracedAdapter(native.adapter, trace),
                    native.activate,
                )

            def report(message):
                trace.emit("event", name="lifecycle", message=message)
                print(message, flush=True)

            print(
                f"Recording {recorder.output}; the native client will enable robot control after warmup.",
                flush=True,
            )
            run_remote(
                TracedPolicy(policy, trace),
                build=traced_build,
                scheduler=TracedScheduler(config.schedule.create(), trace),
                options=config.control.to_options(),
                max_steps=config.max_steps,
                report=report,
            )
    except BaseException as error:
        trace.emit("event", name="client_exit", error=repr(error))
        raise
    finally:
        if monitor:
            monitor.close()
        result = recorder.close(grace_s=options["flush_timeout_s"])
        print(f"Trace {result['status']}: {recorder.output}", flush=True)


def main():
    args = arguments()
    if args.command in ("run", "check"):
        run(args)
    elif args.command == "demo":
        from .demo import create_demo

        create_demo(args.output, robot_assets=args.robot_assets)
    else:
        from .viewer import serve_trace

        serve_trace(args.trace, host=args.host, port=args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130) from None
