"""Top-level command-line interface for PhyAI."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level PhyAI command parser."""
    parser = argparse.ArgumentParser(
        prog="phyai",
        description="PhyAI runtime and deployment commands.",
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
    )
    doctor = commands.add_parser(
        "doctor",
        help="Inspect the local PhyAI runtime and accelerator availability.",
    )
    doctor.set_defaults(handler=_run_doctor)

    env = commands.add_parser(
        "env",
        help="Show registered PHYAI_* environment settings.",
    )
    env.add_argument("--json", action="store_true", help="Emit JSON output.")
    env.set_defaults(handler=_run_env)

    server = commands.add_parser(
        "server", help="Serve a model from a YAML configuration."
    )
    server.add_argument("config", help="Path to the server YAML file.")
    server.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration without starting the model.",
    )
    server.set_defaults(handler=_run_server)
    return parser


def _run_doctor(_args: argparse.Namespace) -> None:
    """Print a lightweight diagnostic report without loading model plugins."""
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device_count"] = int(torch.cuda.device_count())
        report["distributed_available"] = bool(torch.distributed.is_available())
    except Exception as error:  # pragma: no cover - defensive diagnostics path
        report["torch_error"] = f"{type(error).__name__}: {error}"
    for name, value in report.items():
        print(f"{name}: {value}")


def _run_env(args: argparse.Namespace) -> None:
    """Show registered PhyAI environment fields and their current values."""
    from phyai.env import envs

    values = {
        name: field.get()
        for name, field in vars(envs).items()
        if name.startswith("PHYAI_")
    }
    if args.json:
        print(json.dumps(values, sort_keys=True, default=str))
        return
    for name, value in values.items():
        print(f"{name}={value if value is not None else ''}")


def _run_server(args: argparse.Namespace) -> None:
    from phyai.server.config import load_server_config

    config = load_server_config(args.config)
    if args.check:
        print(f"Configuration valid: {args.config}")
        return
    from phyai.server.serving import serve

    serve(config)


def main(argv: Sequence[str] | None = None) -> None:
    """Run a PhyAI command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


__all__ = ["build_parser", "main"]
