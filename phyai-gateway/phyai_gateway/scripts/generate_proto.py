"""Regenerate bindings using the gateway's codegen extra.

Then run `ruff format phyai-gateway/phyai_gateway/bindings` from the workspace root.
"""

from pathlib import Path

from grpc_tools import protoc


if __name__ == "__main__":
    package_root = Path(__file__).resolve().parents[2]
    proto_dir = package_root / "phyai_gateway" / "proto"
    raise SystemExit(
        protoc.main(
            [
                "grpc_tools.protoc",
                f"-Iphyai_gateway/bindings={proto_dir}",
                f"--python_out={package_root}",
                f"--grpc_python_out={package_root}",
                str(proto_dir / "model_inference.proto"),
                str(proto_dir / "robot.proto"),
            ]
        )
    )
