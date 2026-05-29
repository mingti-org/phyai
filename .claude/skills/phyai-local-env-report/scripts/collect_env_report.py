#!/usr/bin/env python3
"""Collect a local phyai environment report.

The script is intentionally read-only. It gathers environment facts and records
command failures inline so one missing tool does not hide the rest of the report.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_MARKERS = ("pyproject.toml", "uv.lock")
WORKSPACE_PACKAGES = (
    "phyai",
    "phyai-kernel",
    "phyai-ext",
    "phyai-model-optimizer",
    "phyai-utils-tools",
)
PACKAGE_IMPORTS = {
    "phyai": "phyai",
    "phyai-kernel": "phyai_kernel",
    "phyai-ext": "phyai_ext",
    "phyai-model-optimizer": "phyai_model_optimizer",
    "phyai-utils-tools": "phyai_utils_tools",
}
INTERESTING_PACKAGES = (
    "phyai",
    "phyai-kernel",
    "phyai-ext",
    "phyai-model-optimizer",
    "phyai-utils-tools",
    "torch",
    "triton",
    "flashinfer-python",
    "flashinfer",
    "transformers",
    "numpy",
    "safetensors",
    "apache-tvm-ffi",
    "tvm-ffi",
    "torch-c-dlpack-ext",
    "torchvision",
    "scikit-build-core",
)
TOOL_NAMES = (
    "uv",
    "python",
    "python3",
    "git",
    "nvidia-smi",
    "nvcc",
    "cmake",
    "ninja",
    "gcc",
    "g++",
    "clang",
    "clang++",
)
CUDA_ENV_KEYS = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "TORCH_CUDA_ARCH_LIST",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
    "CUBLAS_WORKSPACE_CONFIG",
)


@dataclass
class CommandResult:
    command: list[str]
    found: bool
    timed_out: bool
    returncode: int | None
    stdout: str
    stderr: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "found": self.found,
            "timed_out": self.timed_out,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 10.0,
) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(command, False, False, None, "", "command not found")
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return CommandResult(command, True, True, None, stdout, stderr)
    return CommandResult(
        command,
        True,
        False,
        completed.returncode,
        completed.stdout.rstrip(),
        completed.stderr.rstrip(),
    )


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if all((candidate / marker).exists() for marker in REPO_MARKERS):
            return candidate
    return current


def load_pyproject(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        return {"_error": str(exc)}


def metadata_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def import_status(module_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "module": module_name,
        "ok": False,
        "version": None,
        "file": None,
        "error": None,
    }
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["ok"] = True
    result["version"] = getattr(module, "__version__", None)
    result["file"] = getattr(module, "__file__", None)
    return result


def command_summary(result: CommandResult, *, max_chars: int = 4000) -> dict[str, Any]:
    stdout = result.stdout
    stderr = result.stderr
    if len(stdout) > max_chars:
        stdout = stdout[:max_chars] + "\n... truncated ..."
    if len(stderr) > max_chars:
        stderr = stderr[:max_chars] + "\n... truncated ..."
    return {
        "command": " ".join(result.command),
        "found": result.found,
        "timed_out": result.timed_out,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def parse_nvcc_version(output: str) -> str | None:
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)*)", output)
    if match:
        return match.group(1)
    return None


def parse_nvidia_smi_driver(output: str) -> str | None:
    match = re.search(r"Driver Version:\s*([0-9.]+)", output)
    if match:
        return match.group(1)
    return None


def normalize_major(version: str | None) -> str | None:
    if not version:
        return None
    return version.split(".", 1)[0]


def collect_git(repo_root: Path) -> dict[str, Any]:
    commands = {
        "inside_work_tree": ["git", "rev-parse", "--is-inside-work-tree"],
        "branch": ["git", "branch", "--show-current"],
        "commit": ["git", "rev-parse", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }
    out: dict[str, Any] = {}
    for key, command in commands.items():
        result = run_command(command, cwd=repo_root, timeout=5)
        out[key] = command_summary(result, max_chars=12000)
    status = out["status_short"]
    dirty = (
        bool(status.get("stdout", "").strip())
        if status.get("returncode") == 0
        else None
    )
    out["dirty"] = dirty
    return out


def collect_workspace(repo_root: Path) -> dict[str, Any]:
    root_pyproject = load_pyproject(repo_root / "pyproject.toml")
    members = (
        root_pyproject.get("tool", {})
        .get("uv", {})
        .get("workspace", {})
        .get("members", [])
    )

    packages: dict[str, Any] = {}
    for package in WORKSPACE_PACKAGES:
        package_dir = repo_root / package
        pyproject = load_pyproject(package_dir / "pyproject.toml")
        project = pyproject.get("project", {}) if isinstance(pyproject, dict) else {}
        import_name = PACKAGE_IMPORTS[package]
        packages[package] = {
            "path": str(package_dir),
            "pyproject_version": project.get("version"),
            "metadata_version": metadata_version(package),
            "import": import_status(import_name),
        }

    return {
        "root_project": root_pyproject.get("project", {}),
        "workspace_members": members,
        "packages": packages,
    }


def collect_python_packages() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in INTERESTING_PACKAGES:
        packages[name] = metadata_version(name)
    return packages


def collect_torch() -> dict[str, Any]:
    status = import_status("torch")
    if not status["ok"]:
        return {"import": status}

    import torch

    info: dict[str, Any] = {
        "import": status,
        "version": torch.__version__,
        "built_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": None,
        "cuda_device_count": None,
        "current_device": None,
        "cudnn_available": None,
        "cudnn_version": None,
        "nccl_available": None,
        "nccl_version": None,
        "devices": [],
        "error": None,
    }
    try:
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
        if info["cuda_available"]:
            info["current_device"] = torch.cuda.current_device()
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                major, minor = torch.cuda.get_device_capability(idx)
                info["devices"].append(
                    {
                        "index": idx,
                        "name": props.name,
                        "capability": f"sm_{major}{minor}",
                        "multi_processor_count": props.multi_processor_count,
                        "total_memory_gib": round(props.total_memory / (1 << 30), 2),
                        "uuid": str(getattr(props, "uuid", "")),
                    }
                )
        info["cudnn_available"] = torch.backends.cudnn.is_available()
        info["cudnn_version"] = torch.backends.cudnn.version()
        if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "is_available"):
            info["nccl_available"] = torch.cuda.nccl.is_available([])
        if hasattr(torch.cuda, "nccl") and hasattr(torch.cuda.nccl, "version"):
            try:
                info["nccl_version"] = torch.cuda.nccl.version()
            except Exception as exc:  # noqa: BLE001
                info["nccl_version"] = f"error: {exc}"
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def collect_cuda(repo_root: Path, *, gpu_detail: bool) -> dict[str, Any]:
    nvcc = run_command(["nvcc", "--version"], cwd=repo_root, timeout=8)
    smi_query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        cwd=repo_root,
        timeout=10,
    )
    smi_full = run_command(["nvidia-smi"], cwd=repo_root, timeout=10)
    topo = (
        run_command(["nvidia-smi", "topo", "-m"], cwd=repo_root, timeout=10)
        if gpu_detail
        else CommandResult(
            ["nvidia-smi", "topo", "-m"], False, False, None, "", "skipped"
        )
    )

    return {
        "env": {key: os.environ.get(key) for key in CUDA_ENV_KEYS if key in os.environ},
        "tool_paths": {
            "nvcc": shutil.which("nvcc"),
            "nvidia-smi": shutil.which("nvidia-smi"),
        },
        "nvcc": command_summary(nvcc),
        "nvcc_version": parse_nvcc_version(nvcc.stdout),
        "nvidia_smi_query": command_summary(smi_query),
        "nvidia_smi": command_summary(smi_full),
        "driver_version": parse_nvidia_smi_driver(smi_full.stdout),
        "topology": command_summary(topo, max_chars=16000),
    }


def collect_phyai_env() -> dict[str, Any]:
    process_phyai = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("PHYAI_")
    }
    registered: dict[str, Any] = {}
    status = import_status("phyai.env")
    if not status["ok"]:
        return {
            "registered": registered,
            "process": process_phyai,
            "registry_import": status,
        }

    try:
        module = importlib.import_module("phyai.env")
        envs = getattr(module, "envs")
        env_field = getattr(module, "EnvField")
        for name, value in inspect.getmembers(envs):
            if name.startswith("PHYAI_") and isinstance(value, env_field):
                raw = os.environ.get(value.name)
                parsed: Any = None
                error: str | None = None
                if value.is_set():
                    try:
                        parsed = value.get()
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                else:
                    parsed = value.default
                registered[value.name] = {
                    "set": value.is_set(),
                    "raw": raw,
                    "parsed_or_default": str(parsed) if parsed is not None else None,
                    "error": error,
                }
    except Exception as exc:  # noqa: BLE001
        status["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "registered": registered,
        "process": process_phyai,
        "registry_import": status,
    }


def collect_tools() -> dict[str, str | None]:
    return {name: shutil.which(name) for name in TOOL_NAMES}


def collect_commands(repo_root: Path) -> dict[str, Any]:
    command_specs = {
        "uv_version": ["uv", "--version"],
        "python_version": [sys.executable, "--version"],
        "phyai_kernel_show_env": ["uv", "run", "phyai-kernel", "show-env"],
    }
    results: dict[str, Any] = {}
    for key, command in command_specs.items():
        results[key] = command_summary(
            run_command(command, cwd=repo_root, timeout=20), max_chars=16000
        )
    if not results["phyai_kernel_show_env"]["found"] or results[
        "phyai_kernel_show_env"
    ]["returncode"] not in (0, None):
        fallback = run_command(
            [sys.executable, "-m", "phyai_kernel", "show-env"],
            cwd=repo_root,
            timeout=20,
        )
        results["phyai_kernel_show_env_fallback"] = command_summary(
            fallback, max_chars=16000
        )
    return results


def collect_report(repo_root: Path, *, gpu_detail: bool) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data: dict[str, Any] = {
        "generated_at": generated_at,
        "repo_root": str(repo_root),
        "host": {
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
            "cwd": str(Path.cwd()),
        },
        "system": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version,
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
        },
        "tools": collect_tools(),
        "git": collect_git(repo_root),
        "workspace": collect_workspace(repo_root),
        "python_packages": collect_python_packages(),
        "torch": collect_torch(),
        "cuda": collect_cuda(repo_root, gpu_detail=gpu_detail),
        "phyai_env": collect_phyai_env(),
        "commands": collect_commands(repo_root),
    }
    data["diagnostics"] = build_diagnostics(data)
    return data


def build_diagnostics(data: dict[str, Any]) -> list[str]:
    diagnostics: list[str] = []
    torch_info = data.get("torch", {})
    torch_import = torch_info.get("import", {})
    if not torch_import.get("ok"):
        diagnostics.append(f"torch import failed: {torch_import.get('error')}")
    elif not torch_info.get("cuda_available"):
        diagnostics.append("torch.cuda.is_available() is false")

    cuda = data.get("cuda", {})
    if not cuda.get("tool_paths", {}).get("nvidia-smi"):
        diagnostics.append("nvidia-smi not found on PATH")
    if not cuda.get("tool_paths", {}).get("nvcc"):
        diagnostics.append("nvcc not found on PATH")

    torch_cuda = torch_info.get("built_cuda")
    nvcc_version = cuda.get("nvcc_version")
    if normalize_major(torch_cuda) and normalize_major(nvcc_version):
        if normalize_major(torch_cuda) != normalize_major(nvcc_version):
            diagnostics.append(
                f"Torch was built with CUDA {torch_cuda}, but nvcc reports CUDA {nvcc_version}"
            )

    for package, package_info in data.get("workspace", {}).get("packages", {}).items():
        import_info = package_info.get("import", {})
        if not import_info.get("ok"):
            diagnostics.append(f"{package} import failed: {import_info.get('error')}")

    registered_env = data.get("phyai_env", {}).get("registered", {})
    for name, item in registered_env.items():
        if item.get("error"):
            diagnostics.append(f"{name} parse failed: {item['error']}")

    if data.get("git", {}).get("dirty") is True:
        diagnostics.append("git worktree has uncommitted changes")

    return diagnostics


def md_escape(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def section(title: str) -> str:
    return f"\n## {title}\n"


def fenced(text: str, language: str = "text") -> str:
    if not text:
        return "_empty_\n"
    return f"```{language}\n{text.rstrip()}\n```\n"


def render_kv(items: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in items.items():
        lines.append(f"| {md_escape(key)} | {md_escape(value)} |")
    return "\n".join(lines) + "\n"


def render_command(command: dict[str, Any]) -> str:
    parts = [
        f"- command: `{command.get('command')}`",
        f"- found: `{command.get('found')}`",
        f"- timed_out: `{command.get('timed_out')}`",
        f"- returncode: `{command.get('returncode')}`",
    ]
    if command.get("stdout"):
        parts.append("\nstdout:\n" + fenced(command["stdout"]))
    if command.get("stderr"):
        parts.append("\nstderr:\n" + fenced(command["stderr"]))
    return "\n".join(parts) + "\n"


def render_markdown(data: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# phyai Local Environment Report\n")
    lines.append(
        render_kv(
            {
                "Generated at": data["generated_at"],
                "Repository": data["repo_root"],
                "Hostname": data["host"]["hostname"],
                "User": data["host"]["user"],
                "CWD": data["host"]["cwd"],
            }
        )
    )

    diagnostics = data.get("diagnostics", [])
    lines.append(section("Diagnostics"))
    if diagnostics:
        lines.extend(f"- {item}\n" for item in diagnostics)
    else:
        lines.append("- No obvious issues detected.\n")

    lines.append(section("System"))
    lines.append(render_kv(data["system"]))

    lines.append(section("Tools"))
    lines.append(render_kv(data["tools"]))

    lines.append(section("Git"))
    git = data["git"]
    git_table = {
        "Branch": git.get("branch", {}).get("stdout"),
        "Commit": git.get("commit", {}).get("stdout"),
        "Dirty": git.get("dirty"),
    }
    lines.append(render_kv(git_table))
    status = git.get("status_short", {}).get("stdout", "")
    if status:
        lines.append("git status --short:\n")
        lines.append(fenced(status))

    lines.append(section("Workspace Packages"))
    lines.append(
        "| Package | pyproject | installed metadata | import | module version | module file / error |\n"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for name, item in data["workspace"]["packages"].items():
        import_info = item["import"]
        detail = (
            import_info.get("file")
            if import_info.get("ok")
            else import_info.get("error")
        )
        lines.append(
            f"| {md_escape(name)} | {md_escape(item.get('pyproject_version'))} | "
            f"{md_escape(item.get('metadata_version'))} | {md_escape(import_info.get('ok'))} | "
            f"{md_escape(import_info.get('version'))} | {md_escape(detail)} |\n"
        )

    lines.append(section("Python Packages"))
    lines.append(render_kv(data["python_packages"]))

    lines.append(section("CUDA / GPU"))
    cuda = data["cuda"]
    torch_info = data["torch"]
    torch_summary = {
        "torch version": torch_info.get("version"),
        "torch built CUDA": torch_info.get("built_cuda"),
        "torch CUDA available": torch_info.get("cuda_available"),
        "torch CUDA device count": torch_info.get("cuda_device_count"),
        "torch current device": torch_info.get("current_device"),
        "cuDNN available": torch_info.get("cudnn_available"),
        "cuDNN version": torch_info.get("cudnn_version"),
        "NCCL available": torch_info.get("nccl_available"),
        "NCCL version": torch_info.get("nccl_version"),
        "nvcc version": cuda.get("nvcc_version"),
        "driver version": cuda.get("driver_version"),
    }
    lines.append(render_kv(torch_summary))

    devices = torch_info.get("devices") or []
    if devices:
        lines.append("\nTorch CUDA devices:\n")
        lines.append("| Index | Name | Capability | SMs | Memory GiB | UUID |\n")
        lines.append("| --- | --- | --- | --- | --- | --- |\n")
        for dev in devices:
            lines.append(
                f"| {md_escape(dev.get('index'))} | {md_escape(dev.get('name'))} | "
                f"{md_escape(dev.get('capability'))} | {md_escape(dev.get('multi_processor_count'))} | "
                f"{md_escape(dev.get('total_memory_gib'))} | {md_escape(dev.get('uuid'))} |\n"
            )

    if cuda.get("env"):
        lines.append("\nCUDA-related environment:\n")
        lines.append(render_kv(cuda["env"]))

    for label, key in (
        ("nvcc --version", "nvcc"),
        ("nvidia-smi query", "nvidia_smi_query"),
        ("nvidia-smi topo -m", "topology"),
    ):
        lines.append(f"\n### {label}\n")
        lines.append(render_command(cuda[key]))

    lines.append(section("phyai Runtime Environment"))
    env_info = data["phyai_env"]
    lines.append("Registered `PHYAI_*` variables:\n")
    lines.append("| Name | Set | Raw | Parsed/default | Error |\n")
    lines.append("| --- | --- | --- | --- | --- |\n")
    for name, item in sorted(env_info["registered"].items()):
        lines.append(
            f"| {md_escape(name)} | {md_escape(item.get('set'))} | "
            f"{md_escape(item.get('raw'))} | {md_escape(item.get('parsed_or_default'))} | "
            f"{md_escape(item.get('error'))} |\n"
        )
    extra = {
        key: value
        for key, value in env_info["process"].items()
        if key not in env_info["registered"]
    }
    if extra:
        lines.append("\nExtra process `PHYAI_*` variables:\n")
        lines.append(render_kv(extra))

    lines.append(section("phyai-kernel show-env"))
    show_env = data.get("commands", {}).get("phyai_kernel_show_env", {})
    lines.append(render_command(show_env))
    fallback = data.get("commands", {}).get("phyai_kernel_show_env_fallback")
    if fallback:
        lines.append("\nFallback:\n")
        lines.append(render_command(fallback))

    return "".join(lines)


def write_output(content: str, output: Path | None) -> None:
    if output is None:
        print(content)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    print(str(output))


def default_output_path(repo_root: Path, output_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return repo_root / output_dir / f"local-env-{stamp}.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="phyai repository root. Defaults to auto-detection from cwd.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this path instead of stdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory used with --default-output-name.",
    )
    parser.add_argument(
        "--default-output-name",
        action="store_true",
        help="Write to reports/local-env-YYYYMMDD-HHMMSS.md unless --output is set.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Report format.",
    )
    parser.add_argument(
        "--no-gpu-detail",
        action="store_true",
        help="Skip slower GPU detail commands such as nvidia-smi topo -m.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = (
        args.repo_root.resolve() if args.repo_root else find_repo_root(Path.cwd())
    )
    report = collect_report(repo_root, gpu_detail=not args.no_gpu_detail)

    if args.format == "json":
        content = json.dumps(report, indent=2, sort_keys=True) + "\n"
        output = args.output
        if output is None and args.default_output_name:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            output = repo_root / args.output_dir / f"local-env-{stamp}.json"
    else:
        content = render_markdown(report)
        output = args.output
        if output is None and args.default_output_name:
            output = default_output_path(repo_root, args.output_dir)

    write_output(content, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
