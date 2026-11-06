"""Compile the fixed CUDA IPC gather operators and expose its small C ABI."""

import os
import fcntl
import ctypes as C
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path

import torch


def humanwam_goal_gather_library():
    """Compile the small C ABI once; never modify the serving environment."""
    source = Path(__file__).parent.parent / "csrc" / "humanwam_goal_gather"
    nvcc = shutil.which("nvcc") or str(
        Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"
    )
    version = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True
    ).stdout
    capability = torch.cuda.get_device_capability()
    architecture = f"sm_{capability[0]}{capability[1]}"
    digest = hashlib.sha256(version + architecture.encode())
    for path in sorted(source.rglob("*.cu*")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    cache = Path(
        os.environ.get("TORCH_EXTENSIONS_DIR", Path.home() / ".cache" / "phyai")
    )
    cache = cache / "tp_gather" / digest.hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "tp_gather.so"
    with (cache / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = cache / f"build_{os.getpid()}.so"
            cuda = Path(nvcc).resolve().parent.parent
            command = [
                nvcc,
                "-std=c++17",
                "-O3",
                "-shared",
                "-Xcompiler",
                "-fPIC",
                f"-arch={architecture}",
                f"-L{cuda / 'lib64' / 'stubs'}",
                "-lcuda",
                str(source / "tp_gather.cu"),
                "-o",
                str(temporary),
            ]
            # Keep compiler scratch in our writable, content-addressed cache.
            with tempfile.TemporaryDirectory(dir=cache, prefix="nvcc_") as scratch:
                environment = dict(os.environ, TMPDIR=scratch)
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
            if result.returncode:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"TP collective compilation failed:\n{result.stderr[-6000:]}"
                )
            temporary.replace(target)
    library = C.CDLL(str(target))
    library._humanwam_goal_build_hash = digest.hexdigest()
    library._humanwam_goal_binary_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    pointer, integer, size = C.c_void_p, C.c_int, C.c_size_t
    declarations = {
        "error_message": ([], C.c_char_p),
        "signal_size": ([], size),
        "alloc_shared": ([size, C.POINTER(pointer), pointer], integer),
        "open_shared": ([pointer, C.POINTER(pointer)], integer),
        "export_shared": ([pointer, pointer, C.POINTER(size)], integer),
        "close_shared": ([pointer], integer),
        "free_shared": ([pointer], integer),
        "create_gather": (
            [pointer, pointer, size, integer, integer, C.POINTER(pointer)],
            integer,
        ),
        "gather_last_dimension": (
            [pointer, pointer, pointer, integer, integer, pointer, integer],
            integer,
        ),
        "gather_sp_kv": (
            [
                pointer,
                pointer,
                pointer,
                pointer,
                pointer,
                integer,
                integer,
                integer,
                pointer,
                integer,
            ],
            integer,
        ),
        "destroy_gather": ([pointer], integer),
    }
    for name, (arguments, returns) in declarations.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, returns
    return library
