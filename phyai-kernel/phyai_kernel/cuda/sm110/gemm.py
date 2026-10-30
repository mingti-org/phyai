"""BF16 GEMM implementations with per-layer, per-stream plans."""

from __future__ import annotations

import hashlib
import importlib
import json
from functools import cache
from pathlib import Path
import re
import weakref

import torch


@cache
def configurations() -> tuple[dict, ...]:
    return tuple(json.loads(Path(__file__).with_name("gemm_configs.json").read_text()))


def native_source_bytes(path: Path, seen: set[Path] | None = None) -> bytes:
    """Include local dependencies in the compiled source key."""
    seen = set() if seen is None else seen
    if path in seen:
        return b""
    seen.add(path)
    content = path.read_bytes()
    dependencies = [
        path.parent / name.decode()
        for name in re.findall(rb'^#include "([^"]+)"', content, re.MULTILINE)
    ]
    return content + b"".join(
        native_source_bytes(dependency, seen)
        for dependency in dependencies
        if dependency.is_file()
    )


@cache
def load_native(target: str, rows: int, device_index: int):
    from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS
    from phyai_kernel.jit_utils.compile import PHYAI_KERNEL_CUDA_CSRC_DIR, load_cuda_jit

    config = next(
        row for row in configurations() if row["target"] == target and row["m"] == rows
    )
    source = "sm110/" + config.get(
        "source", f"{target.removeprefix('pi05_').lower()}.cu"
    )
    digest = hashlib.sha256(
        native_source_bytes(PHYAI_KERNEL_CUDA_CSRC_DIR / source)
        + b"".join(
            path.read_bytes()
            for path in sorted(
                (
                    PHYAI_KERNEL_CUDA_CSRC_DIR
                    / "sm110/third_party/cutlass_compat/include"
                ).rglob("*.hpp")
            )
        )
        + json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:16]
    flags = [f"-D{key}={value}" for key, value in config["defines"].items()]
    if "STATIC_SCHED" not in config["defines"]:
        flags.append("-DSTATIC_SCHED=0")
    if config["scheduler_runtime"]:
        flags.append("-DPHYAI_SCHEDULER_RUNTIME=1")
    flags += ["-arch=sm_110a", "-lineinfo", "-DNDEBUG"]
    return load_cuda_jit(
        "sm110_bf16",
        target.lower(),
        str(rows),
        digest,
        cuda_files=[source],
        cuda_wrappers=[
            ("create", "phyai_sm110::Create"),
            ("run", "phyai_sm110::Run"),
            ("destroy", "phyai_sm110::Destroy"),
        ],
        extra_cuda_cxx_flags=flags,
        extra_include_paths=[
            str(
                PHYAI_KERNEL_CUDA_CSRC_DIR / "sm110/third_party/cutlass_compat/include"
            ),
            *[str(path) for path in CUTLASS_INCLUDE_DIRS],
        ],
    )


@cache
def load_dsl(target: str, device_index: int):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import make_fake_tensor, make_fake_stream

    config = next(row for row in configurations() if row["target"] == target)
    m, n, k = (config[key] for key in ("m", "n", "k"))
    module = importlib.import_module(
        f"{__package__}.{target.removeprefix('pi05_').lower()}"
    )
    spec = config["dsl"]
    kernel = module.GemmBlackwellFP16(
        acc_dtype=cutlass.Float32,
        mma_tiler_mn=tuple(spec["tile"]),
        cluster_shape_mn=tuple(spec["cluster"]),
        use_2cta=spec["use_2cta"],
        mma_inst_tile_k=spec["k_mult"],
        ab_stages=spec["stages"],
    )
    tensors = [
        make_fake_tensor(cutlass.BFloat16, shape, stride=stride, assumed_align=16)
        for shape, stride in (
            ((n, k, 1), (1, n, n * k)),
            ((m, k, 1), (k, 1, m * k)),
            ((n, m, 1), (1, n, n * m)),
        )
    ]
    return cute.compile(
        kernel,
        *tensors,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --keep-cubin --keep-ptx",
    )


class NativePlan:
    def __init__(self, module, config, x, weight, bias, output):
        stream = torch.cuda.current_stream(x.device).cuda_stream
        self.module = module
        self.handle = module.create(
            x.shape[0],
            config["n"],
            x.shape[1],
            x.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr() if bias is not None else 0,
            output.data_ptr(),
            stream,
            config["swizzle"] or 0,
            config["raster"],
        )
        self.finalizer = weakref.finalize(self, module.destroy, self.handle)

    def __call__(self, x, weight, bias, output):
        self.module.run(
            self.handle,
            x.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr() if bias is not None else 0,
            output.data_ptr(),
            torch.cuda.current_stream(x.device).cuda_stream,
        )


@cache
def module_cache(kind: str, rows: int) -> dict:
    """Share immutable compiled code across selector modes, never plans or weights."""
    return {}


def refresh_packed_weights(layer):
    """Refresh captured storage at the engine's serialized weight-load boundary."""
    state = getattr(layer, "sm110_gemm_state", None)
    if state is None:
        return
    weight = layer.weight
    version = None if weight.is_inference() else weight._version
    identity = (id(weight), weight.data_ptr(), version)
    if state[0][:2] != identity[:2]:
        raise RuntimeError("Rebuild captured SM110 GEMM after replacing weight storage")
    # Weight loaders may copy through .data without incrementing weight._version.
    for packed in state[1].values():
        packed.copy_(weight.t())
    layer.sm110_gemm_state = (identity, state[1], state[2], weight)


class Gemm:
    """One selected geometry; mutable storage belongs to the consuming layer."""

    def __init__(self, target: str, rows: int):
        self.config = next(
            row
            for row in configurations()
            if row["target"] == target and row["m"] == rows
        )
        self.modules = module_cache(target, rows)

    def __call__(
        self, layer, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        config = self.config
        weight = layer.weight
        if (
            not x.is_cuda
            or x.dtype != torch.bfloat16
            or weight.dtype != x.dtype
            or weight.device != x.device
        ):
            raise ValueError("SM110 GEMM requires colocated BF16 input and weight")
        if x.shape[-1] != config["k"] or tuple(weight.shape) != (
            config["n"],
            config["k"],
        ):
            raise ValueError(
                "SM110 GEMM weight or reduction shape changed after selection"
            )
        if (bias is not None) != config["bias"]:
            raise ValueError("SM110 GEMM bias differs from the selected contract")
        if bias is not None and (
            tuple(bias.shape) != (config["n"],)
            or bias.dtype != x.dtype
            or bias.device != x.device
        ):
            raise ValueError("SM110 GEMM requires a colocated BF16 bias vector")
        if not weight.is_contiguous() or weight.data_ptr() % 16:
            raise ValueError("SM110 GEMM requires aligned row-major weights")
        flat = x.reshape(-1, config["k"]).contiguous()
        if flat.data_ptr() % 16:
            flat = flat.clone()
        if bias is not None:
            bias = bias.contiguous()
            if bias.data_ptr() % 16:
                bias = bias.clone()
        if flat.shape[0] != config["m"]:
            raise ValueError("SM110 GEMM row count changed after selection")
        output = torch.empty((config["m"], config["n"]), device=x.device, dtype=x.dtype)
        state = getattr(layer, "sm110_gemm_state", None)
        version = None if weight.is_inference() else weight._version
        identity = (id(weight), weight.data_ptr(), version)
        if state is None or state[0] != identity:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up SM110 GEMM after loading or replacing weights before capture"
                )
            state = (identity, {}, {}, weight)
            layer.sm110_gemm_state = state
            hooks = getattr(layer, "weight_post_load_hooks", None)
            if hooks is not None:
                hooks["sm110_gemm"] = refresh_packed_weights
        packed, plans = state[1:3]
        packed_key = (config["target"], torch.cuda.current_stream(x.device).cuda_stream)
        key = (*packed_key, config["m"])
        with torch.cuda.device(x.device):
            if x.device.index not in self.modules:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm up SM110 GEMM compilation before capture")
                self.modules[x.device.index] = (
                    load_dsl(config["target"], x.device.index)
                    if config["dsl"]
                    else load_native(config["target"], config["m"], x.device.index)
                )
            module = self.modules[x.device.index]
            if config["packed"]:
                if packed_key not in packed:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "Warm up SM110 weight packing before capture"
                        )
                    packed[packed_key] = weight.t().contiguous()
                if version is None:
                    # Inference tensors have no version counter. Capture the
                    # refresh too, so in-place weight updates cannot go stale.
                    packed[packed_key].copy_(weight.t())
                weight = packed[packed_key]
            if config["dsl"]:
                m, n, k = (config[name] for name in ("m", "n", "k"))
                module(
                    weight.as_strided((n, k, 1), (1, n, n * k)),
                    flat.as_strided((m, k, 1), (k, 1, m * k)),
                    output.as_strided((n, m, 1), (1, n, n * m)),
                )
            else:
                if key not in plans:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "Warm up SM110 GEMM on the capture stream before capture"
                        )
                    plans[key] = NativePlan(module, config, flat, weight, bias, output)
                plans[key](flat, weight, bias, output)
        return output.reshape(*x.shape[:-1], config["n"])
