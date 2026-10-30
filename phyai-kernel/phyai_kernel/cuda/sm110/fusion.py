"""Optional BF16 fusions preserving intermediate rounding and output ownership."""

from functools import cache
import hashlib
import json
from pathlib import Path

import torch

from .gemm import NativePlan, configurations, module_cache


def prepare_scatter(facts):
    from .kv_scatter import scatter

    return scatter


def projection_config(name, rows):
    target = "pi05_G10" if name == "vision" else "pi05_G14"
    config = dict(
        next(
            row
            for row in configurations()
            if row["target"] == target and row["m"] == rows
        )
    )
    config["defines"] = dict(config["defines"])
    config["source"] = "vision_gelu.cu" if name == "vision" else "prefix_gated.cu"
    config["defines"]["STATIC_SCHED"] = 0
    if name == "prefix":
        config["defines"]["GATED_CHUNK"] = 32
    if name == "prefix" and rows != 528:
        selected = json.loads(
            Path(__file__).with_name("fusion_configs.json").read_text()
        )[str(rows)]
        config["source"] = selected["source"]
        config["defines"] = {**selected["defines"], "STATIC_SCHED": 0}
    return config


@cache
def load_projection(name, rows, device_index):
    from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS
    import flashinfer
    from phyai_kernel.jit_utils.compile import PHYAI_KERNEL_CUDA_CSRC_DIR, load_cuda_jit

    config = projection_config(name, rows)
    directory = PHYAI_KERNEL_CUDA_CSRC_DIR / "sm110"
    includes = [
        config["source"],
        "gemm_ffi.cuh",
        "bf16_exact_gelu_v1.cuh",
        "third_party/cutlass_compat/include/cute/layout.hpp",
    ]
    digest = hashlib.sha256(
        b"".join((directory / path).read_bytes() for path in includes)
        + b"".join(
            path.read_bytes()
            for path in sorted(
                (directory / "third_party/cutlass_compat/include").rglob("*.hpp")
            )
        )
        + json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:16]
    flags = [f"-D{key}={value}" for key, value in config["defines"].items()]
    flags += ["-arch=sm_110a", "-DPHYAI_SCHEDULER_RUNTIME=1", "-DNDEBUG"]
    return load_cuda_jit(
        "sm110_projection",
        name,
        str(rows),
        digest,
        cuda_files=["sm110/" + config["source"]],
        cuda_wrappers=[
            ("create", "phyai_sm110::Create"),
            ("run", "phyai_sm110::Run"),
            ("destroy", "phyai_sm110::Destroy"),
        ],
        extra_cuda_cxx_flags=flags,
        extra_include_paths=[
            str(directory / "third_party/cutlass_compat/include"),
            *[str(path) for path in CUTLASS_INCLUDE_DIRS],
            str(Path(flashinfer.__file__).parent / "data/include"),
        ],
    )


@cache
def gated_table(device_index):
    import flashinfer
    from phyai_kernel.jit_utils.compile import PHYAI_KERNEL_CUDA_CSRC_DIR, load_cuda_jit

    directory = PHYAI_KERNEL_CUDA_CSRC_DIR / "sm110"
    digest = hashlib.sha256(
        (directory / "gated_table.cu").read_bytes()
        + (directory / "prefix_gelu_streaming.cu").read_bytes()
        + (directory / "fusion_ffi.cuh").read_bytes()
    ).hexdigest()[:16]
    module = load_cuda_jit(
        "sm110_gelu_table",
        digest,
        cuda_files=["sm110/gated_table.cu"],
        cuda_wrappers=[("create_table", "phyai_sm110_fusion::CreateTable")],
        extra_cuda_cxx_flags=["-arch=sm_110a", "--use_fast_math"],
        extra_include_paths=[str(Path(flashinfer.__file__).parent / "data/include")],
    )
    table = torch.empty(
        65536, dtype=torch.float32, device=torch.device("cuda", device_index)
    )
    module.create_table(
        table.data_ptr(),
        device_index,
        torch.cuda.current_stream(device_index).cuda_stream,
    )
    # Publish an immutable table only after its one-time warmup has completed.
    # Waiting on an uncaptured initialization event during Graph capture is invalid.
    torch.cuda.current_stream(device_index).synchronize()
    return table


class DenseMLPFusion:
    def __init__(self, facts):
        self.rows = int(facts.lookup("shape.tokens"))
        self.name = "prefix" if facts.lookup("attrs.gated") else "vision"
        self.config = projection_config(self.name, self.rows)
        self.modules = module_cache("projection_" + self.name, self.rows)

    def __call__(self, mlp, x):
        config = self.config
        layer = mlp.gate_up_proj if mlp.gated else mlp.fc1
        weight = layer.weight
        flat = x.reshape(self.rows, config["k"]).contiguous()
        if flat.data_ptr() % 16:
            flat = flat.clone()
        version = None if weight.is_inference() else weight._version
        identity = (id(weight), weight.data_ptr(), version)
        state = getattr(layer, "sm110_mlp_state", None)
        if state is None or state[0] != identity:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm up fused MLP after weight updates before capture"
                )
            state = (identity, {}, {}, weight)
            layer.sm110_mlp_state = state
            hooks = getattr(layer, "weight_post_load_hooks", None)
            if hooks is not None:
                hooks["sm110_mlp"] = refresh_packed_mlp_weights
        packed, plans = state[1:3]
        with torch.cuda.device(x.device):
            index = x.device.index
            stream = torch.cuda.current_stream(x.device).cuda_stream
            key = (self.name, self.rows, stream)
            if index not in self.modules:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm up fused MLP compilation before capture")
                self.modules[index] = load_projection(self.name, self.rows, index)
            if mlp.gated:
                if stream not in packed:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "Warm up fused MLP weight packing before capture"
                        )
                    packed[stream] = (
                        weight.reshape(2, 16384, 2048)
                        .transpose(0, 1)
                        .reshape(32768, 2048)
                        .contiguous()
                    )
                if version is None:
                    packed[stream].reshape(16384, 2, 2048).copy_(
                        weight.reshape(2, 16384, 2048).transpose(0, 1)
                    )
                weight = packed[stream]
                auxiliary = gated_table(index)
            else:
                auxiliary = layer.bias
            output = torch.empty(
                (self.rows, mlp.intermediate_size), dtype=x.dtype, device=x.device
            )
            if key not in plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Warm up fused MLP on its capture stream")
                plans[key] = NativePlan(
                    self.modules[index], config, flat, weight, auxiliary, output
                )
            plans[key](flat, weight, auxiliary, output)
        activated = output.reshape(*x.shape[:-1], mlp.intermediate_size)
        down = mlp.down_proj if mlp.gated else mlp.fc2
        result, _ = down(activated)
        return result


def refresh_packed_mlp_weights(layer):
    state = getattr(layer, "sm110_mlp_state", None)
    if state is None:
        return
    weight = layer.weight
    version = None if weight.is_inference() else weight._version
    identity = (id(weight), weight.data_ptr(), version)
    if state[0][:2] != identity[:2]:
        raise RuntimeError("Rebuild captured SM110 MLP after replacing weight storage")
    for packed in state[1].values():
        packed.reshape(16384, 2, 2048).copy_(
            weight.reshape(2, 16384, 2048).transpose(0, 1)
        )
    layer.sm110_mlp_state = (identity, state[1], state[2], weight)
