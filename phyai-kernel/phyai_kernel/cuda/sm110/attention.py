"""Split-KV Expert attention with caller-stream temporary storage."""

from functools import cache
import hashlib

import torch


@cache
def load_attention(variant: str, device_index: int):
    from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS
    from phyai_kernel.jit_utils.compile import PHYAI_KERNEL_CUDA_CSRC_DIR, load_cuda_jit

    if variant != "c27":
        raise ValueError(f"Unknown Expert attention variant {variant!r}")
    source = f"sm110/expert_{variant}.cu"
    digest = hashlib.sha256(
        (PHYAI_KERNEL_CUDA_CSRC_DIR / source).read_bytes()
        + (PHYAI_KERNEL_CUDA_CSRC_DIR / "sm110/attention_ffi.cuh").read_bytes()
    ).hexdigest()[:16]
    with torch.cuda.device(device_index):
        module = load_cuda_jit(
            "sm110_expert",
            variant,
            digest,
            cuda_files=[source],
            cuda_wrappers=[
                ("prepare", "phyai_sm110_attention::Prepare"),
                ("run", "phyai_sm110_attention::Run"),
            ],
            extra_cuda_cxx_flags=[
                "-arch=sm_110a",
                "-DCATTN_SPLITS=5",
                "-DPHYAI_TILE_ROWS=64",
            ],
            extra_include_paths=[str(path) for path in CUTLASS_INCLUDE_DIRS],
        )
        module.prepare()
    return module


class ExpertAttention:
    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.modules = {"c27": load_attention("c27", self.device.index)}

    def __call__(self, variant, q, keys, values, indptr, indices, scale):
        if (
            q.shape != (50, 8, 256)
            or q.dtype != torch.bfloat16
            or q.device != self.device
        ):
            raise ValueError(
                "SM110 Expert attention requires colocated BF16 Q[50,8,256]"
            )
        if any(
            t.dtype != q.dtype or t.device != q.device or not t.is_contiguous()
            for t in (keys, values)
        ):
            raise ValueError(
                "SM110 Expert attention requires contiguous BF16 page-size-one KV"
            )
        if keys.shape != values.shape or keys.shape[1:] != (1, 1, 256):
            raise ValueError("SM110 Expert attention requires KV[slots,1,1,256]")
        if q.stride(-1) != 1 or q.data_ptr() % 16 or q.stride(0) % 8 or q.stride(1) % 8:
            q = q.contiguous()
        if any(
            t.dtype != torch.int32 or t.device != q.device or not t.is_contiguous()
            for t in (indptr, indices)
        ):
            raise ValueError(
                "SM110 Expert attention requires contiguous int32 page metadata"
            )
        if indptr.numel() != 2:
            raise ValueError("SM110 Expert attention supports batch size one")
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        partial = torch.empty((5, 400, 256), dtype=torch.float32, device=q.device)
        lse = torch.empty((5, 400), dtype=torch.float32, device=q.device)
        with torch.cuda.device(q.device):
            self.modules[variant].run(
                q.data_ptr(),
                keys.data_ptr(),
                values.data_ptr(),
                indptr.data_ptr(),
                indices.data_ptr(),
                output.data_ptr(),
                partial.data_ptr(),
                lse.data_ptr(),
                50,
                q.stride(0),
                q.stride(1),
                keys.stride(0),
                values.stride(0),
                float(scale),
                torch.cuda.current_stream(q.device).cuda_stream,
            )
        return output
