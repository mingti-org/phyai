from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from phyai_kernel.cuda.pi05_thor_nvfp4_geglu import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MERGED_SIZE,
    nvfp4_scale_shape,
    pi05_thor_nvfp4_geglu,
    supports_pi05_thor_nvfp4_geglu,
)


def test_nvfp4_scale_shape_matches_128x4_layout():
    assert nvfp4_scale_shape(784, HIDDEN_SIZE) == (896, 128)
    assert nvfp4_scale_shape(MERGED_SIZE, HIDDEN_SIZE) == (32768, 128)
    assert nvfp4_scale_shape(784, INTERMEDIATE_SIZE) == (896, 1024)


def test_nvfp4_scale_shape_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="rows must be positive"):
        nvfp4_scale_shape(0, HIDDEN_SIZE)
    with pytest.raises(ValueError, match="divisible by 16"):
        nvfp4_scale_shape(1, 15)


def test_host_tensors_do_not_select_thor_kernel():
    tensors = [torch.empty(1, dtype=torch.uint8) for _ in range(7)]
    assert not supports_pi05_thor_nvfp4_geglu(*tensors)
    with pytest.raises(ValueError, match="CUDA tensors"):
        pi05_thor_nvfp4_geglu(*tensors)


def _thor_test_enabled() -> bool:
    return bool(
        os.environ.get("PHYAI_RUN_THOR_KERNEL_TESTS") == "1"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (11, 0)
    )


@pytest.mark.skipif(
    not _thor_test_enabled(),
    reason="set PHYAI_RUN_THOR_KERNEL_TESTS=1 on an SM110 device",
)
def test_thor_nvfp4_geglu_correctness_and_graph_capture():
    from flashinfer import gemm as flashinfer_gemm
    from flashinfer.quantization import SfLayout, nvfp4_quantize

    m = 784
    generator = torch.Generator(device="cuda").manual_seed(20260814)
    x = torch.randn(
        m,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    interleaved_weight_bf16 = (
        torch.randn(
            MERGED_SIZE,
            HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    down_weight_bf16 = (
        torch.randn(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    activation, activation_scale = nvfp4_quantize(
        x,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        enable_pdl=False,
    )
    interleaved_weight, weight_scale = nvfp4_quantize(
        interleaved_weight_bf16,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        enable_pdl=False,
    )
    down_weight, down_weight_scale = nvfp4_quantize(
        down_weight_bf16,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        enable_pdl=False,
    )
    workspace = torch.zeros(m, INTERMEDIATE_SIZE, dtype=torch.uint8, device="cuda")
    output = torch.empty(m, INTERMEDIATE_SIZE // 2, dtype=torch.uint8, device="cuda")
    output_scale = torch.zeros(
        nvfp4_scale_shape(m, INTERMEDIATE_SIZE),
        dtype=torch.uint8,
        device="cuda",
    )

    assert supports_pi05_thor_nvfp4_geglu(
        activation,
        activation_scale,
        interleaved_weight,
        weight_scale,
        workspace,
        output,
        output_scale,
    )
    pi05_thor_nvfp4_geglu(
        activation,
        activation_scale,
        interleaved_weight,
        weight_scale,
        workspace,
        output,
        output_scale,
    )
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        pi05_thor_nvfp4_geglu(
            activation,
            activation_scale,
            interleaved_weight,
            weight_scale,
            workspace,
            output,
            output_scale,
        )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        pi05_thor_nvfp4_geglu(
            activation,
            activation_scale,
            interleaved_weight,
            weight_scale,
            workspace,
            output,
            output_scale,
        )
    graph.replay()
    torch.cuda.synchronize()

    actual = flashinfer_gemm.mm_fp4(
        output,
        down_weight.t(),
        output_scale,
        down_weight_scale.t().view(torch.uint8),
        global_scale,
        torch.bfloat16,
        None,
        block_size=16,
        use_nvfp4=True,
        backend="cudnn",
    )
    staged_merged = flashinfer_gemm.mm_fp4(
        activation,
        interleaved_weight.t(),
        activation_scale,
        weight_scale.t().view(torch.uint8),
        global_scale,
        torch.bfloat16,
        None,
        block_size=16,
        use_nvfp4=True,
        backend="cudnn",
    )
    staged_hidden = (
        F.gelu(staged_merged[:, 0::2], approximate="tanh") * staged_merged[:, 1::2]
    )
    staged_output, staged_output_scale = nvfp4_quantize(
        staged_hidden,
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        enable_pdl=False,
    )
    expected = flashinfer_gemm.mm_fp4(
        staged_output,
        down_weight.t(),
        staged_output_scale,
        down_weight_scale.t().view(torch.uint8),
        global_scale,
        torch.bfloat16,
        None,
        block_size=16,
        use_nvfp4=True,
        backend="cudnn",
    )
    difference = actual.float() - expected.float()
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        expected.float()
    ).clamp_min(1.0e-12)
    cosine = F.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert relative_l2.item() < 0.05
    assert cosine.item() > 0.999
