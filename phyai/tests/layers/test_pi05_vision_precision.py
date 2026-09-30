"""PI0.5 vision precision boundaries for BF16 engines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import phyai.models.pi05.model_runner_pi05 as runner_mod
import phyai.models.pi05.scheduler_pi05 as scheduler_mod
from phyai.models.pi05.configuration_pi05 import PI05Config, SiglipVisionConfig
from phyai.models.pi05.modeling_pi05 import (
    SiglipVisionEmbeddings,
    SiglipVisionModel,
)
from phyai.models.pi05.scheduler_pi05 import PI05Scheduler


class _CaptureEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_dtype: torch.dtype | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.input_dtype = value.dtype
        return value


def test_bf16_vision_stem_stays_fp32_until_encoder() -> None:
    config = SiglipVisionConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        image_size=4,
        patch_size=2,
        projection_dim=8,
    )
    embeddings = SiglipVisionEmbeddings(config, params_dtype=torch.bfloat16)
    embeddings.patch_embedding.weight.data.normal_()
    assert embeddings.patch_embedding.bias is not None
    embeddings.patch_embedding.bias.data.normal_()
    embeddings.position_embedding.weight.data.normal_()
    embeddings.patch_embedding.post_load()

    patch_dtypes: list[tuple[torch.dtype, torch.dtype]] = []
    embeddings.patch_embedding.register_forward_hook(
        lambda _module, inputs, output: patch_dtypes.append(
            (inputs[0].dtype, output.dtype)
        )
    )
    encoder = _CaptureEncoder()
    model = object.__new__(SiglipVisionModel)
    nn.Module.__init__(model)
    model.embeddings = embeddings
    model.encoder = encoder
    model.post_layernorm = nn.Identity()

    pixels = torch.randn(2, 3, 4, 4, dtype=torch.float32, device="cuda")
    actual = model(pixels)
    patch = F.conv2d(
        pixels,
        embeddings.patch_embedding.weight.float(),
        embeddings.patch_embedding.bias.float(),
        stride=embeddings.patch_embedding.stride,
        padding=embeddings.patch_embedding.padding,
    )
    expected = (
        patch.flatten(2).transpose(1, 2) + embeddings.position_embedding.weight.float()
    ).to(torch.bfloat16)

    assert patch_dtypes == [(torch.float32, torch.float32)]
    assert encoder.input_dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _Stub:
    def __init__(self, *_args, **_kwargs) -> None:
        pass


class _RecordingGraph:
    input_dtype: torch.dtype | None = None

    def capture(self, _fn, inputs: dict[str, torch.Tensor]) -> None:
        type(self).input_dtype = inputs["pixel_values"].dtype


class _RecordingNoiseGraph:
    input_dtype: torch.dtype | None = None

    def capture(self, _fn, inputs: dict[str, torch.Tensor]) -> None:
        type(self).input_dtype = inputs["noise"].dtype


def test_bf16_scheduler_keeps_vision_inputs_fp32(monkeypatch) -> None:
    monkeypatch.setattr(scheduler_mod, "KVCachePool", _Stub)
    monkeypatch.setattr(scheduler_mod, "StaticCache", _Stub)
    monkeypatch.setattr(scheduler_mod, "PI05LLMRunner", _Stub)
    monkeypatch.setattr(scheduler_mod, "PI05ExpertRunner", _Stub)
    monkeypatch.setattr(runner_mod, "CudaGraph", _RecordingGraph)

    cfg = PI05Config()
    vision = nn.Identity()
    vision.config = cfg.vision
    model = _Stub()
    model.config = cfg
    model.params_dtype = torch.bfloat16
    model.vision = vision
    model.paligemma_lm = object()
    model.expert_stack = object()
    model.heads = object()
    model.rope = object()
    model.value_head = None
    scheduler = PI05Scheduler(
        model,
        max_batch_size=1,
        num_images=2,
        device="cuda",
        use_cuda_graph=True,
    )
    scheduler.vision_runner.setup()

    assert scheduler.vision_input_dtype == torch.float32
    assert scheduler.sampler_dtype == torch.float32
    assert scheduler.vision_runner.params_dtype == torch.float32
    assert _RecordingGraph.input_dtype == torch.float32


def test_bf16_inference_graph_keeps_sampler_state_fp32(monkeypatch) -> None:
    monkeypatch.setattr(runner_mod, "CudaGraph", _RecordingNoiseGraph)
    runner = object.__new__(runner_mod.PI05ExpertRunner)
    runner.batch_size = 1
    runner.chunk_size = 2
    runner.max_action_dim = 4
    runner.params_dtype = torch.bfloat16
    runner.device = torch.device("cuda")
    runner._capture_graph()

    assert _RecordingNoiseGraph.input_dtype == torch.float32
