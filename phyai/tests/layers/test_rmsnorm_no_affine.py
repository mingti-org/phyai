"""RMSNorm identity buffers, checkpoint contracts, and kernel parity."""

import pytest
import torch
from torch import nn

from phyai.layers.layer_norm import GemmaRMSNorm, RMSNorm
from phyai.weights import WeightLoadSession


NORM_CLASSES = [
    pytest.param(RMSNorm, id="rms"),
    pytest.param(GemmaRMSNorm, id="gemma"),
]


@pytest.mark.parametrize("norm_cls", NORM_CLASSES)
def test_default_affine_parameter_and_checkpoint_loading(norm_cls: type[RMSNorm]):
    norm = norm_cls(16, device="cpu", dtype=torch.bfloat16, prefix="model.norm")
    assert norm.elementwise_affine
    assert isinstance(norm.weight, nn.Parameter)
    assert set(dict(norm.named_parameters())) == {"weight"}
    assert set(norm.state_dict()) == {"weight"}
    assert norm.weight.hf_keys == [("model.norm.weight", None)]

    weight = torch.linspace(-0.5, 1.5, 16)
    loading = WeightLoadSession(norm)
    loading.load({"model.norm.weight": weight})
    report = loading.finish(strict=True)
    assert report.loaded == ["model.norm.weight"]
    assert not report.missing and not report.unexpected
    torch.testing.assert_close(norm.weight, weight.bfloat16(), atol=0, rtol=0)
    if norm_cls is GemmaRMSNorm:
        with pytest.raises(ValueError, match="standard RMSNorm only"):
            norm_cls(16, device="cpu", cast_before_affine=True)


@pytest.mark.parametrize("norm_cls", NORM_CLASSES)
@pytest.mark.parametrize("cast_before_affine", [False, True])
def test_no_affine_buffer_movement_and_checkpoint_contract(
    norm_cls: type[RMSNorm], cast_before_affine: bool
):
    norm = norm_cls(
        16,
        device="cpu",
        dtype=torch.bfloat16,
        prefix="model.norm",
        elementwise_affine=False,
        cast_before_affine=cast_before_affine,
    )
    identity = 0.0 if norm_cls is GemmaRMSNorm else 1.0
    assert not norm.elementwise_affine
    assert not norm.cast_before_affine
    assert not isinstance(norm.weight, nn.Parameter)
    assert set(dict(norm.named_buffers())) == {"weight"}
    assert not list(norm.parameters())
    assert not norm.state_dict()
    assert not hasattr(norm.weight, "hf_keys")
    assert not hasattr(norm.weight, "weight_loader")
    torch.testing.assert_close(
        norm.weight, torch.full_like(norm.weight, identity), atol=0, rtol=0
    )

    norm.to(dtype=torch.float32)
    assert norm.weight.dtype == torch.float32
    assert not norm.state_dict()
    torch.testing.assert_close(
        norm.weight, torch.full_like(norm.weight, identity), atol=0, rtol=0
    )
    value = torch.linspace(-2.0, 3.0, 32).reshape(2, 16)
    expected = torch.nn.functional.rms_norm(value, (16,), eps=1e-6)
    torch.testing.assert_close(norm(value), expected, atol=1e-6, rtol=1e-6)

    report = WeightLoadSession(norm).finish(strict=True)
    assert not report.loaded and not report.missing and not report.unexpected
    incompatible = WeightLoadSession(norm)
    incompatible.load({"model.norm.weight": torch.ones(16)})
    with pytest.raises(RuntimeError, match="unexpected=1"):
        incompatible.finish(strict=True)


@pytest.mark.parametrize("norm_cls", NORM_CLASSES)
@pytest.mark.parametrize("backend", ["flashinfer", "phyai-kernel"])
@pytest.mark.parametrize("with_residual", [False, True])
def test_no_affine_matches_identity_affine_kernel(
    norm_cls: type[RMSNorm], backend: str, with_residual: bool
):
    torch.manual_seed(507)
    kwargs = dict(dtype=torch.bfloat16, backend=backend, device="cpu")
    norm = norm_cls(128, elementwise_affine=False, **kwargs).to("cuda")
    identity_affine = norm_cls(128, **kwargs).to("cuda")
    assert norm.weight.device.type == "cuda"
    if with_residual:
        value = torch.randn(2, 5, 128, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(value)
        expected, expected_residual = identity_affine(value.clone(), residual.clone())
        actual, actual_residual = norm(value, residual)
        assert actual.data_ptr() == value.data_ptr()
        assert actual_residual.data_ptr() == residual.data_ptr()
        torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)
    else:
        value = torch.randn(2, 3, 4, 128, device="cuda", dtype=torch.bfloat16)
        value = value.transpose(1, 2)
        expected = identity_affine(value)
        actual = norm(value)
    assert actual.shape == value.shape and actual.dtype == value.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["flashinfer", "phyai-kernel"])
def test_no_affine_cuda_graph_replay_uses_changed_input(backend: str):
    torch.manual_seed(508)
    norm = RMSNorm(
        128,
        dtype=torch.bfloat16,
        device="cuda",
        backend=backend,
        elementwise_affine=False,
        cast_before_affine=True,
    )
    value = torch.randn(2, 5, 128, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            norm(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = norm(value)
    graph.replay()
    weight_address = norm.weight.data_ptr()
    previous = captured.clone()
    for _ in range(2):
        replacement = torch.randn_like(value)
        expected = norm(replacement)
        value.copy_(replacement)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        assert not torch.equal(captured, previous)
        assert norm.weight.data_ptr() == weight_address
        previous.copy_(captured)
