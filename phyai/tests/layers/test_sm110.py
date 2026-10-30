"""SM110 layer integration with the engine's real selector and graph wrapper."""

from pathlib import Path

import pytest
import torch

from phyai.kernel.bootstrap import kernel_selector_scope
from phyai.kernel.execution import current_execution, execution_phase
from phyai.kernel.policy import load_policy
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import ModelContext
from phyai.layers.linear import ReplicatedLinear
from phyai.runtime.cuda_graph_manager import CudaGraph, capture_on_stream

pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (11, 0),
    reason="requires SM110",
)
POLICIES = Path(__file__).resolve().parents[3] / "examples/pi05/kernel_policies"


def selector(filename=None):
    catalog = build_catalog()
    return Selector(
        catalog,
        load_policy(POLICIES / filename, catalog) if filename else None,
        device="cuda",
        model=ModelContext("pi05"),
    )


@pytest.mark.parametrize("override", (False, True))
@torch.no_grad()
def test_real_linear_changed_input_graph_and_partition_fallback(override):
    with kernel_selector_scope(selector("thor_bf16_gemm.yaml")):
        layer = ReplicatedLinear(32, 1024, params_dtype=torch.bfloat16, device="cuda")
        layer.weight.normal_(std=0.03)
        layer.bias.zero_()
        x = torch.randn(50, 32, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream() if override else None
        graph = CudaGraph()
        with capture_on_stream(stream):
            graph.capture(lambda x: layer(x)[0], {"x": x})
        x.add_(0.25)
        output = graph.replay({"x": x}).clone()
        expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
        if override:
            assert not hasattr(layer, "sm110_gemm_state")
        else:
            assert hasattr(layer, "sm110_gemm_state")
        graph.reset()


@torch.no_grad()
def test_frozen_inference_and_rollout_keep_separate_gemm_bindings():
    with kernel_selector_scope(selector("thor_bf16_gemm.yaml")):
        layer = ReplicatedLinear(32, 1024, params_dtype=torch.bfloat16, device="cuda")
        layer.weight.normal_(std=0.03)
        layer.bias.zero_()
        x = torch.randn(50, 32, device="cuda", dtype=torch.bfloat16)
        inference = CudaGraph()
        inference.capture(lambda x: layer(x)[0], {"x": x})
        inference_choices = layer.gemm_call.freeze()
        assert {c.kernel_id for c in inference_choices.values()} == {
            "phyai_kernel.gemm.sm110"
        }
        rollout = CudaGraph()
        with execution_phase("rollout"):
            rollout.capture(lambda x: layer(x)[0], {"x": x})
            rollout_choices = layer.gemm_rollout_call.freeze()
        assert layer.gemm_call is not layer.gemm_rollout_call
        assert {c.kernel_id for c in rollout_choices.values()} == {
            "flashinfer.gemm.bf16"
        }
        for delta in (0.1, -0.25):
            x.add_(delta)
            expected = torch.nn.functional.linear(x, layer.weight, layer.bias)
            inferred = inference.replay({"x": x}).clone()
            rolled = rollout.replay({"x": x}).clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(inferred, expected, atol=0.02, rtol=0.02)
            torch.testing.assert_close(rolled, expected, atol=0.02, rtol=0.02)
        with pytest.raises(RuntimeError, match="phase reset"):
            with execution_phase("rollout"):
                raise RuntimeError("phase reset")
        assert current_execution() == "inference"
        torch.testing.assert_close(layer(x)[0], inferred, atol=0, rtol=0)
        inference.reset()
        rollout.reset()


@torch.no_grad()
def test_packed_linear_weight_load_refreshes_existing_graph():
    from phyai.weights.loader import WeightLoadSession

    with kernel_selector_scope(selector("thor_bf16_gemm.yaml")):
        layer = ReplicatedLinear(
            2048,
            2048,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            prefix="projection",
        )
        layer.weight.normal_(std=0.02)
        x = torch.randn(528, 2048, device="cuda", dtype=torch.bfloat16)
        graph = CudaGraph()
        graph.capture(lambda x: layer(x)[0], {"x": x})
        original = graph.replay({"x": x}).clone()
        updated_weight = layer.weight.clone().mul_(0.5)
        update = WeightLoadSession(layer)
        update.load({"projection.weight": updated_weight})
        update.finish(strict=True, require_all=True)
        actual = graph.replay({"x": x}).clone()
        expected = torch.nn.functional.linear(x, updated_weight)
        torch.cuda.synchronize()
        assert not torch.equal(original, actual)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        graph.reset()
        # Replacing .data keeps the Parameter object but invalidates raw pointers.
        layer.weight.data = layer.weight.data.clone()
        with pytest.raises(RuntimeError, match="Rebuild captured"):
            layer.post_load()
