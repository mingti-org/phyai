"""CPU tests for Pi05 checkpoint compatibility and synchronous dispatch."""

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import pytest
from phyai.kernel.policy import load_policy
from phyai.kernel.registry import build_catalog
from phyai.models.pi05.scheduler_pi05 import PI05Scheduler
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.model_runner_rlt_pi05 import PI05RLTRunner

from examples.pi05 import run_rlinf


def test_rlinf_default_kernel_policy(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_rlinf, "Engine", lambda args: args)
    args = run_rlinf.make_engine(tmp_path)
    policy = load_policy(args.config.kernel.config_path, build_catalog())
    assert {rule.use for rule in policy.overrides} >= {
        "torch.activation.gelu",
        "flashinfer.activation.gelu_and_mul",
    }


@pytest.mark.parametrize("activation", [None, "gelu", "gelu_tanh"])
def test_checkpoint_activation(activation):
    metadata = {} if activation is None else {"hidden_activation": activation}
    cfg = PI05Config.from_dict(dict.fromkeys(("vision", "text", "expert"), metadata))
    assert [part.hidden_activation for part in (cfg.vision, cfg.text, cfg.expert)] == [
        activation or "gelu"
    ] * 3


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_rlt_runner_rejects_non_boolean_mask_before_dispatch(graph, dtype):
    model = Mock(input_dim=4, max_visual_tokens=2, rl_token=torch.zeros(1, 1, 4))
    runner = PI05RLTRunner(
        model, batch_size=1, visual_tokens=2, device="cpu", use_cuda_graph=graph
    )
    if graph:
        runner.graph = Mock()
    with pytest.raises(TypeError, match="torch.bool"):
        runner.forward(torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=dtype))
    model.encode.assert_not_called()
    if graph:
        runner.graph.replay.assert_not_called()


def test_action_only_step_releases_prefix_before_denoising():
    prefix_ref = None

    def prepare_prefix(request):
        nonlocal prefix_ref
        prefix = torch.empty(1, 2)
        prefix_ref = weakref.ref(prefix)
        return prefix, None

    def denoise(noise):
        assert prefix_ref() is None
        return noise

    request = SimpleNamespace(pixel_values=torch.empty(1), noise=torch.ones(1, 1, 2))
    scheduler = SimpleNamespace(
        cfg=PI05Config(chunk_size=1, max_action_dim=2),
        device="cpu",
        _validate=lambda request: None,
        max_batch_size=1,
        _prepare_prefix=prepare_prefix,
        rlt_runner=None,
        time_emb_table=torch.empty(1),
        sampler_dtype=torch.float32,
        expert_runner=SimpleNamespace(forward=denoise),
    )
    torch.testing.assert_close(PI05Scheduler.step(scheduler, request), request.noise)
