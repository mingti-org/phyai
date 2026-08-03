"""PI0.5 LIBERO 策略的轻量接口测试。"""

from __future__ import annotations

import numpy as np
import torch

from phyai.policies.pi05_libero import PI05LiberoPolicy


def test_single_infer_delegates_to_one_item_batch(monkeypatch):
    policy = object.__new__(PI05LiberoPolicy)
    expected = {"actions": np.zeros((1, 10, 7), dtype=np.float32)}
    calls = []

    def infer_batch(obs_batch, *, noise=None):
        calls.append((obs_batch, noise))
        return expected

    monkeypatch.setattr(policy, "infer_batch", infer_batch)
    observation = {"state": np.zeros(7, dtype=np.float32)}

    assert policy.infer(observation) is expected
    assert calls == [([observation], None)]


def test_single_card_constructor_defaults_are_preserved():
    defaults = PI05LiberoPolicy.__init__.__kwdefaults__

    assert defaults is not None
    assert defaults["max_batch_size"] == 1
    assert defaults["engine_plugin"] == "pi05"
    assert defaults["world_size"] == 1
    assert defaults["dp_size"] == 1


def test_infer_batch_concatenates_inputs_and_steps_once(monkeypatch):
    policy = object.__new__(PI05LiberoPolicy)
    policy.device = "cpu"
    calls = []

    def observation_to_request_inputs(obs):
        value = int(obs["value"])
        return {
            "pixel_values": torch.full((1, 2, 3, 4, 4), value, dtype=torch.float32),
            "input_ids": torch.full((1, 3), value, dtype=torch.int64),
            "lang_lens": torch.tensor([value], dtype=torch.int64),
        }

    class _Engine:
        def step(self, request):
            calls.append(request)
            return torch.zeros(2, 10, 7)

    monkeypatch.setattr(
        policy, "observation_to_request_inputs", observation_to_request_inputs
    )
    monkeypatch.setattr(policy, "_postprocess_actions", lambda actions: actions.numpy())
    policy.engine = _Engine()

    result = policy.infer_batch([{"value": 1}, {"value": 2}])

    assert len(calls) == 1
    request = calls[0]
    assert request.pixel_values.shape == (2, 2, 3, 4, 4)
    assert request.input_ids.tolist() == [[1, 1, 1], [2, 2, 2]]
    assert request.lang_lens.tolist() == [1, 2]
    assert result["actions"].shape == (2, 10, 7)
