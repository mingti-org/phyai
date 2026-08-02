"""PI0.5 LIBERO 策略的轻量接口测试。"""

from __future__ import annotations

import numpy as np

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
