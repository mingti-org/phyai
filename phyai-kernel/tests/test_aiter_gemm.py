"""aiter bf16 GEMM adapter: tuned-config initialization and numerics."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("aiter")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("aiter GEMM needs a ROCm GPU", allow_module_level=True)

from aiter import tuned_gemm  # noqa: E402

from phyai_kernel.aiter import gemm  # noqa: E402

ENV = "AITER_CONFIG_GEMM_BF16"
SHAPE = (50, 1024, 2048, False, str(torch.bfloat16), str(torch.bfloat16))


class _CountingCacheClear:
    def __init__(self, wrapped, fail_first: bool = False):
        self._wrapped = wrapped
        self.calls = 0
        self._fail = fail_first

    def __call__(self, *args, **kwargs):
        return self._wrapped(*args, **kwargs)

    def cache_clear(self):
        self.calls += 1
        if self._fail:
            self._fail = False
            raise RuntimeError("injected")
        self._wrapped.cache_clear()


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(gemm, "_configured", False)
    monkeypatch.delenv(ENV, raising=False)
    return gemm


def _tuned_paths():
    return [
        p
        for p in os.environ.get(ENV, "").split(os.pathsep)
        if p.endswith("aiter_bf16_tuned_gemm.csv")
    ]


def test_tuned_shapes_apply_after_aiter_config_was_read(fresh):
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx

    if (get_gfx(), get_cu_num()) != ("gfx942", 304):
        pytest.skip("the shipped tuned shapes are for MI300X")
    tuned_gemm.get_GEMM_A16W16_config(*SHAPE)
    fresh.ensure_tuned_config()
    assert tuned_gemm.get_GEMM_A16W16_config(*SHAPE)["libtype"] != "torch"


def test_keeps_existing_config_and_runs_once(fresh, monkeypatch, tmp_path):
    user = str(tmp_path / "user.csv")
    monkeypatch.setenv(ENV, user)
    counter = _CountingCacheClear(tuned_gemm.get_GEMM_A16W16_config)
    monkeypatch.setattr(tuned_gemm, "get_GEMM_A16W16_config", counter)
    fresh.ensure_tuned_config()
    fresh.ensure_tuned_config()
    assert os.environ[ENV].split(os.pathsep)[0] == user
    assert len(_tuned_paths()) == 1
    assert counter.calls == 1


def test_failed_initialization_restores_and_retries(fresh, monkeypatch):
    counter = _CountingCacheClear(tuned_gemm.get_GEMM_A16W16_config, fail_first=True)
    monkeypatch.setattr(tuned_gemm, "get_GEMM_A16W16_config", counter)
    with pytest.raises(RuntimeError, match="injected"):
        fresh.ensure_tuned_config()
    assert ENV not in os.environ
    fresh.ensure_tuned_config()
    assert len(_tuned_paths()) == 1
    assert counter.calls == 2


@pytest.mark.parametrize(
    "M,N,K,with_bias",
    [(50, 1024, 2048, False), (768, 1152, 1152, True), (0, 1024, 2048, False)],
)
def test_mm_matches_torch(M, N, K, with_bias):
    torch.manual_seed(0)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16) if with_bias else None
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    gemm.ensure_tuned_config()
    actual = gemm.mm(x, weight, bias)
    expected = F.linear(x, weight, bias)
    assert actual.shape == expected.shape and actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
