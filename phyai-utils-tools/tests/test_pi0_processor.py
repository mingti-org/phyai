"""Tests for PI0Processor with an offline tokenizer stub."""

from __future__ import annotations

import torch

import phyai_utils_tools.models.pi0.processor_pi0 as proc_mod
from phyai_utils_tools.models.pi0 import PI0ProcessedInputs, PI0Processor


class _StubTokenizer:
    """Minimal HF-tokenizer-like callable that records prompts."""

    def __init__(self, real_len: int = 5):
        self.real_len = real_len
        self.prompts: list[str] = []

    def __call__(
        self, prompts, max_length, padding, padding_side, truncation, return_tensors
    ):
        self.prompts = list(prompts)
        b = len(prompts)
        ids = torch.zeros(b, max_length, dtype=torch.long)
        ids[:, 0] = 2
        mask = torch.zeros(b, max_length, dtype=torch.long)
        mask[:, : self.real_len] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _make_processor(monkeypatch, **kwargs) -> tuple[PI0Processor, _StubTokenizer]:
    tok = _StubTokenizer()
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: tok)
    defaults = dict(
        image_size=224,
        num_channels=3,
        num_images=3,
        tokenizer_max_length=48,
        max_state_dim=32,
        action_dim=7,
        device="cpu",
        params_dtype=torch.float32,
    )
    defaults.update(kwargs)
    return PI0Processor(**defaults), tok


def test_preprocess_shapes_and_prompt_newline(monkeypatch):
    proc, tok = _make_processor(monkeypatch)
    b = 3
    raw = {
        "images": [
            torch.rand(b, 3, 480, 640),
            torch.rand(b, 3, 224, 224),
            torch.rand(b, 3, 240, 320),
        ],
        "task": ["pick up the cup", "open drawer\n", "press button"],
        "state": torch.rand(b, 7),
    }

    out = proc.preprocess(raw)

    assert isinstance(out, PI0ProcessedInputs)
    assert out.pixel_values.shape == (b, 3, 3, 224, 224)
    assert out.input_ids.shape == (b, 48) and out.input_ids.dtype == torch.int64
    assert out.lang_lens.shape == (b,) and out.lang_lens.tolist() == [5, 5, 5]
    assert out.state.shape == (b, 32)
    assert torch.allclose(out.state[:, 7:], torch.zeros(b, 25))
    assert tok.prompts == ["pick up the cup\n", "open drawer\n", "press button\n"]


def test_preprocess_two_camera_shapes(monkeypatch):
    proc, _ = _make_processor(monkeypatch, num_images=2)
    b = 2
    raw = {
        "images": [
            torch.rand(b, 3, 480, 640),
            torch.rand(b, 3, 224, 224),
        ],
        "task": ["pick up the cup"] * b,
        "state": torch.rand(b, 7),
    }

    out = proc.preprocess(raw)

    assert isinstance(out, PI0ProcessedInputs)
    assert out.pixel_values.shape == (b, 2, 3, 224, 224)
    assert out.input_ids.shape == (b, 48)
    assert out.state.shape == (b, 32)


def test_preprocess_stacked_tensor_input_and_pixel_normalize(monkeypatch):
    proc, _ = _make_processor(monkeypatch)
    b = 2
    raw = {
        "images": torch.full((b, 3, 3, 224, 224), 0.5),
        "task": ["do a thing"] * b,
        "state": torch.rand(b, 5),
    }

    out = proc.preprocess(raw)

    assert out.pixel_values.shape == (b, 3, 3, 224, 224)
    assert torch.allclose(out.pixel_values, torch.zeros_like(out.pixel_values))


def test_state_mean_std_normalization_before_padding(monkeypatch):
    stats = {
        "observation.state": {
            "mean": torch.tensor([1.0, 2.0]),
            "std": torch.tensor([2.0, 4.0]),
        }
    }
    proc, _ = _make_processor(monkeypatch, dataset_stats=stats)
    raw = {
        "images": torch.rand(1, 3, 3, 224, 224),
        "task": "normalize state",
        "state": torch.tensor([[3.0, 10.0]]),
    }

    out = proc.preprocess(raw)

    assert torch.allclose(out.state[:, :2], torch.tensor([[1.0, 2.0]]), atol=1e-6)
    assert torch.allclose(out.state[:, 2:], torch.zeros(1, 30))


def test_postprocess_trims_then_unnormalizes(monkeypatch):
    stats = {
        "action": {
            "mean": torch.arange(7, dtype=torch.float32),
            "std": torch.full((7,), 2.0),
        }
    }
    proc, _ = _make_processor(monkeypatch, dataset_stats=stats, action_dim=7)
    action = torch.ones(2, 50, 32)

    out = proc.postprocess(action)

    expected = torch.ones(2, 50, 7) * 2.0 + torch.arange(7, dtype=torch.float32)
    assert out.shape == (2, 50, 7)
    assert out.device.type == "cpu"
    assert torch.allclose(out, expected)


def test_save_then_from_pretrained_roundtrip(monkeypatch, tmp_path):
    proc, tok = _make_processor(monkeypatch, num_images=3, action_dim=7)
    proc.save_pretrained(tmp_path)
    assert (tmp_path / "policy_preprocessor.json").exists()
    assert (tmp_path / "policy_postprocessor.json").exists()

    loaded = PI0Processor.from_pretrained(
        tmp_path,
        tokenizer=tok,
        num_images=2,
        image_size=224,
        action_dim=7,
        device="cpu",
        params_dtype=torch.float32,
    )
    b = 2
    raw = {
        "images": torch.rand(b, 2, 3, 224, 224),
        "task": ["do a thing"] * b,
        "state": torch.rand(b, 5),
    }

    out = loaded.preprocess(raw)
    assert isinstance(out, PI0ProcessedInputs)
    assert out.pixel_values.shape == (b, 2, 3, 224, 224)
    assert out.state.shape == (b, 32)
    assert loaded.postprocess(torch.rand(b, 50, 32)).shape == (b, 50, 7)


def test_make_pi0_processors_factory(monkeypatch):
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: _StubTokenizer())
    pre, post = proc_mod.make_pi0_processors(action_dim=7, device="cpu")
    assert pre.name == "pi0_preprocessor"
    assert post.name == "pi0_postprocessor"
