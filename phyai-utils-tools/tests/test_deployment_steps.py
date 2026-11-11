"""Tests for the deployment-oriented processing options (PIL resize, deltas, suffix)."""

from __future__ import annotations

import numpy as np
import torch
import pytest

import phyai_utils_tools.models.pi05.processor_pi05 as proc_mod
from phyai_utils_tools.models.pi05 import PI05Processor
from phyai_utils_tools.processing.ops.image_ops import normalize_pixels, resize_with_pad
from phyai_utils_tools.models.pi05.steps_pi05 import StateTokenizerPrepareStep
from phyai_utils_tools.processing.steps import (
    TokenizerStep,
    DeltaActionStep,
    ResizeWithPadStep,
)
from phyai_utils_tools.processing.transition import ACTION, PROMPT, STATE, INPUT_IDS


class _StubTokenizer:
    def __call__(
        self, prompts, max_length, padding, padding_side, truncation, return_tensors
    ):
        b = len(prompts)
        return {
            "input_ids": torch.zeros(b, max_length, dtype=torch.long),
            "attention_mask": torch.ones(b, max_length, dtype=torch.long),
        }


class _WordTokenizer:
    """One id per character; enough to check suffix placement and padding."""

    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def __call__(self, prompts, padding=False, truncation=False, verbose=False):
        return {"input_ids": [[ord(c) for c in p] for p in prompts]}


def test_pil_resize_matches_pil_and_is_identity_at_target():
    from PIL import Image

    image = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None]
    out = resize_with_pad(tensor, 32, 32, backend="pil")
    expected = Image.new("RGB", (32, 32))
    expected.paste(
        Image.fromarray(image).resize((32, 24), Image.Resampling.BILINEAR), (0, 4)
    )
    assert torch.equal(out[0].permute(1, 2, 0), torch.from_numpy(np.array(expected)))
    same = resize_with_pad(out, 32, 32, backend="pil")
    assert torch.equal(same, out)


def test_pil_resize_rejects_invalid_inputs():
    floats = torch.rand(1, 3, 8, 8)
    with pytest.raises(ValueError, match="uint8"):
        resize_with_pad(floats, 4, 4, backend="pil")
    pixels = torch.zeros(1, 3, 8, 6, dtype=torch.uint8)
    with pytest.raises(ValueError, match="pad_value"):
        resize_with_pad(pixels, 4, 4, backend="pil", pad_value=-1.0)
    with pytest.raises(ValueError, match="backend"):
        resize_with_pad(pixels, 4, 4, backend="cv2")


def test_uint8_and_float_pixels_normalize_identically():
    pixels = torch.arange(0, 256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    assert torch.equal(
        normalize_pixels(pixels), normalize_pixels(pixels.float() / 255.0)
    )


def test_resize_step_serializes_backend_only_when_non_default():
    torch_step = ResizeWithPadStep(target_size=4, num_images=1, num_channels=3)
    pil_step = ResizeWithPadStep(
        target_size=4, num_images=1, num_channels=3, backend="pil"
    )
    assert "backend" not in torch_step.get_config()
    assert pil_step.get_config()["backend"] == "pil"


def test_delta_action_step_anchors_masked_dims_and_keeps_padding():
    step = DeltaActionStep(delta_mask=[True, False])
    action = torch.zeros(1, 3, 4)
    out = step({ACTION: action, STATE: np.array([2.0, 5.0])})[ACTION]
    assert torch.equal(out[0, :, 0], torch.full((3,), 2.0))
    assert torch.count_nonzero(out[0, :, 1:]) == 0
    assert torch.count_nonzero(action) == 0  # Input is not modified.
    with pytest.raises(ValueError, match="state shape"):
        step({ACTION: action, STATE: np.zeros(3)})
    with pytest.raises(ValueError, match="finite"):
        step({ACTION: action, STATE: np.array([np.nan, 0.0])})


def test_postprocess_passes_raw_state_anchor_unchanged(monkeypatch):
    """With state statistics present, only actions are unnormalized."""
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: _StubTokenizer())
    stats = {
        "observation.state": {"q01": [0.0, 0.0], "q99": [10.0, 10.0]},
        "action": {"q01": [-1.0, -1.0], "q99": [1.0, 1.0]},
    }
    proc = PI05Processor(
        num_images=1,
        action_dim=2,
        dataset_stats=stats,
        device="cpu",
        params_dtype=torch.float32,
    )
    normalized = torch.zeros(1, 3, 2)
    relative = proc.postprocess(normalized)
    proc.postprocessor.steps.insert(1, DeltaActionStep(delta_mask=[True, False]))
    state = np.array([0.5, 0.5], dtype=np.float32)
    absolute = proc.postprocess({"action": normalized, "state": state})
    assert torch.allclose(absolute[..., 0], relative[..., 0] + 0.5)
    assert torch.equal(absolute[..., 1], relative[..., 1])


def test_instruction_prompt_ignores_state_and_round_trips_config():
    step = StateTokenizerPrepareStep(prompt_mode="instruction")
    out = step({"task": ["pick_up\nthe cup "], STATE: torch.zeros(1, 4)})
    assert out[PROMPT] == ["pick up the cup"]
    assert step.get_config() == {"prompt_mode": "instruction"}
    assert StateTokenizerPrepareStep().get_config() == {}
    with pytest.raises(ValueError, match="prompt mode"):
        StateTokenizerPrepareStep(prompt_mode="other")({"task": "x"})


def test_suffix_is_encoded_separately_and_truncated_with_prompt():
    step = TokenizerStep(tokenizer=_WordTokenizer(), max_length=5, suffix="\n")
    out = step({PROMPT: ["ab", "abcdef"]})
    assert out[INPUT_IDS].tolist() == [[97, 98, 10, 0, 0], [97, 98, 99, 100, 101]]
    assert out["lang_lens"].tolist() == [3, 5]
    assert step.get_config()["suffix"] == "\n"
    tokenizer = _WordTokenizer()
    tokenizer.pad_token_id = None
    with pytest.raises(ValueError, match="pad_token_id"):
        TokenizerStep(tokenizer=tokenizer, max_length=5, suffix="\n")({PROMPT: ["a"]})
