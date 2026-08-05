"""Tests for MiniCPMGR00TProcessor — pre/post pipelines with a stub processor.

Avoids a network/HF dependency by injecting a stub in place of the MiniCPM-V
``AutoProcessor``; image conversion, prompt assembly, and state
canonicalization are exercised for real. The chat-template encode is the only
stub (numerical parity of the real processor is validated end-to-end against
the reference inference stack, not here).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from phyai_utils_tools.models.minicpm_gr00t import (
    MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE,
    MiniCPMGR00TProcessedInputs,
    MiniCPMGR00TProcessor,
)
from phyai_utils_tools.models.minicpm_gr00t.ops_minicpm_gr00t import to_uint8_image


class _StubMiniCPMProcessor:
    """Minimal ``apply_chat_template`` stand-in recording its inputs."""

    def __init__(self, seq_len: int = 214):
        self.seq_len = seq_len
        self.last_messages = None
        self.last_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        num_images = sum(
            1 for item in messages[0]["content"] if item.get("type") == "image"
        )
        return {
            "input_ids": torch.ones(1, self.seq_len, dtype=torch.int32),
            "attention_mask": torch.ones(1, self.seq_len, dtype=torch.int32),
            "pixel_values": torch.zeros(1, 3, 14, 28_672),
            "target_sizes": torch.full((num_images, 2), 32, dtype=torch.int32),
        }


def _make_processor(**kwargs) -> MiniCPMGR00TProcessor:
    defaults = dict(
        processor=_StubMiniCPMProcessor(),
        image_size=(448, 448),
        num_images=2,
        state_dim=80,
    )
    defaults.update(kwargs)
    return MiniCPMGR00TProcessor(**defaults)


def _raw(image_hw: tuple[int, int] = (256, 256)) -> dict:
    rng = np.random.default_rng(0)
    return {
        "images": [
            rng.integers(0, 256, size=(*image_hw, 3), dtype=np.uint8) for _ in range(2)
        ],
        "task": "open the middle drawer of the cabinet",
        "state": np.zeros(80, dtype=np.float32),
    }


def test_preprocess_shapes_and_types():
    proc = _make_processor()
    out = proc.preprocess(_raw())
    assert isinstance(out, MiniCPMGR00TProcessedInputs)
    assert out.input_ids.shape == (1, 214) and out.input_ids.dtype == torch.int64
    assert out.attention_mask.shape == (1, 214)
    assert out.attention_mask.dtype == torch.int64
    assert out.pixel_values.shape == (1, 3, 14, 28_672)
    assert out.target_sizes.shape == (2, 2)
    assert out.state.shape == (1, 1, 80) and out.state.dtype == torch.float32


def test_prompt_uses_template_and_passthrough():
    stub = _StubMiniCPMProcessor()
    proc = _make_processor(processor=stub)
    proc.preprocess(_raw())
    text = stub.last_messages[0]["content"][-1]["text"]
    assert text == MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE.format(
        instruction="open the middle drawer of the cabinet"
    )

    already_formatted = MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE.format(
        instruction="put the bowl on the stove"
    )
    raw = _raw()
    raw["task"] = already_formatted
    proc.preprocess(raw)
    assert stub.last_messages[0]["content"][-1]["text"] == already_formatted


def test_chat_template_call_matches_reference_contract():
    stub = _StubMiniCPMProcessor()
    proc = _make_processor(processor=stub)
    proc.preprocess(_raw())
    assert stub.last_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "processor_kwargs": {"padding": False},
    }
    content = stub.last_messages[0]["content"]
    assert [item["type"] for item in content] == ["image", "image", "text"]
    for item in content[:2]:
        assert item["image"].shape == (448, 448, 3)
        assert item["image"].dtype == np.uint8


def test_image_prepare_matches_reference_pil_resize():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(200, 300, 3), dtype=np.uint8)
    expected = np.asarray(Image.fromarray(image).resize((448, 448)))
    actual = to_uint8_image(image, (448, 448))
    assert np.array_equal(actual, expected)

    as_float = image.astype(np.float32) / 255.0
    round_trip = to_uint8_image(as_float, (448, 448))
    direct = to_uint8_image(
        np.clip(as_float * 255.0, 0, 255).astype(np.uint8), (448, 448)
    )
    assert np.array_equal(round_trip, direct)


def test_state_shapes_accepted_and_rejected():
    proc = _make_processor()
    for state in (
        np.zeros(80, dtype=np.float32),
        np.zeros((1, 80), dtype=np.float32),
        np.zeros((1, 1, 80), dtype=np.float32),
        torch.zeros(80),
    ):
        raw = _raw()
        raw["state"] = state
        assert proc.preprocess(raw).state.shape == (1, 1, 80)

    raw = _raw()
    raw["state"] = np.zeros(79, dtype=np.float32)
    with pytest.raises(ValueError, match="80D"):
        proc.preprocess(raw)

    raw = _raw()
    raw["state"] = None
    with pytest.raises(ValueError, match="state"):
        proc.preprocess(raw)


def test_wrong_image_count_rejected():
    proc = _make_processor()
    raw = _raw()
    raw["images"] = raw["images"][:1]
    with pytest.raises(ValueError, match="exactly 2"):
        proc.preprocess(raw)


def test_postprocess_returns_cpu_float32():
    proc = _make_processor()
    action = torch.rand(1, 30, 80, dtype=torch.float64)
    out = proc.postprocess(action)
    assert out.shape == (1, 30, 80)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"


def test_postprocess_slices_action_dim():
    proc = _make_processor(action_dim=7)
    out = proc.postprocess(torch.rand(1, 30, 80))
    assert out.shape == (1, 30, 7)
