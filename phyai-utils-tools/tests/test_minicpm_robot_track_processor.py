"""CPU tests for the MiniCPM-RobotTrack processor contract."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from phyai_utils_tools.models.minicpm_robot_track import (
    MiniCPMRobotTrackProcessedInputs,
    MiniCPMRobotTrackProcessor,
)
from PIL import Image


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(
        self, tasks, *, return_tensors, padding, truncation, max_length
    ) -> dict[str, torch.Tensor]:
        assert tasks == ["follow the red shirt"]
        assert return_tensors == "pt"
        assert padding == "max_length"
        assert truncation is True
        input_ids = torch.zeros(1, max_length, dtype=torch.long)
        attention_mask = torch.zeros(1, max_length, dtype=torch.long)
        input_ids[0, -3:] = torch.tensor([11, 12, 13])
        attention_mask[0, -3:] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_processor(*, resize_workers: int | None = None) -> MiniCPMRobotTrackProcessor:
    return MiniCPMRobotTrackProcessor(
        tokenizer=StubTokenizer(),
        image_size=384,
        history_frames=31,
        text_capacity=35,
        resize_workers=resize_workers,
    )


def raw_frames(height: int = 200, width: int = 300) -> np.ndarray:
    generator = np.random.default_rng(7)
    return generator.integers(0, 256, size=(32, height, width, 3), dtype=np.uint8)


def test_preprocess_shapes_types_and_compact_text() -> None:
    output = make_processor().preprocess(
        {"images": raw_frames(), "task": "follow the red shirt"}
    )
    assert isinstance(output, MiniCPMRobotTrackProcessedInputs)
    assert output.frames.shape == (32, 384, 384, 3)
    assert output.frames.dtype == torch.uint8
    assert output.frames.device.type == "cpu"
    assert output.input_ids.shape == (1, 35)
    assert output.input_ids[0, :3].tolist() == [11, 12, 13]
    assert output.input_ids[0, 3:].eq(0).all()
    assert output.text_lengths.tolist() == [3]


def test_resize_matches_reference_pil_bicubic() -> None:
    frames = raw_frames()[:1]
    bicubic = getattr(Image, "Resampling", Image).BICUBIC
    expected = np.asarray(
        Image.fromarray(frames[0], mode="RGB").resize((384, 384), bicubic),
        dtype=np.uint8,
    )
    output = make_processor().preprocess(
        {"images": frames, "task": "follow the red shirt"}
    )
    assert np.array_equal(output.frames[0].numpy(), expected)


def test_parallel_window_resize_matches_reference_pil_bicubic() -> None:
    frames = raw_frames()
    bicubic = getattr(Image, "Resampling", Image).BICUBIC
    expected = np.stack(
        [
            np.asarray(
                Image.fromarray(frame, mode="RGB").resize((384, 384), bicubic),
                dtype=np.uint8,
            )
            for frame in frames
        ],
        axis=0,
    )
    processor = make_processor(resize_workers=4)
    output = processor.preprocess({"images": frames, "task": "follow the red shirt"})
    processor.close()
    assert np.array_equal(output.frames.numpy(), expected)


def test_processor_close_is_idempotent() -> None:
    processor = make_processor(resize_workers=2)
    processor.preprocess({"images": raw_frames(), "task": "follow the red shirt"})
    processor.close()
    processor.close()


def test_rejects_non_positive_resize_workers() -> None:
    with pytest.raises(ValueError, match="resize_workers"):
        make_processor(resize_workers=0)


def test_accepts_single_chw_float_frame() -> None:
    frame = torch.rand(3, 384, 384)
    output = make_processor().preprocess(
        {"images": frame, "task": "follow the red shirt"}
    )
    assert output.frames.shape == (1, 384, 384, 3)
    assert output.frames.dtype == torch.uint8


def test_rejects_wrong_frame_count_and_float_range() -> None:
    processor = make_processor()
    with pytest.raises(ValueError, match="Expected 1 incremental frame or 32"):
        processor.preprocess(
            {"images": raw_frames()[:2], "task": "follow the red shirt"}
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        processor.preprocess(
            {
                "images": torch.full((1, 384, 384, 3), 2.0),
                "task": "follow the red shirt",
            }
        )


def test_postprocess_returns_cpu_float32() -> None:
    processor = make_processor()
    output = processor.postprocess(torch.ones(1, 8, 3, dtype=torch.float64))
    processor.close()
    assert output.shape == (1, 8, 3)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
