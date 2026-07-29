"""CPU tests for the MiniCPM-RobotTrack processor contract."""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator

import numpy as np
import pytest
import torch
from phyai_utils_tools.models.minicpm_robot_track import (
    MiniCPMRobotTrackProcessedInputs,
    MiniCPMRobotTrackProcessor,
    make_minicpm_robot_track_processors,
)
from PIL import Image


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(
        self,
        token_ids: list[int] | None = None,
        *,
        truncation_side: str = "right",
    ) -> None:
        self.token_ids = token_ids or [11, 12, 13]
        self.truncation_side = truncation_side

    def __call__(
        self, tasks, *, return_tensors, padding, truncation
    ) -> dict[str, torch.Tensor]:
        assert tasks == ["follow the red shirt"]
        assert return_tensors == "pt"
        assert padding is False
        assert truncation is False
        input_ids = torch.tensor([self.token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_processor(
    *,
    resize_workers: int | None = None,
    tokenizer: StubTokenizer | None = None,
    text_capacity: int = 35,
) -> MiniCPMRobotTrackProcessor:
    return MiniCPMRobotTrackProcessor(
        tokenizer=tokenizer or StubTokenizer(),
        image_size=384,
        history_frames=31,
        text_capacity=text_capacity,
        resize_workers=resize_workers,
    )


ProcessorFactory = Callable[..., MiniCPMRobotTrackProcessor]


@pytest.fixture
def processor_factory() -> Generator[ProcessorFactory, None, None]:
    created: list[MiniCPMRobotTrackProcessor] = []

    def _make(**kwargs) -> MiniCPMRobotTrackProcessor:
        processor = make_processor(**kwargs)
        created.append(processor)
        return processor

    yield _make
    for processor in created:
        processor.close()


def raw_frames(height: int = 200, width: int = 300) -> np.ndarray:
    generator = np.random.default_rng(7)
    return generator.integers(0, 256, size=(32, height, width, 3), dtype=np.uint8)


def test_preprocess_shapes_types_and_compact_text(
    processor_factory: ProcessorFactory,
) -> None:
    output = processor_factory().preprocess(
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


def test_resize_matches_reference_pil_bicubic(
    processor_factory: ProcessorFactory,
) -> None:
    frames = raw_frames()[:1]
    bicubic = getattr(Image, "Resampling", Image).BICUBIC
    expected = np.asarray(
        Image.fromarray(frames[0], mode="RGB").resize((384, 384), bicubic),
        dtype=np.uint8,
    )
    output = processor_factory().preprocess(
        {"images": frames, "task": "follow the red shirt"}
    )
    assert np.array_equal(output.frames[0].numpy(), expected)


def test_parallel_window_resize_matches_reference_pil_bicubic(
    processor_factory: ProcessorFactory,
) -> None:
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
    processor = processor_factory(resize_workers=4)
    output = processor.preprocess({"images": frames, "task": "follow the red shirt"})
    processor.close()
    assert np.array_equal(output.frames.numpy(), expected)


def test_processor_close_is_idempotent(processor_factory: ProcessorFactory) -> None:
    processor = processor_factory(resize_workers=2)
    processor.preprocess({"images": raw_frames(), "task": "follow the red shirt"})
    processor.close()
    processor.close()


def test_rejects_non_positive_resize_workers(
    processor_factory: ProcessorFactory,
) -> None:
    with pytest.raises(ValueError, match="resize_workers"):
        processor_factory(resize_workers=0)


def test_accepts_single_chw_float_frame(
    processor_factory: ProcessorFactory,
) -> None:
    frame = torch.rand(3, 384, 384)
    output = processor_factory().preprocess(
        {"images": frame, "task": "follow the red shirt"}
    )
    assert output.frames.shape == (1, 384, 384, 3)
    assert output.frames.dtype == torch.uint8


def test_rejects_wrong_frame_count_and_float_range(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
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


def test_postprocess_returns_cpu_float32(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    output = processor.postprocess(torch.ones(1, 8, 3, dtype=torch.float64))
    processor.close()
    assert output.shape == (1, 8, 3)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"


def test_tokenizer_warns_before_truncating(
    processor_factory: ProcessorFactory, caplog: pytest.LogCaptureFixture
) -> None:
    cases = (
        ("right", list(range(35))),
        ("left", list(range(5, 40))),
    )
    for truncation_side, expected in cases:
        caplog.clear()
        processor = processor_factory(
            tokenizer=StubTokenizer(list(range(40)), truncation_side=truncation_side),
            text_capacity=35,
        )
        with caplog.at_level(logging.WARNING):
            output = processor.preprocess(
                {"images": raw_frames()[:1], "task": "follow the red shirt"}
            )
        assert "produced 40 tokens; truncating to text_capacity=35" in caplog.text
        assert output.input_ids[0].tolist() == expected
        assert output.text_lengths.tolist() == [35]


def test_processor_factory_returns_close_handle() -> None:
    preprocessor, postprocessor, processor = make_minicpm_robot_track_processors(
        tokenizer=StubTokenizer(), resize_workers=2
    )
    try:
        assert preprocessor is processor.preprocessor
        assert postprocessor is processor.postprocessor
    finally:
        processor.close()
