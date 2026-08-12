"""Offline unit tests for the LingBot-VLA 2.0 processor."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import torch

from phyai_utils_tools.models.lingbot_v2 import (
    LingBotV2ProcessedInputs,
    LingBotV2Processor,
    make_lingbot_v2_processors,
)
from phyai_utils_tools.processing.steps import NormalizerStep


class StubTokenizer:
    """Small Qwen-tokenizer-compatible stub."""

    def __init__(self, real_length: int = 6) -> None:
        self.real_length = real_length
        self.chat_inputs: list[str] = []
        self.encoded_prompts: list[str] = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ) -> str:
        assert not tokenize
        assert not add_generation_prompt
        content = messages[0]["content"]
        self.chat_inputs.append(content)
        return f"<qwen-user>{content}</qwen-user>"

    def __call__(
        self,
        prompts,
        *,
        max_length,
        padding,
        padding_side,
        truncation,
        return_tensors,
    ):
        assert padding == "max_length"
        assert padding_side == "right"
        assert truncation
        assert return_tensors == "pt"
        self.encoded_prompts = list(prompts)
        batch_size = len(prompts)
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.int64)
        input_ids[:, : self.real_length] = torch.arange(
            1,
            self.real_length + 1,
        )
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:, : self.real_length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class StubQwenImageProcessor:
    """Return deterministic Qwen-style flattened patches and grid rows."""

    def __init__(self, patch_vector_dim: int = 12) -> None:
        self.patch_vector_dim = patch_vector_dim
        self.received_dtypes: list[torch.dtype] = []

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        grids: list[list[int]] = []
        patches: list[torch.Tensor] = []
        for image_index, image in enumerate(images):
            self.received_dtypes.append(image.dtype)
            grid_h = int(image.shape[-2]) // 16
            grid_w = int(image.shape[-1]) // 16
            grids.append([1, grid_h, grid_w])
            count = grid_h * grid_w
            patches.append(
                torch.full(
                    (count, self.patch_vector_dim),
                    float(image_index + 1),
                )
            )
        return {
            "pixel_values": torch.cat(patches, dim=0),
            "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
        }


def make_processor(
    **kwargs,
) -> tuple[
    LingBotV2Processor,
    StubTokenizer,
    StubQwenImageProcessor,
]:
    tokenizer = StubTokenizer()
    image_processor = StubQwenImageProcessor()
    defaults = {
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "num_images": 2,
        "patch_vector_dim": 12,
        "max_patches_per_image": 4,
        "tokenizer_max_length": 10,
        "max_state_dim": 8,
        "action_dim": 5,
        "device": "cpu",
        "params_dtype": torch.float32,
    }
    defaults.update(kwargs)
    processor = LingBotV2Processor(**defaults)
    return processor, tokenizer, image_processor


def test_preprocess_qwen_patches_masks_text_and_state():
    processor, tokenizer, image_processor = make_processor()
    raw = {
        "images": [
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 3, 16, 16),
        ],
        "image_masks": torch.tensor([[True, False], [True, True]]),
        "task": ["pick up the cup", "open the drawer"],
        "state": torch.arange(10, dtype=torch.float32).reshape(2, 5),
    }

    result = processor.preprocess(raw)

    assert isinstance(result, LingBotV2ProcessedInputs)
    assert result.pixel_values.shape == (2, 2, 4, 12)
    assert result.image_grid_thw.tolist() == [
        [[1, 2, 2], [0, 0, 0]],
        [[1, 2, 2], [1, 1, 1]],
    ]
    assert result.image_masks.tolist() == [[True, False], [True, True]]
    assert torch.count_nonzero(result.pixel_values[0, 1]) == 0
    assert result.input_ids.shape == (2, 10)
    assert result.lang_lens.tolist() == [6, 6]
    assert result.state.shape == (2, 8)
    assert torch.count_nonzero(result.state[:, 5:]) == 0
    assert tokenizer.chat_inputs == ["pick up the cup", "open the drawer"]
    assert tokenizer.encoded_prompts == [
        "<qwen-user>pick up the cup</qwen-user>",
        "<qwen-user>open the drawer</qwen-user>",
    ]
    assert image_processor.received_dtypes == [
        torch.uint8,
        torch.uint8,
        torch.uint8,
    ]


def test_preprocess_stacked_channel_last_images_and_noise():
    processor, _, _ = make_processor()
    noise = torch.randn(1, 4, 8)
    raw = {
        "images": torch.rand(1, 2, 32, 32, 3),
        "task": "move forward",
        "state": torch.rand(1, 8),
        "noise": noise,
    }

    result = processor.preprocess(raw)

    assert result.pixel_values.shape == (1, 2, 4, 12)
    assert result.image_masks.tolist() == [[True, True]]
    assert result.noise is noise


def test_state_quantile_normalize_and_action_unnormalize():
    stats = {
        "norm_stats": {
            "observation.state": {
                "q01": [0.0, 10.0],
                "q99": [10.0, 30.0],
            },
            "action": {
                "q01": [-2.0] * 5,
                "q99": [2.0] * 5,
            },
        }
    }
    processor, _, _ = make_processor(dataset_stats=stats)
    raw = {
        "images": torch.rand(1, 2, 3, 32, 32),
        "task": "normalize",
        "state": torch.tensor([[5.0, 20.0]]),
    }

    result = processor.preprocess(raw)
    action = processor.postprocess(torch.ones(1, 4, 8))

    assert torch.allclose(result.state[:, :2], torch.zeros(1, 2))
    assert torch.count_nonzero(result.state[:, 2:]) == 0
    assert action.shape == (1, 4, 5)
    assert action.dtype == torch.float32
    assert action.device.type == "cpu"
    assert torch.allclose(action, torch.full((1, 4, 5), 2.0))


def test_legacy_prompt_format():
    processor, tokenizer, _ = make_processor(use_chat_template=False)
    result = processor.preprocess(
        {
            "images": torch.rand(1, 2, 3, 32, 32),
            "task": "close gripper",
            "state": torch.rand(1, 3),
        }
    )

    assert result.input_ids.shape == (1, 10)
    assert tokenizer.encoded_prompts == ["<bos>close gripper\n"]


def test_rejects_patch_capacity_overflow():
    processor, _, _ = make_processor(max_patches_per_image=3)

    try:
        processor.preprocess(
            {
                "images": torch.rand(1, 2, 3, 32, 32),
                "task": "overflow",
                "state": torch.rand(1, 3),
            }
        )
    except ValueError as error:
        assert "max_patches_per_image=3" in str(error)
    else:
        raise AssertionError("expected patch-capacity validation to fail")


def test_factory_returns_named_pipelines():
    tokenizer = StubTokenizer()
    image_processor = StubQwenImageProcessor()

    preprocessor, postprocessor = make_lingbot_v2_processors(
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_images=2,
        patch_vector_dim=12,
        max_patches_per_image=4,
        action_dim=5,
        device="cpu",
    )

    assert preprocessor.name == "lingbot_v2_preprocessor"
    assert postprocessor.name == "lingbot_v2_postprocessor"


def test_save_and_from_pretrained_roundtrip():
    processor, tokenizer, image_processor = make_processor()
    with TemporaryDirectory() as directory:
        processor.save_pretrained(directory)
        loaded = LingBotV2Processor.from_pretrained(
            directory,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device="cpu",
            params_dtype=torch.float32,
        )
        result = loaded.preprocess(
            {
                "images": torch.rand(1, 2, 3, 32, 32),
                "task": "round trip",
                "state": torch.rand(1, 4),
            }
        )

    assert result.pixel_values.shape == (1, 2, 4, 12)
    assert result.input_ids.shape == (1, 10)
    assert result.state.shape == (1, 8)
    assert loaded.postprocess(torch.rand(1, 3, 8)).shape == (1, 3, 5)


def test_from_pretrained_preserves_stats_for_pipeline_rebuild():
    stats = {
        "norm_stats": {
            "observation.state": {
                "q01": [0.0] * 4,
                "q99": [2.0] * 4,
            },
            "action": {
                "q01": [-1.0] * 5,
                "q99": [1.0] * 5,
            },
        }
    }
    processor, tokenizer, image_processor = make_processor(dataset_stats=stats)
    with TemporaryDirectory() as directory:
        processor.save_pretrained(directory)
        loaded = LingBotV2Processor.from_pretrained(
            directory,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device="cpu",
            params_dtype=torch.float32,
        )

    assert loaded.dataset_stats is not None
    torch.testing.assert_close(
        loaded.dataset_stats["observation.state"]["q99"],
        torch.full((4,), 2.0),
    )
    rebuilt_normalizer = next(
        step
        for step in loaded.build_preprocessor().steps
        if isinstance(step, NormalizerStep)
    )
    assert set(rebuilt_normalizer.state_dict()) == {
        "observation.state.q01",
        "observation.state.q99",
        "action.q01",
        "action.q99",
    }
