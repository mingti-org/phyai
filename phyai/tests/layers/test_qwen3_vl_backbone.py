"""Shared Qwen3-VL backbone geometry and compatibility contracts."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from phyai.layers.backbones.qwen3_vl import Qwen3VLConfig, Qwen3VLModel
from phyai.layers.transformer_block import TransformerBlock
from phyai.layers.backbones.qwen3_vl.modeling import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)
from phyai.models.qwen3_vl import Qwen3VLModel as LegacyQwen3VLModel


def test_backbone_config_defaults_and_legacy_import():
    config = Qwen3VLConfig()
    assert config.vision.out_hidden_size == config.text.hidden_size
    assert LegacyQwen3VLModel is Qwen3VLModel
    loaded = Qwen3VLConfig.from_dict(
        {
            "text_config": {
                "hidden_size": 256,
                "head_dim": 64,
                "rope_scaling": {"mrope_section": [12, 10, 10]},
            },
            "vision_config": {"out_hidden_size": 256},
        }
    )
    assert loaded.text.mrope_section == (12, 10, 10)
    assert loaded.text.hidden_size == loaded.vision.out_hidden_size == 256


def test_frame_boundaries_and_merged_patch_order():
    grid = torch.tensor([[1, 4, 4], [2, 2, 4]])
    torch.testing.assert_close(
        get_vision_cu_seqlens(grid), torch.tensor([0, 16, 24, 32], dtype=torch.int32)
    )
    positions = get_vision_position_ids(grid, 2)
    assert positions.shape == (32, 2)
    expected = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    torch.testing.assert_close(positions[:4], expected)
    torch.testing.assert_close(positions[16:24], positions[24:32])


def test_learned_position_interpolation_matches_bilinear_grid():
    grid = torch.tensor([[1, 4, 6]])
    indices, weights = get_vision_bilinear_indices_and_weights(grid, 3, 2)
    table = torch.arange(18, dtype=torch.float32).view(9, 2)
    actual = (table[indices] * weights[..., None]).sum(0)
    spatial = table.view(3, 3, 2).permute(2, 0, 1).unsqueeze(0)
    expected_grid = F.interpolate(spatial, (4, 6), mode="bilinear", align_corners=True)
    positions = get_vision_position_ids(grid, 2)
    expected = expected_grid[0, :, positions[:, 0], positions[:, 1]].transpose(0, 1)
    torch.testing.assert_close(actual, expected)


def test_explicit_cpu_device_reaches_every_backbone_parameter(fake_mesh):
    fake_mesh()
    config = Qwen3VLConfig.from_dict(
        {
            "text_config": {
                "vocab_size": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "mrope_section": [4, 2, 2],
                "max_position_embeddings": 32,
            },
            "vision_config": {
                "out_hidden_size": 64,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_heads": 4,
                "depth": 1,
                "deepstack_visual_indexes": [0],
                "num_position_embeddings": 16,
                "patch_size": 2,
            },
        }
    )
    model = Qwen3VLModel(config, device="cpu")
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(buffer.device.type == "cpu" for buffer in model.buffers())
    block = TransformerBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        attn_qk_norm=True,
        sandwich_norm=True,
        device="cpu",
    )
    assert all(parameter.device.type == "cpu" for parameter in block.parameters())
