"""Local checkpoint tokenization and single-image U0 processing."""

from __future__ import annotations

import base64
import re
from pathlib import Path
import unicodedata

import torch

# The pattern and special-token ordering follow the released BAAI UNIS tokenizer.
# Copyright 2025 BAAI. and/or its affiliates. SPDX-License-Identifier: Apache-2.0
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
TEMPLATE = "<|extra_203|>You are a helpful assistant for t2i task. USER: {text} ASSISTANT: <|extra_100|>"
X2I_PREFIX = "<|extra_203|>You are a helpful assistant. USER: "
X2I_SUFFIX = " ASSISTANT: <|extra_100|>"

# Scene Gen prompt from Xiaomi-Robotics-U0's Apache-2.0 inference configs.
SCENE_SYSTEM_PROMPT = (
    "You are an advanced controllable image generation model specialized in synthesizing "
    "high-quality robotic vision data. You will receive a scene description of the robot "
    "workspace. Your task is to generate the high-quality initial observation containing "
    "3 views with one full size view on top and two half size views side by side at the "
    "bottom. You should keep the consistency across different views. Image Style: Real."
)


class U0TextProcessor:
    """Read local vocabularies without executing checkpoint Python code."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        """Load local vocabulary files and validate the released U0 token layout."""
        import tiktoken

        root = Path(checkpoint_dir)
        ranks = {}
        for line in (root / "unis.tiktoken").read_bytes().splitlines():
            if line.strip():
                token, rank = line.split()
                ranks[base64.b64decode(token)] = int(rank)
        vision = [
            line.strip()
            for line in (root / "unis_vision_tokens.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        names = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        names += [f"<|extra_{index}|>" for index in range(205)] + vision
        if len(set(names)) != len(names):
            raise ValueError("Duplicate checkpoint special tokens.")
        self.special_tokens = {name: 151643 + index for index, name in enumerate(names)}
        self.encoding = tiktoken.Encoding(
            "Xiaomi-Robotics-U0",
            pat_str=PATTERN,
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
        )
        if (
            len(ranks) + len(names) != self.encoding.n_vocab
            or self.encoding.n_vocab != 282926
        ):
            raise ValueError("Unsupported U0 vocabulary.")
        required = {
            "<|extra_203|>": 151849,
            "<|extra_204|>": 151850,
            "<|extra_200|>": 151846,
            "<|extra_101|>": 151747,
            "<|image start|>": 151852,
            "<|image end|>": 151853,
            "<|image token|>": 151851,
        }
        if any(
            self.special_tokens.get(name) != value for name, value in required.items()
        ):
            raise ValueError("Unsupported U0 special-token IDs.")
        inverse = {value: name for name, value in self.special_tokens.items()}
        for index in range(131072):
            name = inverse.get(151854 + index, "")
            match = re.fullmatch(r"<\|visual token (\d+)\|>", name)
            if match is None or int(match.group(1)) != index:
                raise ValueError(
                    f"Unexpected visual token at ID {151854 + index}: {name!r}"
                )

    def encode(self, text: str) -> list[int]:
        """Encode with the released tokenizer's NFC and special-token policy."""
        return self.encoding.encode(
            unicodedata.normalize("NFC", text),
            allowed_special="all",
            disallowed_special=(),
        )

    def _validate_prompt(self, prompt: str) -> None:
        """Reject empty user text and reserved control tokens."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty text prompt is required.")
        if any(name in prompt for name in self.special_tokens):
            raise ValueError("Prompt contains a reserved U0 token.")

    def build_inputs(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Build unpadded conditional and no-text CFG prompts on the CPU."""
        self._validate_prompt(prompt)
        conditional = torch.tensor(
            [self.encode(TEMPLATE.format(text=prompt))], dtype=torch.int64
        )
        unconditional = torch.tensor(
            [self.encode(TEMPLATE.format(text=""))], dtype=torch.int64
        )
        return conditional, unconditional

    def build_x2i_inputs(
        self, prompt: str, codes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build X2I CFG prompts that share one serialized reference image."""
        self._validate_prompt(prompt)
        if codes.ndim != 2 or codes.dtype != torch.int64 or not codes.numel():
            raise ValueError("Expected a nonempty two-dimensional int64 code grid.")
        if ((codes < 0) | (codes >= 131072)).any().item():
            raise ValueError("Reference codes must be in [0, 131072).")
        height, width = codes.shape
        prefix = self.encode(X2I_PREFIX)
        prefix += [self.special_tokens["<|image start|>"]]
        prefix += self.encode(f"{height}*{width}")
        prefix += [self.special_tokens["<|image token|>"]]
        rows = (codes.detach().cpu() + 151854).tolist()
        for index, row in enumerate(rows):
            if index:
                prefix.append(self.special_tokens["<|extra_200|>"])
            prefix.extend(row)
        prefix.append(self.special_tokens["<|image end|>"])
        conditional = prefix + self.encode(prompt + X2I_SUFFIX)
        unconditional = prefix + self.encode(X2I_SUFFIX)
        return (
            torch.tensor([conditional], dtype=torch.int64),
            torch.tensor([unconditional], dtype=torch.int64),
        )

    def header_mapping(self) -> dict[str, list[int]]:
        """Derive the scheduler's dimension-header mappings from the vocabulary."""
        return {
            text: self.encode(text)
            for text in [*(str(i) for i in range(10)), "*", "64*64"]
        }

    def build_scene_gen_inputs(
        self, prompt: str, robot_arm_type: str = "AgiBot G1"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build Scene Gen CFG prompts with a shared workspace instruction."""
        self._validate_prompt(prompt)
        self._validate_prompt(robot_arm_type)
        prefix = f"<|extra_203|>{SCENE_SYSTEM_PROMPT}"
        scene = f" Robot Arm Type: {robot_arm_type}. Scene Description: {prompt}"
        suffix = " <|extra_100|>"
        return (
            torch.tensor([self.encode(prefix + scene + suffix)], dtype=torch.int64),
            torch.tensor([self.encode(prefix + suffix)], dtype=torch.int64),
        )


def reference_image_pixels(path: str | Path) -> torch.Tensor:
    """Resize one reference to the official area and return normalized RGB pixels."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        aspect_ratio = image.width / image.height
        height = int((1024 * 1024 / aspect_ratio) ** 0.5)
        width = int(height * aspect_ratio)
        height = ((height + 8) // 16) * 16
        width = ((width + 8) // 16) * 16
        if not height or not width:
            raise ValueError("Reference aspect ratio is too extreme for the U0 grid.")
        rgb = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    pixels = torch.tensor(np.asarray(rgb) / 127.5 - 1.0, dtype=torch.float32)
    return pixels.permute(2, 0, 1).unsqueeze(0)


def visual_codes(tokens: torch.Tensor) -> torch.Tensor:
    """Validate a complete generated-only 64-by-64 image and return codes."""
    if tokens.ndim != 1 or tokens.dtype != torch.int64:
        raise ValueError("Generated tokens must be a one-dimensional int64 tensor.")
    ids = tokens.detach().cpu()
    if ids.numel() != 4169:
        raise ValueError("Expected a complete 4169-token single image.")
    if ids[:7].tolist() != [151852, 21, 19, 9, 21, 19, 151851]:
        raise ValueError("Unexpected fixed-grid image header.")
    if ids[-3:].tolist() != [151853, 151747, 151850]:
        raise ValueError("Expected EOI, ESS and EOS termination.")
    return _visual_grid(ids[7:-3], 64, 64)


def scene_visual_codes(tokens: torch.Tensor) -> torch.Tensor:
    """Extract the complete Scene Gen image and its model-selected grid."""
    if tokens.ndim != 1 or tokens.dtype != torch.int64:
        raise ValueError("Generated tokens must be a one-dimensional int64 tensor.")
    ids = tokens.detach().cpu()
    starts = (ids == 151852).nonzero().flatten()
    if starts.numel() != 1 or int(ids[-1]) != 151853:
        raise ValueError("Expected one complete Scene Gen image ending at EOI.")
    start = int(starts[0])
    # A released dimension header has at most three digits on either side of '*'.
    header = ids[start + 1 : start + 9].tolist()
    if 151851 not in header:
        raise ValueError("Missing Scene Gen dimension header.")
    end = header.index(151851)
    symbols = {15 + digit: str(digit) for digit in range(10)} | {9: "*"}
    shape = "".join(symbols.get(token, "?") for token in header[:end])
    match = re.fullmatch(r"([1-9][0-9]{0,2})\*([1-9][0-9]{0,2})", shape)
    if match is None:
        raise ValueError("Invalid Scene Gen dimension header.")
    height, width = map(int, match.groups())
    return _visual_grid(ids[start + end + 2 : -1], height, width)


def _visual_grid(ids: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if ids.numel() != height * (width + 1) - 1:
        raise ValueError("Visual token count does not match the image grid.")
    rows = []
    for row in range(height):
        start = row * (width + 1)
        values = ids[start : start + width]
        if not ((values >= 151854) & (values < 282926)).all().item():
            raise ValueError(f"Invalid visual code at row {row}.")
        if row < height - 1 and int(ids[start + width]) != 151846:
            raise ValueError(f"Missing EOL at row {row}.")
        rows.append(values - 151854)
    return torch.stack(rows)


def image_from_pixels(pixels: torch.Tensor):
    """Convert one decoded RGB tensor in [-1, 1] to an 8-bit PIL image."""
    import numpy as np
    from PIL import Image

    if (
        pixels.ndim != 4
        or pixels.shape[:2] != (1, 3)
        or not pixels.numel()
        or not torch.isfinite(pixels).all().item()
    ):
        raise ValueError("Expected one nonempty, finite RGB image.")
    pixels = pixels.detach().float().cpu()[0].permute(1, 2, 0)
    pixels = ((pixels + 1.0) * 127.5).clamp(0, 255)
    return Image.fromarray(pixels.numpy().astype(np.uint8))
