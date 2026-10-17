"""Local checkpoint tokenization and single-image U0 postprocessing."""

from __future__ import annotations

import base64
import re
from pathlib import Path
import unicodedata
from typing import Any

import torch

# The pattern and special-token ordering follow the released BAAI UNIS tokenizer.
# Copyright 2025 BAAI. and/or its affiliates. SPDX-License-Identifier: Apache-2.0
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
TEMPLATE = "<|extra_203|>You are a helpful assistant for t2i task. USER: {text} ASSISTANT: <|extra_100|>"


class U0TextProcessor:
    """Read local vocabularies without executing checkpoint Python code."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
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

    def build_inputs(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Build unpadded conditional and no-text CFG prompts on the CPU."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty text prompt is required.")
        # User content must not inject control tokens into the fixed T2I template.
        if any(name in prompt for name in self.special_tokens):
            raise ValueError("Prompt contains a reserved U0 token.")
        conditional = torch.tensor(
            [self.encode(TEMPLATE.format(text=prompt))], dtype=torch.int64
        )
        unconditional = torch.tensor(
            [self.encode(TEMPLATE.format(text=""))], dtype=torch.int64
        )
        return conditional, unconditional

    def header_mapping(self) -> dict[str, list[int]]:
        """Derive the fixed-grid scheduler mapping from the real vocabulary."""
        return {
            text: self.encode(text)
            for text in [*(str(i) for i in range(10)), "*", "64*64"]
        }


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
    rows = []
    for row in range(64):
        start = 7 + row * 65
        values = ids[start : start + 64]
        if not ((values >= 151854) & (values < 282926)).all().item():
            raise ValueError(f"Invalid visual code at row {row}.")
        if row < 63 and int(ids[start + 64]) != 151846:
            raise ValueError(f"Missing EOL at row {row}.")
        rows.append(values - 151854)
    return torch.stack(rows)


def decode_image(tokens: torch.Tensor, vision_model: Any) -> Any:
    """Decode validated codebook IDs using the released FP32 VisionTokenizer."""
    import numpy as np
    from PIL import Image

    codes = visual_codes(tokens)
    device = next(vision_model.parameters()).device
    with torch.inference_mode():
        decoded = vision_model.decode_code(
            codes.to(device)[None], shape=(1, 64, 64, 256)
        ).float()
        if (
            decoded.shape != (1, 3, 1024, 1024)
            or not torch.isfinite(decoded).all().item()
        ):
            raise ValueError("VisionTokenizer returned an invalid image.")
        pixels = ((decoded[0].permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255)
        return Image.fromarray(pixels.cpu().numpy().astype(np.uint8))
