"""Single-sequence U0 generation over independent conditional KV pools."""

from __future__ import annotations

from typing import Any

import torch

from phyai.layers.attention.enums import AttnLayout, AttnMode
from phyai.layers.attention.paged.base import PagedAttnMetadata
from phyai.models.xiaomi_u0.logits_processor_unis import (
    UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor,
)
from phyai.models.xiaomi_u0.model_runner_unis import UNISForwardBatch, UNISModelRunner
from phyai.models.xiaomi_u0.modeling_unis_ar import UNISForCausalLM
from phyai.models.xiaomi_u0.requests_unis import (
    UNISGenerationRequest,
    UNISGenerationResult,
)
from phyai.runtime.schedule import Scheduler
from phyai.utils import get_logger

logger = get_logger(__name__)


class _HeaderTokenizer:
    """Only the tokenizer operations used by the fixed 64-by-64 processor."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        expected = {
            **{str(index): [15 + index] for index in range(10)},
            "*": [9],
            "64*64": [21, 19, 9, 21, 19],
        }
        for key, values in expected.items():
            actual = mapping.get(key)
            if (
                not isinstance(actual, list)
                or any(type(value) is not int for value in actual)
                or actual != values
            ):
                raise ValueError(f"Unsupported U0 header mapping for {key!r}.")
        self.mapping = {key: values.copy() for key, values in expected.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("The U0 header does not add special tokens.")
        return self.mapping[text].copy()

    def decode(self, tokens: list[int] | torch.Tensor) -> str:
        values = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
        matches = [key for key, ids in self.mapping.items() if ids == values]
        if len(matches) != 1:
            raise ValueError("Header tokens do not have a unique mapping.")
        return matches[0]


def _seed_request(device: torch.device, seed: int) -> None:
    """Seed only the generators covered by the request's fork_rng context."""
    if device.type != "cuda" or device.index is None:
        raise ValueError("U0 sampling requires an explicit CUDA device index.")
    torch.random.default_generator.manual_seed(seed)
    torch.cuda.default_generators[device.index].manual_seed(seed)


def _metadata(start: int, end: int, device: torch.device) -> PagedAttnMetadata:
    def ints(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, device=device, dtype=torch.int32)

    return PagedAttnMetadata(
        mode=AttnMode.PREFILL if start == 0 else AttnMode.DECODE,
        layout=AttnLayout.RAGGED_3D,
        batch_size=1,
        num_query_tokens=end - start,
        cu_seqlens_q=ints([0, end - start]),
        cu_seqlens_kv=ints([0, end]),
        seq_lens_kv=ints([end]),
        paged_kv_indptr=ints([0, end]),
        paged_kv_indices=torch.arange(end, device=device, dtype=torch.int32),
        paged_kv_last_page_len=ints([1]),
        write_indices=torch.arange(start, end, device=device, dtype=torch.int64),
        position_ids=torch.arange(start, end, device=device, dtype=torch.int64),
    )


class _Output(dict):
    @property
    def logits(self) -> torch.Tensor:
        return self["logits"]


class _Branch:
    """Bridge the official processor to a runner-owned cache."""

    def __init__(self, runner: UNISModelRunner) -> None:
        self.runner = runner
        self.history: torch.Tensor | None = None
        self.last_logits: torch.Tensor | None = None

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        start = 0 if self.history is None else self.history.shape[1]
        self.history = (
            ids.clone() if start == 0 else torch.cat((self.history, ids), dim=1)
        )
        end = self.history.shape[1]
        self.runner.plan_inference(_metadata(start, end, ids.device))
        positions = torch.arange(start, end, device=ids.device).unsqueeze(0)
        self.last_logits = self.runner.forward(UNISForwardBatch(ids, positions))[
            :, -1
        ].clone()
        return self.last_logits

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: Any,
    ) -> _Output:
        if not use_cache or not torch.all(attention_mask == 1).item():
            raise ValueError("U0 supports unpadded cached unconditional input only.")
        if self.history is None:
            if past_key_values is not None:
                raise ValueError("Unexpected cache handle before the first forward.")
        elif input_ids.shape[1] != 1 or past_key_values is not self:
            raise ValueError(
                "Unconditional decode requires one token and its branch handle."
            )
        logits = self.forward(input_ids)
        if attention_mask.shape[1] != self.history.shape[1]:
            raise ValueError("Unconditional mask length does not match the cache.")
        return _Output(logits=logits.unsqueeze(1), past_key_values=self)


class UNISGenerationScheduler(Scheduler):
    """Own CFG branches, processing, sampling and EOS termination.

    The first implementation supports one unpadded, tokenized T2I request,
    a fixed 64-by-64 visual grid and no_text CFG. Image decode is separate.
    """

    def __init__(
        self,
        model: UNISForCausalLM,
        *,
        capacity: int,
        tokenizer_mapping: dict[str, list[int]],
        paged_backend: str = "torch_reference",
    ) -> None:
        self.model = model
        self.capacity = capacity
        self.tokenizer = _HeaderTokenizer(tokenizer_mapping)
        self.runners = [
            UNISModelRunner(model, capacity=capacity, paged_backend=paged_backend)
            for _ in range(2)
        ]
        self._ready = False

    def setup(self) -> None:
        if self._ready:
            raise RuntimeError("The U0 scheduler is already set up.")
        try:
            for runner in self.runners:
                runner.setup()
        except Exception:
            self.close()
            raise
        self._ready = True

    @torch.inference_mode()
    def step(self, request: UNISGenerationRequest) -> UNISGenerationResult:
        if not self._ready:
            raise RuntimeError("Call setup before U0 generation.")
        if not isinstance(request, UNISGenerationRequest):
            raise TypeError("Expected UNISGenerationRequest.")
        for ids in (request.input_ids, request.unconditional_ids):
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
                raise ValueError("U0 expects one nonempty unpadded token sequence.")
            if ids.dtype != torch.int64:
                raise ValueError("Input token IDs must be int64.")
            if (ids < 0).any().item() or (
                ids >= self.model.lm_head.out_features
            ).any().item():
                raise ValueError("Input tokens must be within the U0 vocabulary.")
        if request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        longest = max(request.input_ids.shape[1], request.unconditional_ids.shape[1])
        if longest + request.max_new_tokens > self.capacity:
            raise ValueError("The request exceeds the configured cache capacity.")
        replay = request.replay_ids
        if replay is not None:
            if replay.ndim != 1 or replay.dtype != torch.int64:
                raise ValueError(
                    "Regression replay IDs must be a one-dimensional int64 tensor."
                )
            if replay.numel() < request.max_new_tokens:
                raise ValueError(
                    "Replay must provide every requested prediction token."
                )
            if (replay < 0).any().item() or (
                replay >= self.model.lm_head.out_features
            ).any().item():
                raise ValueError("Replay tokens must be within the U0 vocabulary.")
        if any(
            step < 0 or step >= request.max_new_tokens for step in request.capture_steps
        ):
            raise ValueError("Capture indices must lie within the generation limit.")
        for runner in self.runners:
            runner.reset()
        device = self.runners[0].device
        conditional, unconditional = (_Branch(runner) for runner in self.runners)
        processor = UnbatchedClassifierFreeGuidanceLogitsForVisualTokenWithDifferentialTopKProcessor(
            guidance_scale=3.0,
            model=unconditional,
            tokenizer=self.tokenizer,
            unconditional_ids=request.unconditional_ids.to(device),
            use_cache=True,
            unconditional_type="no_text",
            target_height=64,
            target_width=64,
            force_same_image_size=True,
            image_cfg_scale=1.0,
            use_differential_sampling=True,
            text_top_k=1024,
            text_top_p=0.9,
            text_temperature=1.0,
            image_top_k=5120,
            image_top_p=1.0,
            image_temperature=1.0,
            cfg_decay="none",
        )
        generated: list[int] = []
        traces: list[dict[str, Any]] = []
        reason = "length"
        query = request.input_ids.to(device)
        with torch.random.fork_rng(devices=[device.index]):
            _seed_request(device, request.seed)
            for step in range(request.max_new_tokens):
                raw = conditional.forward(query)
                scores = processor(conditional.history, raw.clone())
                if (
                    torch.isnan(scores).any().item()
                    or not torch.isfinite(scores).any().item()
                ):
                    raise RuntimeError(
                        "The U0 processor produced invalid sampling scores."
                    )
                if step in request.capture_steps:
                    traces.append(
                        {
                            "step": step,
                            "conditional_logits": raw.detach().cpu(),
                            "unconditional_logits": (
                                None
                                if unconditional.last_logits is None
                                else unconditional.last_logits.detach().cpu()
                            ),
                            "processed_scores": scores.detach().cpu().clone(),
                        }
                    )
                if replay is None:
                    cutoff = scores.topk(131072, dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < cutoff, -torch.inf)
                    query = torch.multinomial(scores.softmax(-1), num_samples=1)
                else:
                    # Teacher forcing diagnoses numerical correctness; it is not sampling.
                    query = replay[step : step + 1].to(device).reshape(1, 1)
                token = int(query.item())
                generated.append(token)
                if step % 64 == 0:
                    logger.info("U0 generation tokens=%d", len(generated))
                if token == 151850:
                    reason = "eos"
                    break
        return UNISGenerationResult(
            generated_ids=torch.tensor(generated, dtype=torch.int64),
            stop_reason=reason,
            replay=replay is not None,
            conditional_cached_tokens=conditional.history.shape[1],
            unconditional_cached_tokens=(
                0 if unconditional.history is None else unconditional.history.shape[1]
            ),
            traces=tuple(traces),
        )

    def close(self) -> None:
        for runner in self.runners:
            runner.close()
        self._ready = False


__all__ = ["UNISGenerationScheduler"]
