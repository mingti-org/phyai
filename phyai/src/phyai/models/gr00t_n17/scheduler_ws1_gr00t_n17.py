"""GR00T-N1.7 single-GPU scheduler boundary.

The scheduler consumes prepared tensors from
``phyai_utils_tools.models.gr00t.GR00TProcessor`` and returns the normalized
action chunk. Image transform, tokenization, and decode stay outside the engine.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging

import torch

from phyai.models.gr00t_n17.model_runner_gr00t_n17 import (
    GR00TN17ActionHeadRunner,
    GR00TN17BackboneRunner,
)
from phyai.models.gr00t_n17.modeling_gr00t_n17 import (
    GR00TN17ActionInput,
    GR00TN17Model,
)
from phyai.runtime.schedule import Scheduler
from phyai.utils import this_rank_log


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GR00TN17Request:
    """One GR00T-N1.7 inference request — prepared model-input tensors.

    ``tensors`` is exactly the dict produced by the out-of-engine processor
    (``GR00TProcessor.process_observation(obs).tensors``): ``state`` /
    ``embodiment_id`` / ``action_mask`` plus the VLM tensors (``pixel_values``,
    ``input_ids``, ``attention_mask``, ``image_grid_thw``).
    ``action_mask`` is optional but, when present, is applied to the normalized
    action chunk before it is returned.
    ``noise`` optionally seeds the flow-matching sampler for deterministic runs.
    """

    tensors: dict[str, torch.Tensor]
    noise: torch.Tensor | None = None


class GR00TN17WS1Scheduler(Scheduler):
    """Single-rank GR00T-N1.7 inference scheduler.

    The scheduler owns runner order and the denoising request lifecycle. It does
    not own preprocessing: it receives prepared tensors and emits the masked
    normalized action chunk.
    """

    def __init__(
        self,
        model: GR00TN17Model,
        *,
        max_batch_size: int = 1,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = True,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}.")
        self.model = model
        self.model.eval()
        self.max_batch_size = int(max_batch_size)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        use_cuda_graph = bool(use_cuda_graph)
        self.backbone_runner = GR00TN17BackboneRunner(
            model,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
        )
        self.action_head_runner = GR00TN17ActionHeadRunner(
            model,
            max_batch_size=self.max_batch_size,
            device=self.device,
            use_cuda_graph=use_cuda_graph,
        )
        self.use_cuda_graph = (
            self.backbone_runner.use_cuda_graph
            and self.action_head_runner.use_cuda_graph
        )
        if (
            self.backbone_runner.use_cuda_graph
            != self.action_head_runner.use_cuda_graph
        ):
            this_rank_log(
                logger,
                logging.WARNING,
                "GR00T-N1.7 CUDA Graphs require both Backbone and Action Head "
                "runners to support capture; disabling CUDA Graphs for both.",
            )
        self.backbone_runner.use_cuda_graph = self.use_cuda_graph
        self.action_head_runner.use_cuda_graph = self.use_cuda_graph

    def setup(
        self,
        *,
        capture_profiles: Sequence[GR00TN17Request] = (),
    ) -> None:
        self.backbone_runner.setup()
        self.action_head_runner.setup()
        self.capture_profiles(capture_profiles)

    def _fixed_noise_profile(
        self,
        request: GR00TN17Request,
    ) -> GR00TN17Request:
        state = request.tensors.get("state")
        if not isinstance(state, torch.Tensor) or state.ndim == 0:
            raise ValueError("A capture profile requires a batched state tensor.")
        action_head = self.model.action_head
        if request.noise is None:
            noise = torch.zeros(
                state.shape[0],
                action_head.action_horizon,
                action_head.action_dim,
                dtype=self.model.params_dtype,
                device=self.device,
            )
        else:
            noise = request.noise.to(
                device=self.device,
                dtype=self.model.params_dtype,
            )
        return GR00TN17Request(tensors=request.tensors, noise=noise)

    def _capture_profile_key(
        self,
        request: GR00TN17Request,
    ) -> tuple[object, ...] | None:
        if request.noise is None:
            raise ValueError("Capture profile key generation requires fixed noise.")
        backbone_inputs, action_inputs = self._prepare_request(request)
        backbone_key = self.backbone_runner.capture_profile_key(backbone_inputs)
        if backbone_key is None:
            return None
        action_key = self.action_head_runner.capture_profile_key(
            backbone_key,
            action_inputs,
            request.noise,
        )
        if action_key is None:
            return None
        return (("backbone", backbone_key), ("action_head", action_key))

    @torch.no_grad()
    def capture_profiles(
        self,
        profiles: Sequence[GR00TN17Request],
    ) -> None:
        """Stabilize and capture the supplied request profiles during setup."""
        profiles = tuple(profiles)
        for index, profile in enumerate(profiles):
            if not isinstance(profile, GR00TN17Request):
                raise TypeError(
                    f"capture_profiles[{index}] must be GR00TN17Request, got "
                    f"{type(profile).__name__}."
                )
        if not self.use_cuda_graph:
            return
        if not profiles:
            raise ValueError(
                "GR00T-N1.7 fixed CUDA Graph mode requires at least one "
                "capture profile during scheduler setup."
            )

        capture_requests = tuple(self._fixed_noise_profile(p) for p in profiles)
        unique_requests: list[tuple[int, GR00TN17Request]] = []
        seen_keys: set[tuple[object, ...]] = set()
        for index, request in enumerate(capture_requests):
            key = self._capture_profile_key(request)
            if key is None:
                raise ValueError(
                    f"capture_profiles[{index}] is not supported by the current "
                    "GR00T-N1.7 CUDA Graph path."
                )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_requests.append((index, request))
        if len(unique_requests) != len(capture_requests):
            this_rank_log(
                logger,
                logging.INFO,
                "Deduplicated %d GR00T-N1.7 capture profiles to %d unique "
                "Graph structures.",
                len(capture_requests),
                len(unique_requests),
            )

        for _, request in unique_requests:
            self._run_setup_profile(request, capture=False)
        for index, request in unique_requests:
            self._run_setup_profile(request, capture=True)
            if (
                self.backbone_runner.last_forward_used_eager_warmup
                or self.action_head_runner.last_forward_used_eager_warmup
            ):
                raise ValueError(
                    f"capture_profiles[{index}] is not supported by the current "
                    "GR00T-N1.7 CUDA Graph path."
                )
        torch.cuda.current_stream(self.device).synchronize()

    def _has_mixed_attention_masks(
        self,
        tensors: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> bool:
        """Whether batch rows carry different valid-token layouts."""
        if batch_size <= 1:
            return False
        attention_mask = tensors["attention_mask"].bool()
        input_ids = tensors["input_ids"]
        if attention_mask.ndim != 2 or input_ids.ndim != 2:
            raise ValueError(
                "GR00T-N1.7 input_ids and attention_mask must be 2-D, got "
                f"{tuple(input_ids.shape)} and {tuple(attention_mask.shape)}."
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "GR00T-N1.7 input_ids and attention_mask must have matching "
                f"shapes, got {tuple(input_ids.shape)} and "
                f"{tuple(attention_mask.shape)}."
            )
        if attention_mask.shape[0] != batch_size:
            raise ValueError(
                "GR00T-N1.7 attention_mask batch does not match state batch: "
                f"{attention_mask.shape[0]} != {batch_size}."
            )
        return not torch.equal(
            attention_mask,
            attention_mask[:1].expand_as(attention_mask),
        )

    def _compact_valid_tokens(
        self,
        tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Left-align valid tokens and right-pad shorter rows."""
        attention_mask = tensors["attention_mask"].bool()
        if attention_mask.ndim != 2:
            raise ValueError(
                "GR00T-N1.7 attention_mask must be 2-D, got "
                f"{tuple(attention_mask.shape)}."
            )
        batch_size, seq_len = attention_mask.shape
        valid_lengths = attention_mask.sum(dim=1, dtype=torch.int64)
        max_valid = int(valid_lengths.max().item())
        if max_valid <= 0:
            raise ValueError("attention_mask has no valid tokens.")

        positions = torch.arange(seq_len, device=attention_mask.device).expand(
            batch_size, -1
        )
        sort_keys = torch.where(attention_mask, positions, positions + seq_len)
        source_indices = sort_keys.argsort(dim=1)[:, :max_valid]
        target_valid = torch.arange(max_valid, device=attention_mask.device).unsqueeze(
            0
        ) < valid_lengths.unsqueeze(1)

        compacted = dict(tensors)
        pad_token_id = int(
            getattr(
                self.model.backbone.qwen3vl_model.config,
                "pad_token_id",
                0,
            )
            or 0
        )

        def gather_sequence(value: torch.Tensor, *, pad_value: int) -> torch.Tensor:
            gather_indices = source_indices.to(value.device)
            gathered = value.gather(1, gather_indices)
            valid = target_valid.to(value.device)
            return torch.where(valid, gathered, gathered.new_full((), pad_value))

        for key in (
            "input_ids",
            "token_type_ids",
            "mm_token_type_ids",
        ):
            value = compacted.get(key)
            if (
                value is not None
                and value.ndim == 2
                and tuple(value.shape) == (batch_size, seq_len)
            ):
                compacted[key] = gather_sequence(
                    value,
                    pad_value=pad_token_id if key == "input_ids" else 0,
                )
        original_attention_mask = tensors["attention_mask"]
        compacted["attention_mask"] = target_valid.to(
            device=original_attention_mask.device,
            dtype=original_attention_mask.dtype,
        )

        position_ids = compacted.get("position_ids")
        if (
            position_ids is not None
            and position_ids.ndim == 3
            and tuple(position_ids.shape[1:]) == (batch_size, seq_len)
        ):
            gather_indices = (
                source_indices.to(position_ids.device)
                .unsqueeze(0)
                .expand(position_ids.shape[0], -1, -1)
            )
            gathered = position_ids.gather(2, gather_indices)
            compacted["position_ids"] = torch.where(
                target_valid.to(position_ids.device).unsqueeze(0),
                gathered,
                torch.zeros((), dtype=position_ids.dtype, device=position_ids.device),
            )
        return compacted

    def _prepare_request(
        self,
        request: GR00TN17Request,
    ) -> tuple[dict[str, torch.Tensor], GR00TN17ActionInput]:
        tensors = request.tensors
        batch = tensors["state"].shape[0]
        if batch > self.max_batch_size:
            raise ValueError(
                "GR00T-N1.7 request batch exceeds scheduler max_batch_size: "
                f"{batch} > {self.max_batch_size}."
            )
        if batch > 1 and self._has_mixed_attention_masks(tensors, batch_size=batch):
            tensors = self._compact_valid_tokens(tensors)

        return self.model.prepare_input(tensors, device=self.device)

    def _run_setup_profile(
        self,
        request: GR00TN17Request,
        *,
        capture: bool,
    ) -> torch.Tensor:
        """Run one setup-only eager warmup or CUDA Graph capture profile."""
        backbone_inputs, action_inputs = self._prepare_request(request)
        force_eager_warmup = not capture
        backbone_output = self.backbone_runner.forward(
            backbone_inputs,
            force_eager_warmup=force_eager_warmup,
            allow_cuda_graph_capture=capture,
        )
        backbone_warmed = self.backbone_runner.last_forward_used_eager_warmup
        return self.action_head_runner.forward(
            backbone_output,
            action_inputs,
            noise=request.noise,
            force_eager_warmup=force_eager_warmup or backbone_warmed,
            allow_cuda_graph_capture=capture,
        )

    @torch.no_grad()
    def step(self, request: GR00TN17Request) -> torch.Tensor:
        """Run backbone + action-head denoising; return the normalized action.

        Returns the ``(B, action_horizon, max_action_dim)`` normalized action
        chunk after applying any request ``action_mask``. The caller maps it
        back to physical units via the processor's ``decode_action`` (which
        needs the per-request ``raw_state``).
        """
        backbone_inputs, action_inputs = self._prepare_request(request)
        backbone_output = self.backbone_runner.forward(backbone_inputs)
        return self.action_head_runner.forward(
            backbone_output,
            action_inputs,
            noise=request.noise,
        )

    def close(self) -> None:
        self.backbone_runner.close()
        self.action_head_runner.close()
        return None


__all__ = ["GR00TN17Request", "GR00TN17WS1Scheduler"]
