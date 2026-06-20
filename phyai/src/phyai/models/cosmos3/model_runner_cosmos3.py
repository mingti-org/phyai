"""Cosmos3 T2V/T2AV model runner"""

from __future__ import annotations

import logging

import torch

from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Condition,
    Cosmos3GenDecoderLayer,
    Cosmos3Transformer,
    Cosmos3UndDecoderLayer,
)
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import this_rank_log


logger = logging.getLogger(__name__)


class Cosmos3T2VRunner(ModelRunner):
    """Owns the per-branch UND condition memo; runs one transformer step per branch."""

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        device: torch.device | str | None = None,
        torch_compile: bool = False,
        compile_kwargs: dict | None = None,
    ) -> None:
        self.transformer = transformer
        if device is None:
            device = next(transformer.parameters()).device
        self.device = torch.device(device)
        # Dense condition per CFG branch (keyed "cond"/"uncond"); not a KVCachePool.
        self._conditions: dict[str, Cosmos3Condition] = {}
        # Optional regional torch.compile of the repeated decoder blocks, applied in
        # setup(); off by default. No CUDA-graph capture (see setup()).
        self._torch_compile = bool(torch_compile)
        self._compile_kwargs = compile_kwargs

    def setup(self) -> None:
        """Optionally regional-``torch.compile`` the repeated decoder blocks."""
        if not self._torch_compile:
            return
        kwargs = (
            {"dynamic": True}
            if self._compile_kwargs is None
            else dict(self._compile_kwargs)
        )
        compiled = 0
        try:
            for module in self.transformer.modules():
                if isinstance(module, (Cosmos3GenDecoderLayer, Cosmos3UndDecoderLayer)):
                    module.compile(**kwargs)
                    compiled += 1
        except Exception as exc:  # pragma: no cover - compile is best-effort
            this_rank_log(
                logger,
                logging.WARNING,
                "Cosmos3T2VRunner: torch.compile failed (%s); using eager.",
                exc,
            )
            return
        this_rank_log(
            logger,
            logging.INFO,
            "Cosmos3T2VRunner: regional torch.compile applied to %d decoder blocks (%s).",
            compiled,
            kwargs,
        )

    def reset(self) -> None:
        """Drop the cached per-branch conditions (call once at the start of a request)."""
        self._conditions.clear()

    def condition(
        self,
        branch: str,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float | None = None,
        *,
        action_len: int = 0,
        sound_len: int = 0,
        sound_fps: float | None = None,
    ) -> Cosmos3Condition:
        """Return ``branch``'s UND condition, encoding it via the transformer on first use.

        The condition is timestep-independent, so the first denoise step for a branch
        pays the UND-tower cost and every later step is a cache hit.
        """
        cond = self._conditions.get(branch)
        if cond is None:
            cond = self.transformer.encode_condition(
                text_ids,
                text_mask,
                video_shape,
                fps,
                action_len=action_len,
                sound_len=sound_len,
                sound_fps=sound_fps,
            )
            self._conditions[branch] = cond
        return cond

    @torch.no_grad()
    def forward(
        self,
        branch: str,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float | None = None,
        noisy_frame_mask: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        action_domain_id: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        sound_latents: torch.Tensor | None = None,
        sound_noisy_mask: torch.Tensor | None = None,
        sound_fps: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """One transformer forward for ``branch``, reusing its cached UND condition.

        Returns the video velocity, or ``(video_velocity, aux_velocity)`` when an
        action / sound stream is supplied (see
        :meth:`Cosmos3Transformer.forward`). The auxiliary token lengths are derived
        from the latents so the condition's GEN positions match.
        """
        action_len = action_latents.shape[1] if action_latents is not None else 0
        sound_len = sound_latents.shape[1] if sound_latents is not None else 0
        cond = self.condition(
            branch,
            text_ids,
            text_mask,
            video_shape,
            fps,
            action_len=action_len,
            sound_len=sound_len,
            sound_fps=sound_fps,
        )
        return self.transformer(
            hidden_states,
            timestep,
            cond,
            noisy_frame_mask=noisy_frame_mask,
            action_latents=action_latents,
            action_domain_id=action_domain_id,
            action_noisy_mask=action_noisy_mask,
            sound_latents=sound_latents,
            sound_noisy_mask=sound_noisy_mask,
        )


__all__ = ["Cosmos3T2VRunner"]
