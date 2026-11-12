"""Cosmos3 T2V/T2AV model runner"""

from __future__ import annotations

from functools import partial
from collections import OrderedDict
from dataclasses import replace

import torch

from phyai.utils import get_logger
from phyai.runtime.model_runner import ModelRunner
from phyai.layers.attention.nocache.base import AttnCtx
from phyai.models.cosmos3.modeling_cosmos3 import (
    Cosmos3Condition,
    Cosmos3Transformer,
    Cosmos3GenDecoderLayer,
    Cosmos3UndDecoderLayer,
    Cosmos3AttentionPreparation,
)
from phyai.models.cosmos3.model_runner_graph_cosmos3 import Cosmos3CudaGraphRunner
from phyai.models.cosmos3.model_runner_fusion_cosmos3 import Cosmos3AttentionFusion
from phyai.models.cosmos3.model_runner_attention_cosmos3 import (
    Cosmos3AttentionContexts,
)

logger = get_logger(__name__)


class Cosmos3T2VRunner(ModelRunner):
    """Owns the per-branch UND condition memo; runs one transformer step per branch.

    ``use_fused_rope`` fuses GEN RoPE and UND/GEN KV concatenation during CUDA
    graph execution. CPU and eager execution retain the original operations.
    """

    def __init__(
        self,
        transformer: Cosmos3Transformer,
        *,
        device: torch.device | str | None = None,
        use_cuda_graph: bool = False,
        use_fused_rope: bool = True,
        fuse_eager_attention: bool = False,
        rounded_residual_norm: bool = False,
        torch_compile: bool = False,
        compile_kwargs: dict | None = None,
    ) -> None:
        self.transformer = transformer
        if device is None:
            device = next(transformer.parameters()).device
        self.device = torch.device(device)
        # Dense condition per CFG branch (keyed "cond"/"uncond"); not a KVCachePool.
        self._conditions: dict[str, Cosmos3Condition] = {}
        self._condition_specs: dict[str, tuple] = {}
        self._condition_revisions: dict[str, int] = {}
        self._next_condition_revision = 0
        self._condition_cache: OrderedDict[tuple, tuple[Cosmos3Condition, int]] = (
            OrderedDict()
        )
        self.use_cuda_graph = bool(use_cuda_graph)
        self._attention_fusion = Cosmos3AttentionFusion(
            enabled=use_fused_rope,
            eager_enabled=fuse_eager_attention and not self.use_cuda_graph,
        )
        self.rounded_residual_norm = bool(
            rounded_residual_norm and not self.use_cuda_graph
        )
        self._graphs: OrderedDict[str, Cosmos3CudaGraphRunner] = OrderedDict()
        self._attention_contexts: OrderedDict[str, Cosmos3AttentionContexts] = (
            OrderedDict()
        )
        # Optional regional torch.compile of the repeated decoder blocks, applied in
        # setup(); CUDA graphs are captured lazily after encoding each condition.
        self._torch_compile = bool(torch_compile)
        self._compile_kwargs = compile_kwargs

    def setup(self) -> None:
        """Optionally regional-``torch.compile`` the repeated decoder blocks."""
        self.invalidate_cache()
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
            logger.warning_rank0(
                "Cosmos3T2VRunner: torch.compile failed (%s); using eager.", exc
            )
            return
        logger.info_rank0(
            "Cosmos3T2VRunner: regional torch.compile applied to %d decoder "
            "blocks (%s).",
            compiled,
            kwargs,
        )

    def reset(self) -> None:
        """Clear active branches while retaining bounded condition and graph caches.

        Call once at the start of every request, including when a new prompt has
        the same token count as the previous prompt.
        """
        self._conditions.clear()
        self._condition_specs.clear()
        self._condition_revisions.clear()

    def close(self) -> None:
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        """Release all cached state after replacing weights, config, or model device."""
        self.reset()
        self._condition_cache.clear()
        for graph in self._graphs.values():
            graph.reset()
        self._graphs.clear()
        for contexts in self._attention_contexts.values():
            contexts.reset()
        self._attention_contexts.clear()
        self._attention_fusion.reset()
        # The helper is cheap when no opt-in route has ever packed a weight.
        from phyai.kernel.impls.thor_gemm import invalidate_thor_gemm

        invalidate_thor_gemm(self.transformer)

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

        The first call for each request compares text contents against at most two
        previous conditions. Later denoise steps use the active branch without a
        host synchronization. Returned condition tensors must be treated as immutable.
        """
        spec = (
            video_shape,
            fps,
            action_len,
            sound_len,
            sound_fps,
            Cosmos3CudaGraphRunner.tensor_signature(
                {"text_ids": text_ids, "text_mask": text_mask}
            ),
        )
        cond = self._conditions.get(branch)
        if cond is None or self._condition_specs.get(branch) != spec:
            parameter = next(self.transformer.parameters())
            key = (
                spec,
                parameter.dtype,
                parameter.device,
                repr(self.transformer.config),
                tuple(text_ids.detach().cpu().reshape(-1).tolist()),
                tuple(text_mask.detach().cpu().reshape(-1).tolist()),
            )
            entry = self._condition_cache.get(key)
            if entry is None:
                cond = self.transformer.encode_condition(
                    text_ids,
                    text_mask,
                    video_shape,
                    fps,
                    action_len=action_len,
                    sound_len=sound_len,
                    sound_fps=sound_fps,
                )
                self._next_condition_revision += 1
                revision = self._next_condition_revision
                self._condition_cache[key] = (cond, revision)
                if len(self._condition_cache) > 2:
                    self._condition_cache.popitem(last=False)
            else:
                cond, revision = entry
                self._condition_cache.move_to_end(key)
            self._conditions[branch] = cond
            self._condition_specs[branch] = spec
            self._condition_revisions[branch] = revision
        return cond

    def _graph_forward(
        self,
        *,
        video_shape: tuple[int, int, int],
        action_len: int,
        sound_len: int,
        num_layers: int,
        attention_contexts: tuple[AttnCtx, ...] | None,
        attention_preparation: Cosmos3AttentionPreparation | None,
        **inputs: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        condition = Cosmos3Condition(
            cached_kv=[
                (inputs[f"key_{index}"], inputs[f"value_{index}"])
                for index in range(num_layers)
            ],
            freqs_gen=(inputs["freqs_cos"], inputs["freqs_sin"]),
            video_shape=video_shape,
            action_len=action_len,
            sound_len=sound_len,
            attention_contexts=attention_contexts,
            attention_preparation=attention_preparation,
        )
        return self.transformer(
            inputs["hidden_states"],
            inputs["timestep"],
            condition,
            noisy_frame_mask=inputs.get("noisy_frame_mask"),
            action_latents=inputs.get("action_latents"),
            action_domain_id=inputs.get("action_domain_id"),
            action_noisy_mask=inputs.get("action_noisy_mask"),
            sound_latents=inputs.get("sound_latents"),
            sound_noisy_mask=inputs.get("sound_noisy_mask"),
        )

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
        # Batched FlashInfer contexts share mutable wrapper plans. Keep that
        # path eager until each graph can own an independent batched plan.
        use_cuda_graph = (
            self.use_cuda_graph
            and hidden_states.is_cuda
            and hidden_states.shape[0] == 1
        )
        attention_key = None
        if self.rounded_residual_norm and not self.use_cuda_graph:
            from phyai.models.cosmos3.model_runner_norm_cosmos3 import (
                Cosmos3ResidualNorm,
            )

            if not hasattr(self, "_residual_norm"):
                self._residual_norm = Cosmos3ResidualNorm()
            cond = replace(cond, residual_norm=self._residual_norm)
        attention_preparation = self._attention_fusion.prepare(
            device=hidden_states.device, use_cuda_graph=use_cuda_graph
        )
        if attention_preparation is not None:
            cond = replace(cond, attention_preparation=attention_preparation)
        if hidden_states.shape[0] == 1:
            contexts = self._attention_contexts.get(branch)
            if contexts is None:
                if len(self._attention_contexts) >= 2:
                    self._attention_contexts.popitem(last=False)
                contexts = Cosmos3AttentionContexts()
                self._attention_contexts[branch] = contexts
            else:
                self._attention_contexts.move_to_end(branch)
            prepared = contexts.prepare(
                self.transformer.gen_layers,
                batch_size=1,
                query_tokens=cond.freqs_gen[0].shape[1],
                text_tokens=cond.cached_kv[0][0].shape[1],
                dtype=cond.cached_kv[0][0].dtype,
                device=hidden_states.device,
                use_cuda_graph=use_cuda_graph,
            )
            cond = replace(cond, attention_contexts=prepared)
            attention_key = (contexts.signature, contexts.revision)
        if use_cuda_graph:
            graph = self._graphs.get(branch)
            if graph is None:
                if len(self._graphs) >= 2:
                    _, previous = self._graphs.popitem(last=False)
                    previous.reset()
                graph = Cosmos3CudaGraphRunner()
                self._graphs[branch] = graph
            else:
                self._graphs.move_to_end(branch)
            inputs = {"hidden_states": hidden_states, "timestep": timestep}
            for name, tensor in (
                ("noisy_frame_mask", noisy_frame_mask),
                ("action_latents", action_latents),
                ("action_domain_id", action_domain_id),
                ("action_noisy_mask", action_noisy_mask),
                ("sound_latents", sound_latents),
                ("sound_noisy_mask", sound_noisy_mask),
            ):
                if tensor is not None:
                    inputs[name] = tensor
            constants = {
                "freqs_cos": cond.freqs_gen[0],
                "freqs_sin": cond.freqs_gen[1],
            }
            for index, (key, value) in enumerate(cond.cached_kv):
                constants[f"key_{index}"] = key
                constants[f"value_{index}"] = value
            forward = partial(
                self._graph_forward,
                video_shape=cond.video_shape,
                action_len=cond.action_len,
                sound_len=cond.sound_len,
                num_layers=len(cond.cached_kv),
                attention_contexts=cond.attention_contexts,
                attention_preparation=cond.attention_preparation,
            )
            output = graph.forward(
                forward,
                inputs,
                static_inputs=constants,
                static_key=self._condition_revisions[branch],
                graph_key=(
                    cond.video_shape,
                    cond.action_len,
                    cond.sound_len,
                    attention_key,
                    attention_preparation,
                ),
            )
            # UniPC keeps previous velocities; subsequent graph replays must not
            # overwrite these tensors or the other CFG branch's output.
            if isinstance(output, tuple):
                return tuple(tensor.clone() for tensor in output)
            return output.clone()
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
