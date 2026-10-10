"""Request-local KV and action-conditioning state for MolmoAct2."""

from collections import OrderedDict

import torch

from phyai.layers.attention import AttnCtx, AttnLayout, AttnMask, AttnMode
from phyai.layers.attention.nocache.backends.sdpa import (
    SdpaAttentionBackend,
    SdpaAttentionPlan,
)
from phyai.parallel.state import graph_capture
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner
from phyai.models.molmoact2.policy_molmoact2 import (
    MolmoAct2Output,
    MolmoAct2ForConditionalGeneration,
)
from phyai.models.molmoact2.modeling_action_expert import (
    ActionExpertContext,
    ActionExpertStepModulation,
)


class MolmoAct2Runner(ModelRunner):
    def __init__(
        self,
        model: MolmoAct2ForConditionalGeneration,
        *,
        use_cuda_graph: bool = False,
        max_cuda_graphs: int = 2,
    ) -> None:
        if max_cuda_graphs < 1:
            raise ValueError("max_cuda_graphs must be positive.")
        self.model = model
        self.use_cuda_graph = bool(use_cuda_graph)
        self.max_cuda_graphs = max_cuda_graphs
        self.past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = (
            None
        )
        self.context: ActionExpertContext | None = None
        self.modulations: tuple[ActionExpertStepModulation, ...] = ()
        self.graph_modulations: tuple[torch.Tensor, ...] = ()
        self.graphs: OrderedDict[tuple, CudaGraph] = OrderedDict()
        self.action_graph: CudaGraph | None = None
        self.graph_capture_count = 0
        self.graph_replay_count = 0
        # SDPA's fixed dense masks allow values to change without KV packing.
        self.graph_attention_backend = SdpaAttentionBackend()
        self.graph_attention_plan = SdpaAttentionPlan()

    @property
    def device(self) -> torch.device:
        return self.model.lm_head.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model.lm_head.weight.dtype

    def setup(self) -> None:
        self.model.eval()

    def reset(self) -> None:
        self.past_key_values = None
        self.context = None
        self.modulations = ()
        self.graph_modulations = ()
        self.action_graph = None

    @staticmethod
    def pack_modulation(modulation: ActionExpertStepModulation) -> torch.Tensor:
        return torch.stack(
            (
                modulation.conditioning,
                *(value for block in modulation.block_modulations for value in block),
                *modulation.final_modulation,
            ),
            dim=1,
        )

    def capture_action_forward(
        self,
        actions: torch.Tensor,
        modulation: torch.Tensor,
        kv_contexts: torch.Tensor,
        cross_key_mask: torch.Tensor | None = None,
        self_key_mask: torch.Tensor | None = None,
        valid_action: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expert = self.model.model.action_expert
        if expert is None:
            raise RuntimeError("This checkpoint has no continuous action expert.")
        cross_mask = (
            None if cross_key_mask is None else AttnMask.from_key_mask(cross_key_mask)
        )
        self_mask = (
            None if self_key_mask is None else AttnMask.from_key_mask(self_key_mask)
        )
        rope_cache = None
        if rope_cos is not None and rope_sin is not None:
            rope_cache = (rope_cos, rope_sin)
        elif rope_cos is not None or rope_sin is not None:
            raise ValueError("Action graphs require both RoPE cosine and sine inputs.")
        context = ActionExpertContext(
            kv_contexts=tuple((layer[0], layer[1]) for layer in kv_contexts),
            cross_mask=cross_mask,
            self_mask=self_mask,
            valid_action=valid_action,
            rope_cache=rope_cache,
            self_attn_ctx=AttnCtx(
                backend=self.graph_attention_backend,
                plan=self.graph_attention_plan,
                mode=AttnMode.PREFILL,
                layout=AttnLayout.PADDED_4D,
                mask=self_mask,
            ),
            cross_attn_ctx=AttnCtx(
                backend=self.graph_attention_backend,
                plan=self.graph_attention_plan,
                mode=AttnMode.PREFILL,
                layout=AttnLayout.PADDED_4D,
                mask=cross_mask,
            ),
        )
        step_modulation = ActionExpertStepModulation(
            conditioning=modulation[:, 0],
            block_modulations=tuple(
                tuple(modulation[:, 1 + 9 * index + chunk] for chunk in range(9))
                for index in range(len(expert.blocks))
            ),
            final_modulation=(modulation[:, -2], modulation[:, -1]),
        )
        return expert.forward_with_context(
            actions,
            step_modulation.conditioning,
            context=context,
            modulation=step_modulation,
        )

    @torch.inference_mode()
    def prepare_action_graph(self, *, action_horizon: int) -> None:
        context = self.context
        if context is None or not self.modulations:
            raise RuntimeError("Prepare action conditioning before graph capture.")
        self.graph_modulations = tuple(
            self.pack_modulation(modulation) for modulation in self.modulations
        )
        context_inputs = {
            "kv_contexts": torch.stack(
                tuple(torch.stack(pair) for pair in context.kv_contexts)
            ),
        }
        for name, mask in (
            ("cross_key_mask", context.cross_mask),
            ("self_key_mask", context.self_mask),
        ):
            if mask is not None:
                if mask.key_mask is None or mask.segments is not None:
                    raise ValueError("Action graphs require per-token validity masks.")
                context_inputs[name] = mask.key_mask
        if context.valid_action is not None:
            context_inputs["valid_action"] = context.valid_action
        if context.rope_cache is not None:
            context_inputs["rope_cos"], context_inputs["rope_sin"] = context.rope_cache
        examples = {
            "actions": self.graph_modulations[0].new_zeros(
                self.graph_modulations[0].shape[0],
                action_horizon,
                self.model.config.max_action_dim,
            ),
            "modulation": self.graph_modulations[0],
            **context_inputs,
        }
        key = tuple(
            (name, tuple(tensor.shape), tensor.dtype, tensor.device)
            for name, tensor in examples.items()
        )
        graph = self.graphs.get(key)
        if graph is None:
            if len(self.graphs) >= self.max_cuda_graphs:
                _, evicted = self.graphs.popitem(last=False)
                evicted.reset()
            # Bind capture-mode kernels before entering CUDA stream capture.
            graph = CudaGraph()
            with torch.cuda.device(self.device):
                with graph_capture():
                    self.capture_action_forward(**examples)
                graph.capture(self.capture_action_forward, examples)
            self.graphs[key] = graph
            self.graph_capture_count += 1
        else:
            self.graphs.move_to_end(key)
            # Every request refreshes all conditioning buffers before replay.
            for name, tensor in context_inputs.items():
                graph.input_buffer(name).copy_(tensor)
        self.action_graph = graph

    def encoder_attention_mask(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        ids = inputs["input_ids"]
        source = inputs.get("attention_mask")
        mask = (ids != -1) if source is None else source.to(torch.bool).clone()
        if mask.ndim != 2:
            raise ValueError(
                "Action conditioning requires a two-dimensional padding mask."
            )
        config = self.model.config
        if config.action_mode == "both":
            mask &= ids != config.eos_token_id
            for row_ids, row_mask in zip(ids, mask):
                starts = (
                    (row_ids == config.action_start_token_id)
                    .nonzero()
                    .flatten()
                    .tolist()
                )
                ends = (
                    (row_ids == config.action_end_token_id).nonzero().flatten().tolist()
                )
                end_index = 0
                for start in starts:
                    while end_index < len(ends) and ends[end_index] < start:
                        end_index += 1
                    end = ends[end_index] + 1 if end_index < len(ends) else ids.shape[1]
                    row_mask[start:end] = False
                    end_index += 1
        return mask

    @torch.inference_mode()
    def prefill(
        self, inputs: dict[str, torch.Tensor], *, output_logits: bool = False
    ) -> MolmoAct2Output:
        if output_logits:
            output = self.model(**inputs, logits_to_keep=1)
        else:
            output = self.model.model(**inputs)
        self.past_key_values = output.past_key_values
        return output

    @torch.inference_mode()
    def prepare_action(
        self, inputs: dict[str, torch.Tensor], *, action_horizon: int, num_steps: int
    ) -> None:
        self.reset()
        expert = self.model.model.action_expert
        if expert is None:
            raise ValueError("This checkpoint has no continuous action expert.")
        encoder_mask = self.encoder_attention_mask(inputs)
        if not bool(encoder_mask.any(dim=-1).all()):
            raise ValueError(
                "Each sample needs at least one visible action-conditioning token."
            )
        self.prefill(inputs)
        batch_size = inputs["input_ids"].shape[0]
        kv_states = tuple(
            (
                key.transpose(1, 2).reshape(batch_size, key.shape[2], -1),
                value.transpose(1, 2).reshape(batch_size, value.shape[2], -1),
            )
            for key, value in self.past_key_values
        )
        self.context = expert.prepare_context(
            encoder_kv_states=kv_states,
            encoder_attention_mask=encoder_mask,
            batch_size=batch_size,
            seq_len=action_horizon,
            device=self.device,
            dtype=self.dtype,
        )
        timesteps = tuple(
            torch.full(
                (batch_size,),
                index / num_steps,
                device=self.device,
                dtype=torch.float32,
            )
            for index in range(num_steps)
        )
        self.modulations = tuple(expert.prepare_modulation_cache(timesteps))
        if self.use_cuda_graph and self.device.type == "cuda":
            self.prepare_action_graph(action_horizon=action_horizon)

    @torch.inference_mode()
    def forward(self, batch: torch.Tensor, *, step_index: int) -> torch.Tensor:
        expert = self.model.model.action_expert
        if self.context is None or expert is None:
            raise RuntimeError("Call prepare_action before evaluating action velocity.")
        if self.action_graph is not None:
            with torch.cuda.device(self.device):
                out = self.action_graph.replay(
                    {"actions": batch, "modulation": self.graph_modulations[step_index]}
                )
            self.graph_replay_count += 1
            return out.clone()
        modulation = self.modulations[step_index]
        return expert.forward_with_context(
            batch,
            modulation.conditioning,
            context=self.context,
            modulation=modulation,
        )

    @torch.inference_mode()
    def decode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
    ) -> MolmoAct2Output:
        if self.past_key_values is None:
            raise RuntimeError("Call prefill before decoding tokens.")
        output = self.model(
            token_ids,
            past_key_values=self.past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logits_to_keep=1,
        )
        self.past_key_values = output.past_key_values
        return output

    def close(self) -> None:
        self.reset()
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()
