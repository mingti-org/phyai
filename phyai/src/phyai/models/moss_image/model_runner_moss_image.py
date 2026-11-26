"""Condition caching and component placement for MOSS-Image."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from phyai.models.moss_image.modeling_moss_image import (
    MOSSImageConditionSample,
    MOSSImagePreparedCondition,
    MOSSImageTransformer,
    validate_skip_layers,
)
from phyai.kernel.call import select
from phyai.layers.attention import AttnCtx, AttnLayout, AttnMetadata, AttnMode
from phyai.layers.attention.nocache.backends.flashinfer import FlashInferAttentionPlan
from phyai.parallel.state import graph_capture
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import get_logger
from phyai.utils.cuda import resolve_device

if TYPE_CHECKING:
    from phyai.models.moss_image.model_runner_text_encoder import MOSSImageTextRunner
    from phyai.models.moss_image.model_runner_vae import MOSSImageVAERunner


logger = get_logger(__name__)


def moss_image_graph_text_capacity(tokens: int) -> int | None:
    if type(tokens) is not int or tokens < 1:
        raise ValueError("Text token count must be a positive integer")
    if tokens > 4096:
        return None
    return max(64, 1 << (tokens - 1).bit_length())


def normalize_moss_image_graph_buckets(buckets) -> tuple[tuple[int, int, int], ...]:
    normalized = []
    for height, width, tokens in buckets:
        capacity = moss_image_graph_text_capacity(tokens)
        if capacity is None:
            raise ValueError("Graph warmup supports at most 4096 text tokens")
        normalized.append((height, width, capacity))
    return tuple(dict.fromkeys(normalized))


@dataclass
class MOSSImageGraphBucket:
    graph: CudaGraph
    contexts: tuple[AttnCtx, ...]
    metadata: AttnMetadata
    valid_tokens: int
    condition: MOSSImageConditionSample | None = None


class MOSSImageRunner(ModelRunner):
    def __init__(
        self,
        transformer: MOSSImageTransformer,
        *,
        text_runner: MOSSImageTextRunner | None = None,
        vae_runner: MOSSImageVAERunner | None = None,
        device: torch.device | str,
        cpu_offload: bool = False,
        torch_compile: bool = False,
        use_cuda_graph: bool = False,
        graph_cache_size: int = 4,
    ) -> None:
        self.transformer = transformer
        self.text_runner = text_runner
        self.vae_runner = vae_runner
        self.device = torch.device(device)
        self.cpu_offload = cpu_offload
        self.torch_compile = torch_compile
        if type(graph_cache_size) is not int or graph_cache_size < 1:
            raise ValueError("graph_cache_size must be a positive integer")
        self.use_cuda_graph = use_cuda_graph
        self.graph_cache_size = graph_cache_size
        self.graphs: OrderedDict[tuple, MOSSImageGraphBucket] = OrderedDict()
        self.request_graph_keys: set[tuple] = set()
        self.transformer_resident = False
        self.graph_attention_supported = True
        self.conditions: dict[str, MOSSImagePreparedCondition] = {}
        self.active_component: str | None = None
        self.denoise = transformer.forward

    @property
    def depth(self) -> int:
        return self.transformer.config.num_layers

    def components(self) -> dict[str, torch.nn.Module]:
        modules: dict[str, torch.nn.Module] = {"transformer": self.transformer}
        if self.text_runner is not None:
            modules["text_encoder"] = self.text_runner.encoder
        if self.vae_runner is not None:
            modules["vae"] = self.vae_runner.vae
        return modules

    def setup(self) -> None:
        self.clear_graphs()
        self.graph_attention_supported = True
        self.conditions.clear()
        self.active_component = None
        self.transformer_resident = False
        if self.text_runner is not None:
            self.text_runner.setup(cpu_offload=self.cpu_offload, device=self.device)
        if self.vae_runner is not None:
            self.vae_runner.setup()
        if self.cpu_offload:
            for module in self.components().values():
                module.to("cpu")
        self.validate_memory_capacity()
        if self.torch_compile:
            self.transformer.enable_compile()

    def validate_memory_capacity(self) -> None:
        if not self.cpu_offload or self.device.type != "cuda":
            return

        def tensor_bytes(module: torch.nn.Module) -> int:
            return sum(
                tensor.numel() * tensor.element_size()
                for tensor in (*module.parameters(), *module.buffers())
            )

        transformer_bytes = tensor_bytes(self.transformer)
        transient_bytes = (
            tensor_bytes(self.vae_runner.vae) if self.vae_runner is not None else 0
        )
        if self.text_runner is not None:
            encoder = self.text_runner.encoder
            transient_bytes = max(
                transient_bytes,
                *(
                    tensor_bytes(module)
                    for module in (encoder.embed_tokens, *encoder.layers, encoder.norm)
                ),
            )
        required = (
            transformer_bytes + transient_bytes
            if self.use_cuda_graph
            else max(transformer_bytes, transient_bytes)
        )
        device = resolve_device(self.device)
        free, _ = torch.cuda.mem_get_info(device)
        reusable = max(
            0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        )
        available = free + reusable
        # This lower bound excludes activations, graph pools and CUDA workspaces.
        if available < required:
            raise torch.cuda.OutOfMemoryError(
                f"Insufficient GPU memory for MOSS-Image on {device}: "
                f"{available / 1024**3:.2f} GiB available, but model weights alone need "
                f"at least {required / 1024**3:.2f} GiB with the current offload settings "
                f"(CUDA graphs={self.use_cuda_graph}). Activations and CUDA workspaces "
                "need additional memory. Select a GPU with more free memory using "
                "--devices in the serving example or CUDA_VISIBLE_DEVICES."
            )
        logger.info_rank0(
            "MOSS-Image memory on %s: %.2f GiB available; weights require at least %.2f GiB before activations and CUDA workspaces",
            device,
            available / 1024**3,
            required / 1024**3,
        )

    def activate(self, name: str) -> None:
        if not self.cpu_offload or self.active_component == name:
            return
        modules = self.components()
        if self.active_component is not None:
            if self.active_component != "transformer" or not self.use_cuda_graph:
                modules[self.active_component].to("cpu")
        if name == "transformer" and self.use_cuda_graph:
            # Captured kernels retain parameter and buffer addresses across requests.
            if not self.transformer_resident:
                modules[name].to(self.device)
                self.transformer_resident = True
        elif name != "text_encoder":
            modules[name].to(self.device)
        self.active_component = name

    def reset(self) -> None:
        self.conditions.clear()
        self.request_graph_keys.clear()
        for bucket in self.graphs.values():
            bucket.condition = None
        if self.cpu_offload and self.active_component is not None:
            if self.active_component != "transformer" or not self.use_cuda_graph:
                self.components()[self.active_component].to("cpu")
            self.active_component = None

    def clear_graphs(self) -> None:
        for bucket in self.graphs.values():
            bucket.graph.reset()
        self.graphs.clear()
        self.request_graph_keys.clear()

    def attention_context(
        self, sample: MOSSImageConditionSample, valid_tokens: int, total_tokens: int
    ) -> tuple[AttnCtx, AttnMetadata] | None:
        layer = self.transformer.layers[0].attention.attention
        selection = select(
            "attention",
            role=layer.kernel_role,
            device=sample.text.device,
            dtype={
                name: sample.text.dtype for name in ("input", "key", "value", "output")
            },
            shape={
                "tokens": total_tokens,
                "kv_tokens": total_tokens,
                "heads": layer.num_heads,
                "kv_heads": layer.num_kv_heads,
                "head_dim": layer.head_dim,
            },
            attrs={
                "layout": "ragged",
                "mask_kind": None,
                "causal": False,
                "sliding_window": None,
                "logits_soft_cap": None,
            },
            mode="capture",
            prefer=layer.prefer,
        )
        backend = selection.implementation(self)
        if backend.name != "flashinfer":
            self.graph_attention_supported = False
            logger.warning_once(
                "MOSS-Image capacity buckets require FlashInfer ragged attention; using uncaptured inference for the selected backend"
            )
            return None
        # Padding forms a second independent sequence. The sentinel guarantees
        # it stays nonempty even when the real text fills its capacity bucket.
        lengths = torch.tensor(
            [0, valid_tokens, total_tokens],
            device=sample.text.device,
            dtype=torch.int32,
        )
        meta = AttnMetadata(
            mode=AttnMode.PREFILL,
            layout=AttnLayout.RAGGED_3D,
            batch_size=2,
            num_query_tokens=total_tokens,
            cu_seqlens_q=lengths,
            cu_seqlens_kv=lengths,
            layer_proto=layer,
            q_dtype=sample.text.dtype,
            kv_dtype=sample.text.dtype,
        )
        plan = backend.init_capture_metadata(meta)
        if (
            not isinstance(plan, FlashInferAttentionPlan)
            or plan.wrapper is None
            or not plan.wrapper.is_cuda_graph_enabled
        ):
            self.graph_attention_supported = False
            logger.warning_once(
                "MOSS-Image capacity buckets need a CUDA-graph-enabled FlashInfer plan; using uncaptured inference for the selected policy"
            )
            return None
        return (
            AttnCtx(
                backend=backend,
                plan=plan,
                mode=meta.mode,
                layout=meta.layout,
                cu_seqlens_q=lengths,
                cu_seqlens_kv=lengths,
            ),
            meta,
        )

    def captured_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        condition: MOSSImagePreparedCondition,
        skip_layers: tuple[int, ...],
    ) -> torch.Tensor:
        timestep = torch.as_tensor(
            timestep, device=latents.device, dtype=torch.float32
        ).reshape(-1)
        if timestep.numel() not in (1, latents.shape[0]):
            raise ValueError("timestep must be scalar or have one value per sample")
        if not self.graph_attention_supported:
            return self.denoise(
                latents, timestep, condition=condition, skip_layers=skip_layers
            )
        # Visit long samples first so shorter ones can reuse a newly captured
        # bucket, while the per-sample padding bound still limits wasted rows.
        order = sorted(
            range(len(condition.samples)),
            key=lambda index: condition.samples[index].text.shape[1],
            reverse=True,
        )
        outputs: dict[int, torch.Tensor] = {}
        for index in order:
            sample = condition.samples[index]
            capacity = moss_image_graph_text_capacity(sample.text.shape[1])
            latent = latents[index : index + 1]
            time = timestep if timestep.numel() == 1 else timestep[index : index + 1]
            if capacity is None or not self.graph_attention_supported:
                outputs[index] = self.denoise(
                    latent,
                    time,
                    condition=MOSSImagePreparedCondition(
                        (sample,), condition.height, condition.width
                    ),
                    skip_layers=skip_layers,
                )
            else:
                outputs[index] = self.captured_sample_forward(
                    latent, time, sample, capacity, skip_layers
                )
        return (
            outputs[0]
            if len(outputs) == 1
            else torch.cat([outputs[index] for index in range(len(outputs))], dim=0)
        )

    def captured_sample_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        sample: MOSSImageConditionSample,
        capacity: int,
        skip_layers: tuple[int, ...],
    ) -> torch.Tensor:
        height, width = latents.shape[-2:]
        image_tokens = height * width
        valid_tokens = image_tokens + sample.text.shape[1]
        key = (
            tuple(latents.shape),
            latents.dtype,
            (1,),
            capacity,
            skip_layers,
        )
        bucket = self.graphs.get(key)
        if bucket is None:
            # A warm larger bucket is preferable to a new capture when the
            # extra token rows cost at most 25% over the preferred capacity.
            reusable = [
                candidate
                for candidate in self.graphs
                if candidate[:3] == key[:3]
                and candidate[4:] == key[4:]
                and candidate[3] >= capacity
                and 4 * (image_tokens + candidate[3] + 1)
                <= 5 * (image_tokens + capacity + 1)
            ]
            if reusable:
                key = min(reusable, key=lambda candidate: candidate[3])
                bucket = self.graphs[key]
                capacity = key[3]
        condition = MOSSImagePreparedCondition((sample,), height, width)
        if bucket is None:
            if len(self.graphs) >= self.graph_cache_size:
                evict = next(
                    (
                        cached
                        for cached in self.graphs
                        if cached not in self.request_graph_keys
                    ),
                    None,
                )
                if evict is None:
                    logger.warning_once(
                        "MOSS-Image graph cache cannot hold all branches of this request; increase graph_cache_size to capture the remaining shapes"
                    )
                    return self.denoise(
                        latents, timestep, condition=condition, skip_layers=skip_layers
                    )
                self.graphs.pop(evict).graph.reset()
            total_tokens = image_tokens + capacity + 1
            attention = self.attention_context(sample, valid_tokens, total_tokens)
            if attention is None:
                return self.denoise(
                    latents, timestep, condition=condition, skip_layers=skip_layers
                )
            ctx, metadata = attention
            contexts = (ctx,)
            text = sample.text.new_zeros(1, capacity + 1, sample.text.shape[-1])
            cos = sample.cos.new_ones(1, total_tokens, sample.cos.shape[-1])
            sin = sample.sin.new_zeros(1, total_tokens, sample.sin.shape[-1])
            text[:, : sample.text.shape[1]].copy_(sample.text)
            cos[:, :valid_tokens].copy_(sample.cos)
            sin[:, :valid_tokens].copy_(sample.sin)
            inputs = {
                "latents": latents,
                "timestep": timestep,
                "text": text,
                "cos": cos,
                "sin": sin,
            }

            def forward(latents, timestep, text, cos, sin):
                prepared = MOSSImagePreparedCondition(
                    (MOSSImageConditionSample(text, cos, sin),),
                    height,
                    width,
                )
                return self.denoise(
                    latents,
                    timestep,
                    condition=prepared,
                    skip_layers=skip_layers,
                    attention_contexts=contexts,
                )

            graph = CudaGraph()
            try:
                with graph_capture():
                    graph.capture(forward, inputs)
            except BaseException:
                graph.reset()
                raise
            bucket = MOSSImageGraphBucket(
                graph, contexts, metadata, valid_tokens, sample
            )
            self.graphs[key] = bucket
            logger.info_rank0("Captured MOSS-Image CUDA graph bucket %s", key)
        self.graphs.move_to_end(key)
        self.request_graph_keys.add(key)
        if bucket.condition is not sample:
            if bucket.valid_tokens != valid_tokens:
                lengths = torch.tensor(
                    [0, valid_tokens, image_tokens + capacity + 1],
                    dtype=torch.int32,
                    device="cpu",
                )
                bucket.metadata.cu_seqlens_q.copy_(lengths)
                # Re-plan outside capture on the same wrapper and fixed buffers.
                # CPU indptr avoids a device-to-host synchronization in plan().
                metadata = replace(
                    bucket.metadata, cu_seqlens_q=lengths, cu_seqlens_kv=lengths
                )
                bucket.contexts[0].backend.init_forward_metadata(metadata)
                bucket.valid_tokens = valid_tokens
            for name, fill, length in (
                ("text", 0, sample.text.shape[1]),
                ("cos", 1, valid_tokens),
                ("sin", 0, valid_tokens),
            ):
                buffer = bucket.graph.input_buffer(name)
                buffer.fill_(fill)
                buffer[:, :length].copy_(getattr(sample, name))
            bucket.condition = sample
        output = bucket.graph.replay({"latents": latents, "timestep": timestep})
        # CFG branches and the Engine IPC transport must not alias graph output storage.
        return output.clone()

    @torch.no_grad()
    def warmup_graphs(self, buckets: tuple[tuple[int, int, int], ...]) -> None:
        if not buckets:
            return
        if not self.use_cuda_graph or self.device.type != "cuda":
            raise ValueError("Graph warmup requires CUDA graphs on a CUDA device")
        buckets = normalize_moss_image_graph_buckets(buckets)
        if len(set(buckets)) * 2 > self.graph_cache_size:
            raise ValueError(
                "graph_cache_size must hold both CFG branches of each warmup bucket"
            )
        try:
            for height, width, tokens in buckets:
                if (
                    height <= 0
                    or width <= 0
                    or height % 16
                    or width % 16
                    or not 1 <= tokens <= 4096
                ):
                    raise ValueError(
                        "Warmup buckets require image sizes divisible by 16 and 1..4096 text tokens"
                    )
                embeddings = torch.zeros(
                    1,
                    tokens,
                    self.transformer.config.text_dim,
                    device=self.device,
                    dtype=self.transformer.video_embedder.weight.dtype,
                )
                mask = torch.ones(1, tokens, device=self.device, dtype=torch.bool)
                self.prepare_condition(
                    "warmup", embeddings, mask, height // 16, width // 16
                )
                latents = torch.zeros(
                    1,
                    self.transformer.config.latent_channels,
                    height // 16,
                    width // 16,
                    device=self.device,
                    dtype=torch.float32,
                )
                timestep = torch.ones((), device=self.device)
                for skipped in ((), (22,)):
                    self.forward(
                        latents, timestep, condition="warmup", skip_layers=skipped
                    )
        finally:
            self.reset()

    @torch.no_grad()
    def encode(
        self, input_ids: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_runner is None:
            raise ValueError("String prompts require the text encoder")
        self.activate("text_encoder")
        return self.text_runner.encode(input_ids)

    @torch.no_grad()
    def prepare_condition(
        self,
        name: str,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        height: int,
        width: int,
    ) -> None:
        self.activate("transformer")
        self.conditions[name] = self.transformer.prepare_condition(
            embeddings, mask, height, width
        )

    @torch.no_grad()
    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        *,
        condition: str = "positive",
        skip_layers: tuple[int, ...] = (),
    ) -> torch.Tensor:
        self.activate("transformer")
        if condition not in self.conditions:
            raise RuntimeError(f"Condition {condition!r} was not prepared")
        if self.use_cuda_graph and latents.is_cuda:
            prepared = self.conditions[condition]
            if (
                latents.ndim != 4
                or latents.shape[1] != self.transformer.config.latent_channels
                or latents.shape[0] < 1
            ):
                raise ValueError("latents must have shape [B, latent_channels, H, W]")
            if len(prepared.samples) != latents.shape[0] or (
                prepared.height,
                prepared.width,
            ) != tuple(latents.shape[-2:]):
                raise ValueError(
                    "Prepared condition batch and latent dimensions must match"
                )
            return self.captured_forward(
                latents,
                timestep,
                prepared,
                tuple(sorted(validate_skip_layers(skip_layers, self.depth))),
            )
        return self.denoise(
            latents,
            timestep,
            condition=self.conditions[condition],
            skip_layers=skip_layers,
        )

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae_runner is None:
            raise ValueError("Decoded outputs require the FLUX.2 VAE")
        self.activate("vae")
        return self.vae_runner.decode(latents)

    def close(self) -> None:
        self.clear_graphs()
        self.reset()
        if self.cpu_offload and self.transformer_resident:
            self.transformer.to("cpu")
            self.transformer_resident = False
        if self.text_runner is not None:
            self.text_runner.close()
        if self.vae_runner is not None:
            self.vae_runner.close()


__all__ = ["MOSSImageRunner"]
