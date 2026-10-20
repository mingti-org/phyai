"""Per-prompt execution and padding for the MOSS text encoder."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch.utils.hooks import RemovableHandle

from phyai.models.moss_image.text_encoder import MOSSImageTextEncoder
from phyai.runtime.model_runner import ModelRunner


class MOSSImageTextRunner(ModelRunner):
    def __init__(
        self,
        encoder: MOSSImageTextEncoder,
        *,
        cache_size: int = 16,
        cache_max_bytes: int = 256 * 1024**2,
    ) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (cache_size, cache_max_bytes)
        ):
            raise ValueError("Text cache limits must be nonnegative integers")
        self.encoder = encoder
        self.execution_device: torch.device | None = None
        self.offload_hooks: list[RemovableHandle] = []
        self.offload_parameters: dict[
            torch.nn.Module,
            list[tuple[torch.nn.Module, str, torch.Tensor, dict[str, object]]],
        ] = {}
        self.offload_buffers: dict[
            torch.nn.Module,
            list[tuple[torch.nn.Module, str, torch.Tensor | None]],
        ] = {}
        self.cache_size = cache_size
        self.cache_max_bytes = cache_max_bytes
        self.cache_bytes = 0
        self.feature_cache: OrderedDict[
            tuple[tuple[int, ...], torch.dtype], torch.Tensor
        ] = OrderedDict()

    def clear_cache(self) -> None:
        """Invalidate prompt features after replacing or updating encoder weights."""
        self.feature_cache.clear()
        self.cache_bytes = 0

    def cache_feature(
        self, key: tuple[tuple[int, ...], torch.dtype], feature: torch.Tensor
    ) -> None:
        size = feature.numel() * feature.element_size()
        if not self.cache_size or size > self.cache_max_bytes:
            return
        while self.feature_cache and (
            len(self.feature_cache) >= self.cache_size
            or self.cache_bytes + size > self.cache_max_bytes
        ):
            _, evicted = self.feature_cache.popitem(last=False)
            self.cache_bytes -= evicted.numel() * evicted.element_size()
        self.feature_cache[key] = feature.detach().to(device="cpu", copy=True)
        self.cache_bytes += size

    def setup(
        self, *, cpu_offload: bool = False, device: torch.device | str | None = None
    ) -> None:
        self.close()
        self.encoder.eval()
        if cpu_offload:
            if device is None:
                raise ValueError("Text offload requires an execution device")
            self.execution_device = torch.device(device)
            self.encoder.to("cpu")
            modules = (
                self.encoder.embed_tokens,
                *self.encoder.layers,
                self.encoder.norm,
            )
            for module in modules:
                self.offload_hooks.append(
                    module.register_forward_pre_hook(self.load_module)
                )
                self.offload_hooks.append(
                    module.register_forward_hook(self.offload_module, always_call=True)
                )

    def load_module(self, module: torch.nn.Module, inputs: tuple) -> None:
        # Retain the CPU weights: inference leaves them unchanged, so returning
        # a layer to CPU only needs to restore these storage references.
        parameters = []
        buffers = []
        for owner in module.modules():
            parameters.extend(
                (owner, name, parameter.detach(), vars(parameter).copy())
                for name, parameter in owner._parameters.items()
                if parameter is not None
            )
            buffers.extend(
                (owner, name, value) for name, value in owner._buffers.items()
            )
        self.offload_parameters[module] = parameters
        self.offload_buffers[module] = buffers
        try:
            module.to(self.execution_device)
        except BaseException:
            self.restore_module(module)
            raise

    def restore_module(self, module: torch.nn.Module) -> None:
        for owner, name, value, metadata in self.offload_parameters.pop(module, ()):
            parameter = owner._parameters[name]
            parameter.data = value
            vars(parameter).update(metadata)
        for owner, name, value in self.offload_buffers.pop(module, ()):
            owner._buffers[name] = value

    def offload_module(
        self, module: torch.nn.Module, inputs: tuple, output: object
    ) -> None:
        self.restore_module(module)

    @torch.inference_mode()
    def encode(
        self, input_ids: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not input_ids:
            raise ValueError("At least one tokenized prompt is required")
        device = self.execution_device or self.encoder.embed_tokens.weight.device
        features = []
        for ids in input_ids:
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                raise ValueError(
                    "Each prompt must contain input_ids with shape [1, sequence]"
                )
            key = None
            if self.cache_size and self.cache_max_bytes:
                token_ids = tuple(
                    ids.to(device="cpu", dtype=torch.long).reshape(-1).tolist()
                )
                key = (token_ids, self.encoder.embed_tokens.weight.dtype)
                cached = self.feature_cache.get(key)
                if cached is not None:
                    self.feature_cache.move_to_end(key)
                    features.append(cached.to(device=device))
                    continue
            feature = self.encoder(ids.to(device=device, dtype=torch.long))[0]
            if key is not None:
                self.cache_feature(key, feature)
            features.append(feature)
        max_length = max(feature.shape[0] for feature in features)
        embeddings = features[0].new_zeros(
            len(features), max_length, self.encoder.config.hidden_size
        )
        mask = torch.zeros((len(features), max_length), dtype=torch.bool, device=device)
        for index, feature in enumerate(features):
            embeddings[index, : feature.shape[0]] = feature
            mask[index, : feature.shape[0]] = True
        return embeddings, mask

    def forward(self, batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(batch)

    def close(self) -> None:
        self.clear_cache()
        for hook in self.offload_hooks:
            hook.remove()
        self.offload_hooks.clear()
        for module in tuple(self.offload_parameters):
            self.restore_module(module)
        if self.execution_device is not None:
            self.encoder.to("cpu")
        self.execution_device = None


__all__ = ["MOSSImageTextRunner"]
