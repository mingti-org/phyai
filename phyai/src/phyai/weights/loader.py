"""Top-level safetensors -> model loader.

The whole load chain in one place:

1. Walk ``model.named_parameters()``, collect ``param.hf_keys`` and
   ``param.weight_loader`` into a dispatch index keyed by HF tensor
   name. Params without ``hf_keys`` are skipped (tied weights, RoPE
   buffers, etc.).
2. Resolve ``source`` to a concrete list of safetensors shards: a
   checkpoint folder is expanded via
   :func:`phyai.utils.checkpoint.find_safetensors` (honouring
   ``model.safetensors.index.json``); a single file path becomes
   ``[path]``; an iterable is consumed as-is.
3. Open every shard lazily; for each key, optionally remap via
   ``remap`` (callable or dict), look up in the index, and dispatch.
4. Track every key seen, every cast, every miss; build a
   :class:`LoadReport`. Strict mode raises if anything required is
   missing or any HF key was unexpected.
5. Walk ``model.modules()``; call ``module.post_load()`` where defined
   so quant specs can do scale fixups (e.g. fp8 per-tensor ->
   per-channel fan-out).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from tqdm.auto import tqdm

from phyai.utils.checkpoint import find_safetensors, resolve_checkpoint
from phyai.utils.logging import this_rank_log
from phyai.weights.shards import WeightLoader, replicated


_logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    """Outcome of a :func:`load_pretrained` call.

    Attributes
    ----------
    loaded : list of HF keys successfully copied into a phyai param.
    missing : HF keys claimed by some param's plan but absent in the
        checkpoint, where the source was *required*.
    optional_missing : same but for params marked ``optional=True``
        (typically quant scales on a non-quant checkpoint).
    unexpected : HF keys present in the checkpoint that no param
        claimed.
    casts : ``(hf_key, src_dtype, dst_dtype)`` triples — the dtype
        differed and ``copy_`` did an implicit cast.
    """

    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    optional_missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    casts: list[tuple[str, torch.dtype, torch.dtype]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"loaded={len(self.loaded)}",
            f"missing={len(self.missing)}",
            f"optional_missing={len(self.optional_missing)}",
            f"unexpected={len(self.unexpected)}",
            f"casts={len(self.casts)}",
        ]
        if self.missing:
            lines.append(
                f"  missing keys: {self.missing[:5]}{'...' if len(self.missing) > 5 else ''}"
            )
        if self.unexpected:
            lines.append(
                f"  unexpected keys: {self.unexpected[:5]}{'...' if len(self.unexpected) > 5 else ''}"
            )
        return " | ".join(lines)


def _resolve_remap(
    remap: Callable[[str], str | None] | dict[str, str] | None,
) -> Callable[[str], str | None]:
    """Normalise the ``remap`` argument to a single callable.

    A dict is treated as a substring rewrite map: each (src, dst) pair
    means "if `src` appears in the key, replace it with `dst`". Multiple
    matching pairs apply in iteration order.
    """
    if remap is None:
        return lambda k: k
    if callable(remap):
        return remap
    if isinstance(remap, dict):
        rules = list(remap.items())

        def apply_rules(key: str) -> str | None:
            for src, dst in rules:
                if src in key:
                    key = key.replace(src, dst)
            return key

        return apply_rules
    raise TypeError(
        f"remap must be callable, dict, or None; got {type(remap).__name__}"
    )


def _resolve_source(
    source: str | Path | Iterable[str | Path],
    *,
    revision: str | None = None,
) -> list[Path]:
    """Normalise ``source`` to a concrete list of safetensors file paths.

    Accepts three shapes:

    * a checkpoint folder or a HuggingFace repo id (``str``/``Path``) —
      resolved to a local folder via
      :func:`phyai.utils.checkpoint.resolve_checkpoint` (a repo id is
      downloaded; ``revision`` selects the branch/tag/commit), then
      expanded via :func:`phyai.utils.checkpoint.find_safetensors`,
    * a single safetensors file path (``str``/``Path`` pointing at a
      file) — wrapped as ``[path]``,
    * an iterable of file paths — materialised as a list (always treated
      as already-local; no repo-id download for this form).
    """
    if isinstance(source, (str, Path)):
        resolved = resolve_checkpoint(source, revision=revision)
        if resolved.is_dir():
            return find_safetensors(resolved)
        return [resolved]
    return [Path(p) for p in source]


def _source_label(source: str | Path | Iterable[str | Path]) -> str:
    """A short human label for the progress bar (folder / file name)."""
    if isinstance(source, (str, Path)):
        return Path(source).name or str(source)
    return "weights"


def _count_keys(paths: list[Path]) -> int:
    """Sum the tensor-key count across shards (header-only, cheap)."""
    total = 0
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            total += len(f.keys())
    return total


def _progress_disable(progress: bool | None) -> bool | None:
    """Resolve the tqdm ``disable`` flag for a load.

    Only rank 0 ever renders a bar. ``progress=None`` (auto) defers to
    tqdm's own TTY detection, so piped / CI / captured (pytest) runs stay
    silent while interactive terminals and notebooks show the bar.
    ``True`` forces it on (rank 0, even off-TTY); ``False`` disables it.
    """
    rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    if progress is False or not rank0:
        return True
    if progress is True:
        return False
    return None


def load_pretrained(
    model: nn.Module,
    source: str | Path | Iterable[str | Path],
    *,
    remap: Callable[[str], str | None] | dict[str, str] | None = None,
    strict: bool = True,
    progress: bool | None = None,
    revision: str | None = None,
) -> LoadReport:
    """Load HF safetensors checkpoints into ``model``.

    Parameters
    ----------
    model : the model to fill. Each parameter that should load must
        have ``param.hf_keys`` and ``param.weight_loader`` attached
        (the standard parallel-Linear classes do this in their
        ``__init__``).
    source : one of —

        * a checkpoint **folder** (single ``str``/``Path``) —
          ``model.safetensors.index.json`` is consumed if present,
          otherwise ``model.safetensors`` / glob fallback;
        * a HuggingFace **repo id** (single ``str``/``Path`` that is not a
          local path) — the repo is downloaded via
          :func:`huggingface_hub.snapshot_download` and loaded from the
          local cache;
        * a single safetensors **file** path; or
        * an iterable of safetensors file paths (advanced / test) —
          always treated as already-local (no repo-id download).

    remap : optional HF-key rewriter. If callable, called with each
        file key; return the lookup key, or ``None`` to drop the key.
        If a dict, treated as substring rewrite rules applied in
        iteration order. The plan keys (``param.hf_keys``) are always
        written in the post-remap namespace.
    strict : raise if any *required* key is missing or any HF key was
        unexpected. Optional missing keys never raise.
    progress : control the per-tensor progress bar (rank 0 only).
        ``None`` (default) auto-detects — shown on an interactive TTY /
        notebook, silent when output is piped or captured (CI, pytest).
        ``True`` forces it on even off-TTY; ``False`` disables it.
    revision : branch / tag / commit selected when ``source`` is a repo id
        downloaded from the Hub. Ignored for local sources.

    Returns
    -------
    LoadReport with diagnostics.
    """
    remap_fn = _resolve_remap(remap)
    paths = _resolve_source(source, revision=revision)

    # 1. Build dispatch index from data on params.
    index: dict[str, tuple[nn.Parameter, "int | str | None", WeightLoader]] = {}
    optional: set[str] = set()
    for _name, param in model.named_parameters():
        keys = getattr(param, "hf_keys", None)
        if keys is None:
            continue
        loader: WeightLoader = getattr(param, "weight_loader", None) or replicated()
        is_optional = bool(getattr(param, "optional", False))
        for hf_key, shard_id in keys:
            if hf_key in index:
                raise RuntimeError(
                    f"hf_key {hf_key!r} is claimed by two params; "
                    f"second hit on {_name!r}."
                )
            index[hf_key] = (param, shard_id, loader)
            if is_optional:
                optional.add(hf_key)

    report = LoadReport()
    seen: set[str] = set()

    # 2. Stream safetensors; dispatch. A rank-0 progress bar advances once
    #    per tensor key (total = key count across all shards), so it always
    #    fills to 100% regardless of remap drops / unexpected keys.
    disable = _progress_disable(progress)
    total = None if disable is True else _count_keys(paths)
    bar = tqdm(
        total=total,
        disable=disable,
        unit="tensor",
        desc=f"Loading {_source_label(source)}",
        leave=False,
    )
    for path in paths:
        bar.set_postfix_str(path.name, refresh=False)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for raw in f.keys():
                bar.update(1)
                hf = remap_fn(raw)
                if hf is None:
                    continue
                hit = index.get(hf)
                if hit is None:
                    report.unexpected.append(hf)
                    continue
                param, shard_id, loader = hit
                tensor = f.get_tensor(raw)
                if tensor.dtype != param.dtype:
                    report.casts.append((hf, tensor.dtype, param.dtype))
                loader(param, tensor, shard_id)
                seen.add(hf)
                report.loaded.append(hf)
    bar.close()

    # 3. Diagnose missing.
    for hf_key in index:
        if hf_key in seen:
            continue
        if hf_key in optional:
            report.optional_missing.append(hf_key)
        else:
            report.missing.append(hf_key)

    if strict and (report.missing or report.unexpected):
        raise RuntimeError(f"load_pretrained strict failure: {report.summary()}")

    # 4. Per-module post-load hook (e.g. fp8 scale fixup).
    for module in model.modules():
        post = getattr(module, "post_load", None)
        if callable(post):
            post()

    if report.casts:
        for hf_key, src_dtype, dst_dtype in report.casts[:10]:
            _logger.warning(
                "load_pretrained dtype cast at %r: %s -> %s",
                hf_key,
                src_dtype,
                dst_dtype,
            )

    this_rank_log(
        _logger,
        logging.INFO,
        "load_pretrained(%s): %s",
        _source_label(source),
        report.summary(),
    )

    return report


__all__ = ["LoadReport", "load_pretrained"]
