"""Adapt released VPP2 checkpoint containers to the PHYAI weight loader."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from phyai.utils import get_logger
from phyai.utils.checkpoint import find_checkpoint_files
from phyai.weights import LoadReport, load_pretrained
from phyai.weights.loader import checkpoint_format, iter_checkpoint_tensors


logger = get_logger(__name__)


def read_vpp2_checkpoint(path: str | Path) -> Mapping:
    """Memory-map release tensors while reading architecture metadata."""
    path = Path(path)
    if not path.is_file() or path.suffix in {".safetensors", ".json"}:
        raise ValueError(
            "Architecture metadata requires a PyTorch release checkpoint; "
            "provide VPP2Args.config for safetensors or sharded sources."
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError as error:
        if "mmap can only be used" not in str(error):
            raise
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("VPP2 checkpoints must contain a tensor mapping.")
    return checkpoint


def vpp2_checkpoint_files(path: str | Path) -> list[Path]:
    """Resolve a local file, shard directory, or explicit shard index."""
    path = Path(path).resolve()
    if path.is_dir():
        return find_checkpoint_files(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name.endswith(".index.json"):
        index = json.loads(path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Checkpoint index must have a nonempty weight_map.")
        if any(not isinstance(filename, str) for filename in weight_map.values()):
            raise ValueError("Checkpoint index shard names must be strings.")
        files = sorted(
            {(path.parent / filename).resolve() for filename in weight_map.values()}
        )
        for shard in files:
            if not shard.is_file():
                raise FileNotFoundError(
                    f"Checkpoint index references missing shard: {shard}"
                )
            checkpoint_format(shard)
        return files
    checkpoint_format(path)
    return [path]


def canonical_vpp2_key(name: str, component: str) -> str:
    """Normalize container aliases while retaining original attention leg names."""
    wrappers = ("module.", "state_dict.", "model_state_dict.", "model_state.", "model.")
    while name.startswith(wrappers):
        name = name.split(".", 1)[1]
    if component == "video":
        prefixes = (
            "mot.mixtures.video.",
            "mixtures.video.",
            "video_expert.",
            "video.",
            "dit.",
            "diffusion_model.",
        )
        for prefix in prefixes:
            if name.startswith(prefix):
                return canonical_vpp2_key(name[len(prefix) :], component)
        return f"video_expert.{name}"
    if name.startswith("proprio_encoder."):
        return name
    for prefix in (
        "mot.mixtures.action.",
        "mixtures.action.",
        "action_expert.",
        "action.",
    ):
        if name.startswith(prefix):
            return canonical_vpp2_key(name[len(prefix) :], component)
    return f"action_expert.{name}"


def iter_vpp2_tensors(path: Path, component: str) -> Iterator[tuple[str, torch.Tensor]]:
    if checkpoint_format(path) == "safetensors":
        for name, value in iter_checkpoint_tensors(path):
            yield canonical_vpp2_key(name, component), value
        return
    state = read_vpp2_checkpoint(path)
    wrappers = (
        "dit",
        "state_dict",
        "model_state_dict",
        "module",
        "model_state",
        "model",
    )
    if component == "video":
        wrappers = (*wrappers, "video_expert")
    while True:
        nested = next(
            (state[key] for key in wrappers if isinstance(state.get(key), Mapping)),
            None,
        )
        if nested is None:
            break
        state = nested
    if component == "action" and isinstance(state.get("action_expert"), Mapping):
        for prefix in ("action_expert", "proprio_encoder"):
            values = state.get(prefix)
            if isinstance(values, Mapping):
                for name, value in values.items():
                    if isinstance(name, str) and isinstance(value, torch.Tensor):
                        yield canonical_vpp2_key(f"{prefix}.{name}", component), value
        return
    for name, value in state.items():
        if isinstance(name, str) and isinstance(value, torch.Tensor):
            yield canonical_vpp2_key(name, component), value


def canonical_vpp2_safetensors(files: list[Path], component: str) -> bool:
    """Inspect headers only; canonical sources need no cache copy."""
    found = False
    names: set[str] = set()
    for path in files:
        if checkpoint_format(path) != "safetensors":
            return False
        with safe_open(str(path), framework="pt", device="cpu") as source:
            for name in source.keys():
                found = True
                if canonical_vpp2_key(name, component) != name:
                    return False
                if name in names:
                    raise ValueError(f"Duplicate VPP2 checkpoint tensor: {name}")
                names.add(name)
    return found


def prepare_vpp2_checkpoint(
    path: str | Path,
    *,
    component: str,
    cache_dir: str | Path,
) -> Path:
    """Convert noncanonical tensors to reusable shards, serialized across TP ranks."""
    if component not in {"video", "action"}:
        raise ValueError("component must be 'video' or 'action'.")
    sources = vpp2_checkpoint_files(path)
    if len(sources) == 1:
        source = sources[0]
        stat = source.stat()
        # Retain compatibility with caches prepared from the original releases.
        fingerprint_data = (
            f"vpp2-v1:{component}:{source}:{stat.st_size}:{stat.st_mtime_ns}"
        )
    else:
        fingerprint_data = (
            json.dumps(
                [
                    (str(source), source.stat().st_size, source.stat().st_mtime_ns)
                    for source in sources
                ]
            )
            + component
        )
    fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:24]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / fingerprint
    with (cache_dir / f"{fingerprint}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "complete.json").is_file():
            return destination
        logger.info_rank0("Preparing VPP2 %s checkpoint shards.", component)
        with tempfile.TemporaryDirectory(
            prefix=f"{fingerprint}-", dir=cache_dir
        ) as staging:
            staging_path = Path(staging)
            shard: dict[str, torch.Tensor] = {}
            shard_bytes = 0
            shard_index = 0
            weight_map: dict[str, str] = {}

            def flush() -> None:
                nonlocal shard_index, shard_bytes
                if not shard:
                    return
                name = f"model-{shard_index:05d}.safetensors"
                save_file(shard, str(staging_path / name))
                weight_map.update({key: name for key in shard})
                shard.clear()
                shard_index += 1
                shard_bytes = 0

            for source in sources:
                for name, tensor in iter_vpp2_tensors(source, component):
                    if name in shard or name in weight_map:
                        raise ValueError(f"Duplicate VPP2 checkpoint tensor: {name}")
                    size = tensor.numel() * tensor.element_size()
                    if shard_bytes + size > 1024**3:
                        flush()
                    shard[name] = tensor.detach().contiguous()
                    shard_bytes += size
            flush()
            if not weight_map:
                raise ValueError("Checkpoint contains no model tensors.")
            (staging_path / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {}, "weight_map": weight_map})
            )
            (staging_path / "complete.json").write_text(
                json.dumps({"tensors": len(weight_map)})
            )
            os.rename(staging_path, destination)
        return destination


def load_vpp2_weights(
    model: torch.nn.Module,
    video_path: str | Path,
    action_path: str | Path | None,
    *,
    remap: Callable[[str], str | None],
    cache_dir: str | Path = ".cache/vpp2/weights",
    progress: bool = False,
) -> LoadReport:
    files = []
    for component, source in (("video", video_path), ("action", action_path)):
        if source is None:
            continue
        resolved = vpp2_checkpoint_files(source)
        if canonical_vpp2_safetensors(resolved, component):
            files.extend(resolved)
        else:
            prepared = prepare_vpp2_checkpoint(
                source, component=component, cache_dir=cache_dir
            )
            files.extend(find_checkpoint_files(prepared))
    report = load_pretrained(model, files, remap=remap, strict=True, progress=progress)
    if report.missing:
        raise RuntimeError(
            f"VPP2 checkpoint has {len(report.missing)} missing weights."
        )
    return report
