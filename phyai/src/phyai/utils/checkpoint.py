"""HuggingFace-style checkpoint folder helpers.

A "checkpoint" in phyai follows the same on-disk layout as a
``transformers`` snapshot: one folder containing

* ``config.json`` — the model's architecture/geometry,
* one or more ``*.safetensors`` shards, optionally indexed by
  ``model.safetensors.index.json``.

This module covers the filesystem side of that layout:

* :func:`resolve_checkpoint` — turn a ``source`` into a local folder/file
  path, downloading from the HuggingFace Hub when ``source`` is a repo id
  rather than a path that exists on disk.
* :func:`find_safetensors` — list the safetensors shards in a folder,
  honouring ``model.safetensors.index.json`` when present.
* :func:`load_config` — parse ``config.json`` into a
  :class:`~phyai.models.configuration.PretrainedConfig` subclass.

Actually loading tensors into an :class:`nn.Module` is the job of
:func:`phyai.weights.load_pretrained`, which accepts a checkpoint folder
directly and reuses :func:`find_safetensors` internally — call sites do
not need to expand the folder themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from phyai.models.configuration import PretrainedConfig


T = TypeVar("T", bound="PretrainedConfig")


_SAFETENSORS_SINGLE = "model.safetensors"
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_DEFAULT_CONFIG_FILENAME = "config.json"


def _ensure_dir(folder: Path) -> Path:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"checkpoint folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(
            f"expected a checkpoint folder, got a file path: {folder}. "
            f"Pass the folder containing config.json + safetensors shard(s)."
        )
    return folder


def resolve_checkpoint(source: str | Path, *, revision: str | None = None) -> Path:
    """Resolve ``source`` to a local path, downloading from the Hub if needed.

    Resolution:

    1. If ``source`` already exists on disk (a folder *or* a file), it is
       returned unchanged — the existing local-only behaviour.
    2. Otherwise ``source`` is interpreted as a HuggingFace **repo id** and
       the whole repo is fetched with :func:`huggingface_hub.snapshot_download`;
       the local snapshot directory is returned.

    To keep step 1 free of any network/SDK cost, ``huggingface_hub`` is imported
    lazily inside the download branch — purely local loads never touch it.

    Before any network call, the candidate is checked with
    :func:`huggingface_hub.utils.validate_repo_id`. A string that is neither a
    local path nor a syntactically valid repo id (a typo'd path such as
    ``/data/typo`` or ``./ckpt``) raises :class:`FileNotFoundError` immediately,
    offline — so a mistyped path fails fast instead of hanging on a doomed
    download. Authentication, cache location, and offline mode are taken from the
    usual ``HF_TOKEN`` / ``HF_HOME`` / ``HF_HUB_OFFLINE`` environment variables;
    ``revision`` (branch / tag / commit) is the one per-call knob.
    """
    path = Path(source)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(str(source))
    except HFValidationError as exc:
        raise FileNotFoundError(
            f"{source!r} is not a local path and is not a valid HuggingFace "
            f"repo id. If you meant a local checkpoint, check the path."
        ) from exc

    local = snapshot_download(repo_id=str(source), repo_type="model", revision=revision)
    return Path(local)


def find_safetensors(folder: str | Path) -> list[Path]:
    """List safetensors shard files in ``folder``.

    Resolution order, mirroring HuggingFace's snapshot layout:

    1. ``model.safetensors.index.json`` exists -> parse its
       ``weight_map`` and return every distinct shard it references,
       sorted by filename.
    2. ``model.safetensors`` exists -> return ``[folder/model.safetensors]``.
    3. Otherwise fall back to a glob of ``*.safetensors`` (catches
       non-canonical filenames). Raises if none are found.

    Returned paths are absolute. Files referenced by the index but
    missing on disk raise :class:`FileNotFoundError`.
    """
    folder = _ensure_dir(folder)

    index_path = folder / _SAFETENSORS_INDEX
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"{index_path}: missing or empty 'weight_map' — cannot resolve shards."
            )
        shards = sorted({str(v) for v in weight_map.values()})
        resolved = [(folder / shard).resolve() for shard in shards]
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(
                    f"{index_path} references {path.name} but the file is "
                    f"missing from {folder}."
                )
        return resolved

    single = folder / _SAFETENSORS_SINGLE
    if single.is_file():
        return [single.resolve()]

    fallback = sorted(folder.glob("*.safetensors"))
    if fallback:
        return [p.resolve() for p in fallback]

    raise FileNotFoundError(
        f"no safetensors shards found in {folder}: expected one of "
        f"{_SAFETENSORS_INDEX!r}, {_SAFETENSORS_SINGLE!r}, or any '*.safetensors'."
    )


def load_config(
    folder: str | Path,
    config_cls: type[T],
    *,
    filename: str = _DEFAULT_CONFIG_FILENAME,
    revision: str | None = None,
) -> T:
    """Read ``filename`` from ``folder`` and parse it via ``config_cls.from_json``.

    ``folder`` may be a local checkpoint directory or a HuggingFace repo id —
    it is passed through :func:`resolve_checkpoint` first (``revision`` selects
    the branch / tag / commit when downloading).

    ``config_cls`` must be a :class:`~phyai.models.configuration.PretrainedConfig`
    subclass. Unknown JSON keys are silently dropped — see
    :meth:`PretrainedConfig.from_dict`.
    """
    folder = _ensure_dir(resolve_checkpoint(folder, revision=revision))
    config_path = folder / filename
    if not config_path.is_file():
        raise FileNotFoundError(
            f"config file not found: {config_path}. "
            f"Pass a checkpoint folder that contains '{filename}'."
        )
    return config_cls.from_json(config_path)


__all__ = ["find_safetensors", "load_config", "resolve_checkpoint"]
