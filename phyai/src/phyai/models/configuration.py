"""Base config for phyai.models.

A :class:`PretrainedConfig` is a frozen dataclass with three jobs:

* be constructable from a JSON file or dict, silently dropping keys
  the dataclass doesn't declare (so phyai configs can ride along on
  upstream ``config.json`` files that carry many unrelated knobs);
* expose every declared field via mapping-style access
  (``cfg["hidden_size"]``, ``for k, v in cfg.items()``) so generic
  builders can inspect a config without knowing the concrete subclass;
* be hashable and immutable, so configs can travel through
  ``functools.lru_cache`` and graph-capture machinery.

Concrete subclasses just declare their fields with
``@dataclass(frozen=True)``. Two hooks let a subclass load an upstream
``config.json`` whose schema *nests* or *renames* values the phyai config keeps
flat, with no bespoke ``from_*`` method:

* a **nested sub-config field** (one whose default is itself a
  :class:`PretrainedConfig`) is built recursively from its dict form by
  :meth:`PretrainedConfig.from_dict` — automatically, nothing to declare;
* :attr:`PretrainedConfig.nested_sources` declares, per flat field, where to
  find its value when the upstream schema buries it under a different key or
  inside a nested dict (e.g. ``mrope_section`` living in ``rope_scaling``, or a
  ``vision`` sub-config arriving as ``vision_config``).
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Iterator, Mapping, TypeVar


T = TypeVar("T", bound="PretrainedConfig")


def _dig(data: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    """Descend a dotted path through nested dicts; return ``(found, value)``.

    ``found`` is ``False`` (and ``value`` ``None``) if any segment is missing or
    a non-dict is hit partway, so a declared source the checkpoint omits simply
    leaves the field at its default.
    """
    cur: Any = data
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return False, None
        cur = cur[key]
    return True, cur


def _subconfig_type(default_factory: Any) -> type[PretrainedConfig] | None:
    """The config subclass a field's ``default_factory`` produces, else ``None``.

    Lets :meth:`PretrainedConfig.from_dict` recognise a nested sub-config field
    (so a dict value is built into that config rather than stored raw) without
    evaluating string type annotations. The common case — ``default_factory`` is
    the config class itself — needs no throwaway instance.
    """
    if default_factory is MISSING:
        return None
    if isinstance(default_factory, type) and issubclass(
        default_factory, PretrainedConfig
    ):
        return default_factory
    try:
        produced = default_factory()
    except Exception:
        return None
    return type(produced) if isinstance(produced, PretrainedConfig) else None


@dataclass(frozen=True)
class PretrainedConfig:
    """Base for every model config in phyai.models.

    Subclass with ``@dataclass(frozen=True)``. Every field must declare a
    default. :meth:`from_dict` filters the input dict to declared field names
    (so unknown upstream keys are dropped instead of raising), recursively
    builds nested sub-config fields, and honours :attr:`nested_sources`.
    """

    #: Optional per-subclass map ``flat_field -> source(s)`` for values the
    #: upstream schema nests or renames. A source is a dotted path into the raw
    #: dict (``"rope_scaling.mrope_section"``); a tuple of paths is tried in
    #: order, first hit wins. A top-level key matching the field name always
    #: takes precedence over any nested source. Empty for flat configs.
    nested_sources: ClassVar[Mapping[str, str | tuple[str, ...]]] = {}

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Build an instance from a dict; unknown keys are silently dropped.

        Beyond filtering to declared fields, this:

        * lifts each :attr:`nested_sources` entry from its (possibly nested or
          renamed) location to the flat field — unless that field is already
          present at the top level, which wins;
        * recursively builds any nested sub-config field (a field whose default
          is a :class:`PretrainedConfig`) from its dict form.

        So a full upstream ``config.json`` — nested sub-configs, buried RoPE
        knobs, and unrelated optimizer/device keys alike — loads directly, with
        no bespoke per-model conversion method.
        """
        data = dict(data)  # shallow copy: lifted keys must not touch the caller's dict
        # Lift declared nested / renamed sources to flat fields (top-level wins).
        for name, sources in cls.nested_sources.items():
            if name in data:
                continue
            for path in (sources,) if isinstance(sources, str) else sources:
                found, value = _dig(data, path)
                if found:
                    data[name] = value
                    break
        # Filter to declared fields, recursively building sub-config fields.
        factories = {f.name: f.default_factory for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for name in cls.field_names():
            if name not in data:
                continue
            value = data[name]
            if isinstance(value, dict):
                sub = _subconfig_type(factories[name])
                if sub is not None:
                    value = sub.from_dict(value)
            kwargs[name] = value
        return cls(**kwargs)

    @classmethod
    def from_json(cls: type[T], path: str | Path) -> T:
        """Read JSON from ``path`` and dispatch through :meth:`from_dict`."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a JSON object at the top level, got "
                f"{type(data).__name__}."
            )
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------ #
    # Mapping-like read access                                           #
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: str) -> Any:
        if key not in self.field_names():
            raise KeyError(
                f"{type(self).__name__} has no field {key!r}; "
                f"valid fields: {sorted(self.field_names())!r}."
            )
        return getattr(self, key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.field_names()

    def keys(self) -> Iterator[str]:
        return iter(self.field_names())

    def items(self) -> Iterator[tuple[str, Any]]:
        return ((f.name, getattr(self, f.name)) for f in fields(self))


__all__ = ["PretrainedConfig"]
