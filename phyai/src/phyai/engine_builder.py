"""Build an Engine from a YAML file using the existing configuration types."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, fields, is_dataclass, replace
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import torch
import yaml
from yaml.constructor import ConstructorError

from phyai.engine import Engine, EngineArgs, EngineCore
from phyai.engine_config import EngineConfig
from phyai.env import _parse_dtype
from phyai.kernel.config import KernelConfig
from phyai.server.deployment import DeploymentConfig
from phyai.server.worker_supervisor import WorkerSupervisorConfig


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConstructorError(
                    None, None, "mapping keys must be strings", key_node.start_mark
                )
            if key in mapping:
                raise ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


class ConfigError(ValueError):
    """A configuration error whose message already includes its field path."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(
            f"{location}: expected a mapping, got {type(value).__name__}."
        )
    return dict(value)


class ConfigReader:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def convert(self, value: Any, expected: Any, location: str) -> Any:
        origin = get_origin(expected)
        args = get_args(expected)
        if expected is Any:
            return value
        if origin in (Union, UnionType):
            if value is None and type(None) in args:
                return None
            alternatives = [arg for arg in args if arg is not type(None)]
            if len(alternatives) == 1:
                return self.convert(value, alternatives[0], location)
            errors = []
            for alternative in alternatives:
                try:
                    return self.convert(value, alternative, location)
                except ConfigError as error:
                    errors.append(str(error))
            raise ConfigError("; ".join(errors))
        if is_dataclass(expected):
            return self.dataclass(value, expected, location)
        if expected is torch.dtype:
            if not isinstance(value, str):
                raise ConfigError(
                    f"{location}: expected a dtype name, such as bfloat16."
                )
            try:
                return _parse_dtype(value)
            except ValueError as error:
                raise ConfigError(f"{location}: {error}") from error
        if expected is Path:
            return self.path(self.convert(value, str, location))
        if origin is Literal:
            if any(type(value) is type(arg) and value == arg for arg in args):
                return value
            raise ConfigError(f"{location}: expected one of {args!r}, got {value!r}.")
        if (origin or expected) in (list, tuple, Sequence):
            if not isinstance(value, list):
                raise ConfigError(f"{location}: expected a YAML sequence.")
            container = origin or expected
            if container is tuple and args and args[-1] is not Ellipsis:
                if len(value) != len(args):
                    raise ConfigError(f"{location}: expected {len(args)} items.")
                item_types = args
            else:
                item_types = (args[0] if args else Any,) * len(value)
            items = [
                self.convert(item, item_type, f"{location}[{index}]")
                for index, (item, item_type) in enumerate(zip(value, item_types))
            ]
            return tuple(items) if container is tuple else items
        if (origin or expected) in (dict, Mapping):
            data = _mapping(value, location)
            key_type, value_type = args or (str, Any)
            return {
                self.convert(key, key_type, f"{location} key"): self.convert(
                    item, value_type, f"{location}.{key}"
                )
                for key, item in data.items()
            }
        if (origin or expected) is Callable or expected is torch.Tensor:
            raise ConfigError(
                f"{location}: callable and tensor values require the Python Engine API."
            )
        if expected in (str, bool, int, float, type(None)):
            if type(value) is expected:
                return value
            if expected is float and type(value) is int:
                return float(value)
            raise ConfigError(
                f"{location}: expected {expected.__name__}, got {type(value).__name__}."
            )
        raise ConfigError(f"{location}: {expected!r} is not supported in YAML.")

    def dataclass(self, value: Any, cls: type, location: str) -> Any:
        data = _mapping(value, location)
        declared = {field.name: field for field in fields(cls) if field.init}
        for name in data:
            if name not in declared:
                raise ConfigError(
                    f"{location}.{name}: unknown field for {cls.__name__}."
                )
        hints = get_type_hints(cls)
        kwargs = {}
        for name, field in declared.items():
            field_path = f"{location}.{name}"
            if name not in data:
                if field.default is MISSING and field.default_factory is MISSING:
                    raise ConfigError(f"{field_path}: required field is missing.")
                continue
            expected = hints[name]
            # Deployment deliberately leaves this field open for Python callers.
            if cls is DeploymentConfig and name == "process_config":
                expected = WorkerSupervisorConfig | None
            converted = self.convert(data[name], expected, field_path)
            if (
                cls is KernelConfig
                and name == "policy_config"
                and converted is not None
            ):
                self.validate_policy(converted, field_path)
            kwargs[name] = self.path_value(converted, field_path)
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{location}: {error}") from error

    def validate_policy(self, value: Mapping[str, Any], location: str) -> None:
        from phyai.kernel.policy import policy_from_mapping
        from phyai.kernel.registry import build_catalog

        try:
            policy_from_mapping(value, build_catalog(), source=location)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{location}: {error}") from error

    def kernel_policy(self, value: Any, config: EngineConfig | None) -> EngineConfig:
        config = config if config is not None else EngineConfig.auto()
        kernel = config.kernel
        if kernel.config_path is not None or kernel.policy_config is not None:
            raise ConfigError(
                "kernel_policy: cannot also set config.kernel.config_path or "
                "config.kernel.policy_config."
            )
        if isinstance(value, str):
            if not value.strip():
                raise ConfigError("kernel_policy: expected a non-empty policy path.")
            kernel = replace(kernel, config_path=str(self.path(value)))
        else:
            policy = _mapping(value, "kernel_policy")
            self.validate_policy(policy, "kernel_policy")
            kernel = replace(kernel, policy_config=policy)
        return replace(config, kernel=kernel)

    def path(self, value: str) -> Path:
        return (self.directory / Path(value).expanduser()).resolve()

    def path_value(self, value: Any, location: str) -> Any:
        if value is None:
            return None
        if location in (
            "config.kernel.config_path",
            "config.kernel.autotune_cache",
            "config.runtime.debug_tensor_dump_dir",
        ):
            return str(self.path(value))
        if location in ("plugin_args.checkpoint_dir", "plugin_args.checkpoint"):
            # A bare name such as org/model must remain a HuggingFace repo ID.
            if isinstance(value, str) and (
                value.startswith(("./", "../", "/", "~")) or value in (".", "..")
            ):
                return str(self.path(value))
        if location == "config.runtime.debug_tensor_dump_filter_fn":
            module, separator, function = value.rpartition(":")
            if not separator or not module or not function:
                raise ConfigError(
                    f"{location}: expected 'pkg.module:func' or './file.py:func'."
                )
            if module.endswith(".py"):
                return f"{self.path(module)}:{function}"
        return value


def load_yaml_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Read a local YAML mapping and reject duplicate or non-string keys."""
    source = Path(path).expanduser().resolve()
    try:
        with source.open(encoding="utf-8") as stream:
            data = _mapping(yaml.load(stream, Loader=_UniqueKeyLoader), "<root>")
    except (yaml.YAMLError, ConfigError) as error:
        raise ValueError(f"{source}: {error}") from error
    return source, data


def engine_args_from_mapping(
    data: Mapping[str, Any], *, directory: str | Path
) -> tuple[EngineArgs, DeploymentConfig | None]:
    """Parse engine and deployment settings without constructing an Engine."""
    data = _mapping(data, "<root>")
    for name in data:
        if name not in (
            "plugin",
            "plugin_args",
            "config",
            "deployment",
            "kernel_policy",
        ):
            raise ConfigError(f"{name}: unknown top-level field.")
    reader = ConfigReader(Path(directory).expanduser().resolve())
    if "plugin" not in data:
        raise ConfigError("plugin: required field is missing.")
    plugin = reader.convert(data["plugin"], str, "plugin")
    try:
        entry_cls = EngineCore.plugin_class(plugin)
    except ValueError as error:
        raise ConfigError(f"plugin: {error}") from error
    plugin_args = reader.convert(
        data.get("plugin_args", {}), entry_cls.args_cls, "plugin_args"
    )
    config = reader.convert(data.get("config"), EngineConfig | None, "config")
    if data.get("kernel_policy") is not None:
        config = reader.kernel_policy(data["kernel_policy"], config)
    deployment = reader.convert(
        data.get("deployment"), DeploymentConfig | None, "deployment"
    )
    return EngineArgs(plugin=plugin, plugin_args=plugin_args, config=config), deployment


def _load_engine_config(path: str | Path) -> tuple[EngineArgs, DeploymentConfig | None]:
    source, data = load_yaml_mapping(path)
    try:
        return engine_args_from_mapping(data, directory=source.parent)
    except ConfigError as error:
        raise ValueError(f"{source}: {error}") from error


def build_engine(path: str | Path) -> Engine:
    """Build an Engine from plugin, config, deployment, and kernel policy YAML.

    Omitted fields keep their Python defaults; PHYAI environment overrides
    apply at startup. Local paths are relative to the YAML directory. Prefix
    local checkpoint paths with ``./`` or ``../`` to distinguish Hub repo IDs.
    Call under an ``if __name__ == "__main__"`` guard when using managed workers.
    """
    args, deployment = _load_engine_config(path)
    return Engine(args, deployment=deployment)


__all__ = [
    "ConfigError",
    "ConfigReader",
    "build_engine",
    "engine_args_from_mapping",
    "load_yaml_mapping",
]
