"""Configuration for a model server and its request adapter."""

from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phyai.engine import EngineArgs
from phyai.engine_builder import (
    ConfigError,
    ConfigReader,
    engine_args_from_mapping,
    load_yaml_mapping,
)
from phyai.server.deployment import DeploymentConfig


@dataclass(frozen=True)
class ServerOptions:
    model_name: str
    host: str = "0.0.0.0"
    port: int = 50063
    workers: int = 1
    max_message_bytes: int = 100 * 1024 * 1024
    grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name in ("model_name", "host"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\0" in value:
                raise ValueError(f"{name} must be a nonempty string")
            object.__setattr__(self, name, value.strip())
        if self.host.startswith("[") and self.host.endswith("]"):
            object.__setattr__(self, "host", self.host[1:-1])
        if not self.host or any(
            char.isspace() or char in "/\\?#@[]" for char in self.host
        ):
            raise ValueError("host must be a hostname or IP address without a port")
        if ":" in self.host:
            try:
                ipaddress.IPv6Address(self.host)
            except ValueError as error:
                raise ValueError(
                    "host must be a hostname or IP address without a port"
                ) from error
        for name, maximum in (
            ("port", 65535),
            ("workers", None),
            ("max_message_bytes", 2**31 - 1),
        ):
            value = getattr(self, name)
            if (
                type(value) is not int
                or value < 1
                or (maximum is not None and value > maximum)
            ):
                limit = (
                    f" in 1..{maximum}" if maximum is not None else " greater than zero"
                )
                raise ValueError(f"{name} must be an integer{limit}")
        if (
            isinstance(self.grace_seconds, bool)
            or not isinstance(self.grace_seconds, (float, int))
            or not math.isfinite(self.grace_seconds)
            or self.grace_seconds < 0
        ):
            raise ValueError("grace_seconds must be finite and nonnegative")


@dataclass(frozen=True)
class ServerConfig:
    server: ServerOptions
    engine_args: EngineArgs
    deployment: DeploymentConfig | None
    adapter_factory: str
    adapter_args: dict[str, Any]


def _adapter_paths(value: Any, directory: Path) -> Any:
    if isinstance(value, dict):
        return {name: _adapter_paths(item, directory) for name, item in value.items()}
    if isinstance(value, list):
        return [_adapter_paths(item, directory) for item in value]
    if isinstance(value, str) and (
        value.startswith(("./", "../", "/", "~")) or value in (".", "..")
    ):
        return str((directory / Path(value).expanduser()).resolve())
    return value


def load_server_config(path: str | Path) -> ServerConfig:
    """Validate YAML settings without loading weights or constructing a backend."""
    source, data = load_yaml_mapping(path)
    reader = ConfigReader(source.parent)
    try:
        if "server" not in data:
            raise ConfigError("server: required field is missing.")
        server = reader.convert(data.pop("server"), ServerOptions, "server")
        adapter = reader.convert(data.pop("adapter", {}), dict[str, Any], "adapter")
        for name in adapter:
            if name not in ("factory", "args"):
                raise ConfigError(f"adapter.{name}: unknown field.")
        factory = reader.convert(
            adapter.get("factory", "phyai.server.backends.tensor:TensorBackend"),
            str,
            "adapter.factory",
        )
        module, separator, symbol = factory.partition(":")
        if (
            not separator
            or not all(part.isidentifier() for part in module.split("."))
            or not all(part.isidentifier() for part in symbol.split("."))
        ):
            raise ConfigError("adapter.factory: expected 'module:callable'.")
        adapter_args = reader.convert(
            adapter.get("args", {}), dict[str, Any], "adapter.args"
        )
        for reserved in ("engine_args", "deployment"):
            if reserved in adapter_args:
                raise ConfigError(f"adapter.args.{reserved}: supplied by the server.")
        engine_args, deployment = engine_args_from_mapping(
            data, directory=source.parent
        )
        return ServerConfig(
            server=server,
            engine_args=engine_args,
            deployment=deployment,
            adapter_factory=factory,
            adapter_args=_adapter_paths(adapter_args, source.parent),
        )
    except ConfigError as error:
        raise ValueError(f"{source}: {error}") from error


__all__ = ["ServerConfig", "ServerOptions", "load_server_config"]
