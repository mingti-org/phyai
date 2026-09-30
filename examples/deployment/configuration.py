"""Reusable structured-YAML loading with dotted command-line overrides.

Callers supply their own dataclass schema, default YAML, and optional semantic
validator. This module knows no robot, model, control mode, or action layout.
Schema field types are checked by OmegaConf before the caller's validator runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Callable, Sequence

from yaml import YAMLError
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

# Resolve from this module, not the process cwd: launching from another directory
# must not silently select a different checkpoint or kernel configuration.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def repository_path(value: str) -> Path:
    """Resolve an example's runtime path against the checkout, expanding '~'.

    Runtime paths do not change meaning when a YAML file is moved to another
    directory. The --config path itself still resolves from the working directory.
    Absolute paths stay absolute; relative paths are rooted at REPOSITORY_ROOT.
    This function normalizes a path but does not require it to exist. The module
    that owns the setting decides whether a file or a directory is required.
    """
    return (REPOSITORY_ROOT / Path(value).expanduser()).resolve()


def parse_config[ConfigT](
    schema: type[ConfigT],
    *,
    default_config: Path,
    argv: Sequence[str] | None = None,
    validate: Callable[[ConfigT], None] | None = None,
) -> ConfigT:
    """Merge schema < default YAML < selected YAML < dotted CLI overrides.

    Validation is supplied by the entry point, keeping device/model constraints
    out of this loader. --print-config permits missing required values and skips
    semantic validation, so it can inspect configuration without creating any
    runtime resources. Normal parsing requires all mandatory fields and runs
    validation before returning the fully typed dataclass object.

    Args:
        schema: Dataclass type whose fields define the allowed keys and types.
        default_config: Bundled YAML defaults; a user YAML may override a subset.
        argv: CLI arguments without the executable name, or None for sys.argv.
        validate: Optional callback for cross-field or resource-path checks.
            Raising ValueError reports a normal argparse usage error.

    Returns:
        An instance of schema, including typed nested dataclasses. OmegaConf's
        mutable configuration objects do not escape into the deployment loop.

    Raises:
        SystemExit: Help/print requests exit with code 0. Invalid options, missing
            mandatory values, unreadable files, and validation errors exit with 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="YAML configuration (may override only selected defaults)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged YAML and exit without creating runtime resources",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="Typed dotted overrides; field names come from the selected schema",
    )
    args = parser.parse_args(argv)
    try:
        if any("=" not in item or not item.split("=", 1)[0] for item in args.overrides):
            raise ValueError("overrides must use key=value syntax")
        # The structured first layer rejects unknown keys and checks field
        # types in every later layer. Load bundled defaults even when --config
        # selects another file, so a private YAML can contain only its overrides.
        # Dot-list parsing preserves values such as true, null, and numeric lists.
        merged = OmegaConf.merge(
            OmegaConf.structured(schema),
            OmegaConf.load(default_config),
            OmegaConf.load(args.config.expanduser()),
            OmegaConf.from_dotlist(list(args.overrides)),
        )
        # Print before materializing the dataclass: unresolved mandatory fields
        # such as checkpoint: ??? are useful in a template, but not at runtime.
        if args.print_config:
            print(OmegaConf.to_yaml(merged, resolve=True), end="")
            parser.exit()
        # Materialization rejects missing mandatory fields. Semantic validation
        # runs only after type checking, before the caller creates ROS/GPU objects.
        config = OmegaConf.to_object(merged)
        assert isinstance(config, schema)
        if validate is not None:
            validate(config)
        return config
    # Keep user configuration mistakes on the CLI error path; unexpected
    # programming errors are deliberately not hidden behind a broad catch.
    except (OSError, ValueError, YAMLError, OmegaConfBaseException) as error:
        parser.error(str(error))
