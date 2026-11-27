"""Build pi0.5 processing from an explicit server configuration."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .configuration import repository_path
from .pi05 import Pi05Processing, Pi05ProcessorConfig


def _file_module(path: Path) -> ModuleType:
    """Give each source file a private package namespace for relative imports."""
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(
            f"processor.factory source must be an existing Python file: {path}"
        )
    name = "_phyai_processor_" + hashlib.sha256(str(path).encode()).hexdigest()
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)]
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load processor source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Failed imports must not leave partially initialized helpers cached.
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]
        raise
    return module


def build_processing(
    checkpoint: Path, *, task: str, config: Pi05ProcessorConfig
) -> Pi05Processing:
    """Instantiate the default processor or an explicitly selected factory.

    An external factory receives (checkpoint, *, task, config, **options) and
    returns a Pi05Processing instance. Source files may live outside the checkout
    and use relative imports for sibling helpers. Their dependencies run in the
    server's interpreter. Loading does not alter sys.path or discover checkpoint
    code automatically. Restart the server after changing an imported source.
    """
    config.validate()
    if config.factory is None:
        return Pi05Processing(checkpoint, task=task, config=config)
    source, _, attribute = config.factory.rpartition(":")
    if not source or not attribute.isidentifier():
        raise ValueError("processor.factory must be module:factory or file.py:factory")
    if source.endswith(".py") or "/" in source or source.startswith("~"):
        module = _file_module(repository_path(source))
    else:
        if not all(part.isidentifier() for part in source.split(".")):
            raise ValueError(f"Invalid processor module name: {source}")
        module = importlib.import_module(source)
    try:
        factory = getattr(module, attribute)
    except AttributeError as error:
        raise ValueError(
            f"Processor factory {attribute!r} was not found in {source!r}"
        ) from error
    if not callable(factory):
        raise TypeError(f"Processor factory is not callable: {config.factory}")
    processing = factory(checkpoint, task=task, config=config, **config.options)
    if not isinstance(processing, Pi05Processing):
        raise TypeError("Processor factory must return a Pi05Processing instance")
    return processing
