"""phyai.models — per-model definitions, runners, and engine plugins.

Each model lives in its own subpackage (e.g. :mod:`phyai.models.pi05`)
holding its configuration, ``nn.Module`` definitions, runners, scheduler,
and the ``main_*`` plugin module that self-registers with
:class:`phyai.engine.Engine`. :mod:`configuration` holds the shared base
config helpers.

Models are imported by their full path (``from phyai.models.pi05 import
...``) or pulled in implicitly when :mod:`phyai.engine` registers its
plugins; this package intentionally re-exports nothing so importing it
stays free of any heavy model dependency.
"""

from __future__ import annotations

__all__: list[str] = []
