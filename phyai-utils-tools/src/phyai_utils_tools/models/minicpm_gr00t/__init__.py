"""MiniCPM-GR00T processor."""

from __future__ import annotations

from phyai_utils_tools.models.minicpm_gr00t.ops_minicpm_gr00t import (
    MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE,
)
from phyai_utils_tools.models.minicpm_gr00t.processor_minicpm_gr00t import (
    MiniCPMGR00TProcessedInputs,
    MiniCPMGR00TProcessor,
    make_minicpm_gr00t_processors,
)

__all__ = [
    "MINICPM_GR00T_DEFAULT_PROMPT_TEMPLATE",
    "MiniCPMGR00TProcessedInputs",
    "MiniCPMGR00TProcessor",
    "make_minicpm_gr00t_processors",
]
