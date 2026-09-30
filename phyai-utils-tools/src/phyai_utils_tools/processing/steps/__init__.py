"""Composable processor steps.

Importing this package registers every step with
:class:`~phyai_utils_tools.processing.pipeline.ProcessorStepRegistry`.

* image: :class:`ResizeWithPadStep`, :class:`NormalizeImageStep`
* normalize: :class:`NormalizerStep`, :class:`UnnormalizerStep`
  (+ :class:`NormalizationMode`, :class:`FeatureType`, :class:`PolicyFeature`)
* text: :class:`TokenizerStep` (generic). pi0.5's state-prompt prep step lives
  in :mod:`phyai_utils_tools.models.pi05.steps_pi05`.
* device: :class:`DeviceStep`
* action: :class:`SliceActionStep`
* lerobot-compat: :class:`RenameObservationsStep`, :class:`AddBatchDimensionStep`
"""

from __future__ import annotations

from phyai_utils_tools.processing.steps.action_steps import (
    DeltaActionStep,
    SliceActionStep,
)
from phyai_utils_tools.processing.steps.batch_steps import AddBatchDimensionStep
from phyai_utils_tools.processing.steps.device_steps import DeviceStep
from phyai_utils_tools.processing.steps.image_steps import (
    NormalizeImageStep,
    ResizeWithPadStep,
)
from phyai_utils_tools.processing.steps.normalize_steps import (
    FeatureType,
    NormalizationMode,
    NormalizerStep,
    PolicyFeature,
    UnnormalizerStep,
)
from phyai_utils_tools.processing.steps.rename_steps import RenameObservationsStep
from phyai_utils_tools.processing.steps.text_steps import TokenizerStep

__all__ = [
    "AddBatchDimensionStep",
    "DeviceStep",
    "DeltaActionStep",
    "FeatureType",
    "NormalizationMode",
    "NormalizeImageStep",
    "NormalizerStep",
    "PolicyFeature",
    "RenameObservationsStep",
    "ResizeWithPadStep",
    "SliceActionStep",
    "TokenizerStep",
    "UnnormalizerStep",
]
