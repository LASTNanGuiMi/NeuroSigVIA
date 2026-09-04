"""MedformerGraph with Adaptive Temporal Granularity Selection (ATGS).

The package has two independent layers:

``renderer``
    Deterministic multichannel graph rendering and the multi-granularity RGB
    candidate bank.

``selector``
    A trainable embedding-level gate that selects candidates after frozen
    vision-feature extraction.

``timemosaic_adaptive``
    The current pre-render path: a TimeMosaic-style gate selects 4/8/16 from
    each raw 16-sample region before one Activity Graph is constructed.

ATGS is the method name used in code and documentation.  Legacy class names
remain exported so existing NeuroSigViT configurations continue to load.
"""

from .renderer import (
    MedActitivy_graph,
    MedActivityGraph,
    MedActivityGranularityBank,
    MedformerGraphRenderer,
    MultiGranularityGraphBank,
    TemporalGranularityGraphBank,
)
from .selector import (
    AdaptiveGranularitySelector,
    AdaptiveTemporalGranularitySelector,
)
from .timemosaic_adaptive import (
    ADAPTATION_VERSION as TIMEMOSAIC_ADAPTATION_VERSION,
    AdaptiveActivityGraphRenderer,
    TimeMosaicAdaptiveActivityGraphRenderer,
    TimeMosaicAdaptiveRenderer,
    TimeMosaicRegionGate,
)


METHOD_NAME = "Adaptive Temporal Granularity Selection"
METHOD_ABBREVIATION = "ATGS"


__all__ = [
    "METHOD_NAME",
    "METHOD_ABBREVIATION",
    "MedformerGraphRenderer",
    "TemporalGranularityGraphBank",
    "MultiGranularityGraphBank",
    "AdaptiveTemporalGranularitySelector",
    "MedActivityGraph",
    "MedActitivy_graph",
    "MedActivityGranularityBank",
    "AdaptiveGranularitySelector",
    "TIMEMOSAIC_ADAPTATION_VERSION",
    "AdaptiveActivityGraphRenderer",
    "TimeMosaicAdaptiveActivityGraphRenderer",
    "TimeMosaicAdaptiveRenderer",
    "TimeMosaicRegionGate",
]
