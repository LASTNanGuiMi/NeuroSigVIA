"""MedformerGraph with Adaptive Temporal Granularity Selection (ATGS).

The package has two independent layers:

``renderer``
    Deterministic multichannel graph rendering and the multi-granularity RGB
    candidate bank.

``selector``
    A trainable embedding-level gate that selects candidates after frozen
    vision-feature extraction.

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
]
