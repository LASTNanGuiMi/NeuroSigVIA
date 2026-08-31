"""Backward-compatible imports for the MedformerGraph / ATGS package.

New code should import from :mod:`src.medformer_graph`.  This shim preserves
the historical module path used by existing scripts and external callers.
"""

from src.medformer_graph.renderer import (
    MedActitivy_graph,
    MedActivityGraph,
    MedActivityGranularityBank,
    MedformerGraphRenderer,
    MultiGranularityGraphBank,
    TemporalGranularityGraphBank,
    _inverse_occurrence_row_weights,
    _pair_covering_order,
)


__all__ = [
    "MedformerGraphRenderer",
    "TemporalGranularityGraphBank",
    "MultiGranularityGraphBank",
    "MedActivityGraph",
    "MedActitivy_graph",
    "MedActivityGranularityBank",
    "_inverse_occurrence_row_weights",
    "_pair_covering_order",
]
