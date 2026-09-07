"""Source attribution for NeuroSigVIA's adaptive granularity component.

Only the categorical region gate and hard straight-through Gumbel selection
are adapted from the pinned TimeMosaic component below. The independent
activity graph and fusion modules are NeuroSigVIA code. See SOURCE_NOTES.md
for the retained provenance and adaptation boundary.
"""

UPSTREAM_REPOSITORY = "https://github.com/BenchCouncil/TimeMosaic"
UPSTREAM_COMMIT = "214423b7f0b4653d04620814380a9301580285cc"
UPSTREAM_COMPONENT = "models/TimeMosaic.py:AdaptivePatchEmbedding"
ADAPTATION_BOUNDARY = (
    "Only the 16-64-3 categorical gate and hard straight-through Gumbel "
    "selection are adapted from TimeMosaic. Activity statistics, graph "
    "propagation, pair-cover ordering, normalization, and rendering are "
    "NeuroSigVIA-specific."
)


__all__ = [
    "ADAPTATION_BOUNDARY",
    "UPSTREAM_COMMIT",
    "UPSTREAM_COMPONENT",
    "UPSTREAM_REPOSITORY",
]
