"""Source attribution for NeuroSigVIA's adaptive granularity component.

Only the categorical region gate and hard straight-through Gumbel selection
are adapted from the pinned TimeMosaic component below. NeuroSigVIA applies
the gate to piecewise-mean waveform candidates before the Activity Graph
layout of Yang et al. (Algorithms 1 and 3, DOI 10.1109/TII.2022.3142315).
Rasterization choices and fusion remain project components. See
SOURCE_NOTES.md for the separate sources and adaptation boundaries.
"""

UPSTREAM_REPOSITORY = "https://github.com/BenchCouncil/TimeMosaic"
UPSTREAM_COMMIT = "214423b7f0b4653d04620814380a9301580285cc"
UPSTREAM_COMPONENT = "models/TimeMosaic.py:AdaptivePatchEmbedding"
ADAPTATION_BOUNDARY = (
    "Only the 16-64-3 categorical gate and hard straight-through Gumbel "
    "selection are adapted from TimeMosaic. Selecting 4/8/16 piecewise-mean "
    "waveforms before rendering is a NeuroSigVIA adaptation. Pair-coverage "
    "ordering and cyclic previous/current/next waveform columns follow "
    "Yang et al. Algorithms 1 and 3 (10.1109/TII.2022.3142315). Plot bounds, "
    "differentiable rasterization and fusion are project choices; this "
    "does not reproduce the complete AGCNN model."
)


__all__ = [
    "ADAPTATION_BOUNDARY",
    "UPSTREAM_COMMIT",
    "UPSTREAM_COMPONENT",
    "UPSTREAM_REPOSITORY",
]
