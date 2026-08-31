# MedformerGraph / ATGS

This package implements **Adaptive Temporal Granularity Selection (ATGS)** for
the NeuroSigViT Medformer-inspired graph path.

## Data flow

`TemporalGranularityGraphBank` consumes the complete signal `[B,C,T]`; the
candidate regimes do not crop or change `T`. A temporal patch length `p`
partitions each channel into `ceil(T/p)` patches. Per-patch activity combines
RMS amplitude and mean absolute within-patch variation. The renderer then
computes channel affinity/propagation, performs cross-scale router exchange,
and resizes every result to the same RGB image size.

There are two distinct levels of scale:

- Inside one RGB image, the three patch lengths map to `R/G/B =
  fine/medium/coarse`.
- The default adaptive bank contains three complete RGB regimes: `(1,2,4)` is
  relatively fine, `(2,4,8)` is the legacy/base regime, and `(4,8,16)` is
  relatively coarse. Thus `g1` is not a single fine channel; it is a complete
  three-scale RGB graph whose whole scale set is finer.

Granularity therefore means temporal aggregation length. It does **not** mean
different channel groups, a different final node count, a different input
window, or a trainable GNN. Channel count/order and final image size remain
fixed.

The complete adaptive path is:

1. Render the RGB candidate bank: `[B,C,T] -> [B,K,3,224,224]`.
2. Encode every image sequentially with the same frozen visual backbone and
   the same token aggregation: `[B,K,3,224,224] -> [B,K,D]`.
3. L2-normalize and cache candidates in scale-major order as `[B,K*D]`.
4. Reshape to `[B,K,D]`, learn sample-wise softmax weights `[B,K]`, and return
   `normalize(sum_k w_k * normalize(g_k))` with shape `[B,D]` before multimodal
   fusion.

For the maintained CLIP ViT-H/14 checkpoint, `D=1024`, so the default `K=3`
cache branch has width `3072`. This is an image-plus-shared-ViT path; no
separate GNN projection is required.

The selector belongs after frozen feature extraction; placing a trainable gate
inside the renderer would prevent it from learning in the cached MLP workflow.

## Stability and artifacts

Training fails fast on non-finite candidate features, selector logits,
classifier logits, probabilities, or losses. Evaluation can fall back only
when at least one complete candidate is finite; the chosen candidate and
reason are recorded. The supervised objective adds a small normalized
batch-balance term to discourage global one-regime collapse and a weaker
per-sample entropy term to discourage a permanently uniform soft mixture.

Adaptive runs save an `atgs/<dataset>/` directory containing a CPU-loadable
`atgs_checkpoint.pt`, per-sample train/validation/test diagnostics, a JSON
diagnostic summary, and training history. Diagnostics include weights,
top-1 assignments, normalized entropy, mutual information, and fallback codes.

## Compatibility

`src.med_activity_graph` remains a re-export shim.  The historical names
`MedActitivy_graph`, `MedActivityGraph`, `MedActivityGranularityBank`, and
`AdaptiveGranularitySelector` remain supported.  The CLI value stays
`--image_mode med_activity_graph`, preserving existing commands. Fixed-mode
feature caches retain schema 5 compatibility; adaptive caches intentionally use
the stricter schema 7 provenance signature and invalidate older adaptive keys.
