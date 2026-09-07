# Source and adaptation credits

NeuroSigVIA names its current pre-render component **adaptive granularity**.
The following references describe the implementation's upstream sources and
the boundaries of the adaptations.

## Adaptive granularity gate

The categorical region gate in `src/temporal_granularity.py` adapts the
region-classification and hard straight-through Gumbel selection mechanism
from TimeMosaic:

- Repository: [BenchCouncil/TimeMosaic](https://github.com/BenchCouncil/TimeMosaic).
- Pinned commit: `214423b7f0b4653d04620814380a9301580285cc`.
- Reference component: [`models/TimeMosaic.py::AdaptivePatchEmbedding`, lines 59–144](https://github.com/BenchCouncil/TimeMosaic/blob/214423b7f0b4653d04620814380a9301580285cc/models/TimeMosaic.py#L59-L144).
- Adapted mechanism: a `16 -> 64 -> 3` categorical region classifier and
  hard straight-through Gumbel selection. Training uses a one-hot forward
  decision with gradients through the corresponding soft probabilities.

`AdaptiveGranularityGate` chooses Activity Graph granularities `4/8/16`
for channel-wise 16-sample regions. The chosen numeric activity maps are
combined before channel propagation, so `AdaptiveActivityGraphRenderer`
renders one adaptive Activity Graph per input window.

The RMS/variation activity statistics, activity-map construction, channel
propagation, pair-covering row order, image normalization and RGB rendering
are NeuroSigVIA components. Line-query/graph-key-value cross-attention,
visual-temporal InfoNCE, final `concat_attn` fusion and classification are
also NeuroSigVIA components. This adaptation does not embed the complete
TimeMosaic forecasting model, its embedding, encoder, forecasting head or
auxiliary objective. The upstream repository, commit, component and
adaptation boundary are recorded in `src/provenance.py` and the gate's
provenance metadata.

The removed historical selector-comparison entry selected among already
rendered and encoded graph candidates. Its checkpoints and results do not
establish results for the current pre-render architecture. Historical code
remains available from Git history and existing historical worktrees.

## Shared graph and routing components

The fixed renderer and candidate graph bank retain the earlier Medformer-inspired
patching and cross-channel activity design; they are image transforms, not the
Medformer classifier. Shared optional router paths retain per-window decisions,
multiple scale experts, noisy top-k routing and load-balancing patterns originally
described as TimeMosaic- and Pathformer-inspired. These shared utilities are not
complete upstream forecasting models. They remain separate from the current
pre-render adaptive granularity gate.

## Baseline models

The Medformer, Crossformer, FEDformer, Autoformer, PatchTST and Transformer
implementations and their shared layers were copied from
[DL4mHealth/Medformer](https://github.com/DL4mHealth/Medformer) at commit
`446275f27b713a9f09917a6ba0bc51a18e921597` into `third_party/medformer/`.
Their original model names and MIT attribution remain intact; see
[the source notice](third_party/medformer/NOTICE.md),
[the license](third_party/medformer/LICENSE) and
[the file hashes](third_party/medformer/SOURCE_MANIFEST.json).

`run_baseline.py` is the study's own training adapter and uses the maintained
subject splits, labels, sequence lengths and normalization. Its method
scripts provide a shared initial configuration; these settings and resulting
runs must not be presented as original-paper tuned configurations or
published baseline scores.

## Frozen encoders and datasets

The visual and temporal encoders are the public
[`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
and [`paris-noah/Mantis-8M`](https://huggingface.co/paris-noah/Mantis-8M)
models. Their identifiers remain unchanged. Dataset sources and protocol
details are documented in [README.md](README.md) and
[DATA_PROCESSING.md](DATA_PROCESSING.md); their original licenses and access
terms continue to apply.
