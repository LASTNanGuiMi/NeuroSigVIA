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
for channel-wise 16-sample regions. Each candidate replaces consecutive
blocks of the corresponding length with their mean, preserving the region's
16-sample length. The gate selects one candidate waveform per region before
`AdaptiveActivityGraphRenderer` draws one Activity Graph per input window.
This waveform preparation and the gate's placement before rendering are
NeuroSigVIA adaptations.

Line-query/graph-key-value cross-attention, visual-temporal InfoNCE, final
`concat_attn` fusion and classification are NeuroSigVIA components. The
Activity Graph layout has a separate paper source described below. This
adaptation does not embed the complete
TimeMosaic forecasting model, its embedding, encoder, forecasting head or
auxiliary objective. The upstream repository, commit, component and
adaptation boundary are recorded in `src/provenance.py` and the gate's
provenance metadata.

The removed historical selector-comparison entry selected among already
rendered and encoded graph candidates. Its checkpoints and results do not
establish results for the current pre-render architecture. Historical code
remains available from Git history and existing historical worktrees.

## Activity Graph layout and rendering

The waveform layout in `src/activity_graph.py` implements Algorithms 1 and 3
from P. Yang, C. Yang, V. Lanfranchi and F. Ciravegna,
[*Activity Graph Based Convolutional Neural Network for Human Activity
Recognition Using Acceleration and Gyroscope Data*](https://doi.org/10.1109/TII.2022.3142315),
IEEE Transactions on Industrial Informatics, 18(10), 2022:

- Algorithm 1 extends a deterministic signal order until every unordered
  signal pair occurs in adjacent positions. Tensor indices are zero-based.
- Algorithm 3 draws the previous, current and next signals in three columns
  for each position in that order, with cyclic neighbors at the boundaries.

The paper reports a `360 x 360` image. This implementation draws a grayscale
waveform image, repeats it over RGB channels, and resizes it to `224 x 224`
with bilinear interpolation for the visual encoder. The original plotting
bounds, line width and rasterization details are not fully specified. The
per-signal min/max plot bounds, 0.05 vertical margin, one-pixel nominal line
width measured in the intermediate lane raster and differentiable Gaussian
stroke rasterizer over all valid sample-to-sample segments are explicit engineering
choices in this implementation. Every row is rasterized with at least eight
vertical samples before area reduction to 360, so high channel counts do not
omit entire rows. Pixel-identical reproduction is not claimed.

The adaptive branch supplies its selected piecewise-mean waveforms to this
layout. NeuroSigVIA combines the gate with frozen OpenCLIP/Mantis encoders
and its fusion objectives. This is an implementation of the cited graph
layout within NeuroSigVIA, not a reproduction of the complete AGCNN model and training
pipeline. Fixed rendering and candidate graph banks share the same waveform
layout; optional selector utilities retain their own adaptation boundaries.

## Baseline models

The Medformer, Crossformer, FEDformer, Autoformer, PatchTST and Transformer
implementations and their shared layers were copied from
[DL4mHealth/Medformer](https://github.com/DL4mHealth/Medformer) at commit
`446275f27b713a9f09917a6ba0bc51a18e921597` into `third_party/medformer/`.
Their original model names and MIT attribution remain intact; see
[the source notice](third_party/medformer/NOTICE.md),
[the license](third_party/medformer/LICENSE) and
[the file hashes](third_party/medformer/SOURCE_MANIFEST.json).

`runners/baselines.py` is the study's own training adapter and uses the maintained
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
