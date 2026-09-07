# Source modules

The modules under `src/` are organized by their function in NeuroSigVIA.
The Medformer baseline and the other five baseline implementations live in
`third_party/medformer/` and are called by `runners/baselines.py`. Its shared
dataset loader is `data_loading/experiment.py`.

## Current adaptive Activity Graph path

`python -m runners.neurosigvia --modal_interaction adaptive_granularity`
selects the current path. Root `main.py` forwards existing queued commands
to this module.
Its modules have the following responsibilities:

| Module | Responsibility |
| --- | --- |
| `temporal_granularity.py` | `AdaptiveGranularityGate`: choose granularity 4, 8 or 16 for each channel-wise 16-sample region; load gate weights and retain their adaptation provenance |
| `adaptive_activity_graph.py` | `AdaptiveActivityGraphRenderer`: construct selected activity maps before channel propagation and render one differentiable Activity Graph per window |
| `line_graph_cross_attention.py` | Use the pooled line-plot feature as Query and the 4 x 4 graph spatial tokens as Key/Value |
| `multimodal_fusion.py` | `AdaptiveGranularityFusionModule`: fuse encoded line/graph features, align visual/temporal window features with InfoNCE, and apply final `concat_attn` fusion |
| `adaptive_graph_training.py` | `NeuroSigVIAClassifier`, static feature caches, training/evaluation and current checkpoint save/load |

See [source and adaptation credits](../SOURCE_NOTES.md) for the gate's
upstream reference and the boundaries of its adaptation.

## Shared modules and retained paths

| Module | Responsibility |
| --- | --- |
| `activity_graph.py` | Graph row-order/weight primitives, fixed MedActivity image rendering and `TemporalGranularityGraphBank` |
| `granularity_selector.py` | `AdaptiveTemporalGranularitySelector` for selecting among already encoded graph candidates in shared fusion paths |
| `neurosigvia.py` | Visual encoder wrappers and image transforms |
| `patch_mindts.py` | Window construction, frozen encoder helpers and cache utilities shared by the current path; also retains Patch-MindTS / Router fusion paths |
| `mlp_classifier.py` | Shared `FusionModule`, MLP classifiers and feature-level training paths |
| `classifier.py` | Classification metrics and conventional classifier helpers |
| `embedding.py` | Shared feature extraction and concatenation |
| `analysis.py`, `mutual_knn.py` | Embedding analysis and representation-alignment measurements |
| `datautils.py` | Dataset loading, subject splits and normalization protocols |
| `arguments.py` | Command-line argument definitions |
| `compatibility.py` | Read existing queued commands and same-architecture checkpoints using the canonical names |
| `provenance.py` | Pinned upstream source and adaptation boundary used by gate metadata |
| `utils.py`, `privacy.py` | Split, resizing, rendering, result-output and runtime-path anonymization helpers |

The feature-level selector and graph bank remain shared implementation
dependencies. They render and encode candidate graphs before selection;
the current region gate selects activity maps before rendering. The
`med_activity_graph` CLI value and existing class names remain available
through these functional modules; there is no `medformer_graph` package or
`med_activity_graph.py` re-export module.

The former root selector-comparison scripts, policy replacements and archived
checkpoint-reproduction utilities have been removed. Existing shared fusion
and rendering paths are retained because the maintained entry points still
import them. Changes to this module layout do not change the current model's
hyperparameters, data protocol or checkpoint tensor names.
