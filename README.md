# NeuroSigViT

**NeuroSigViT** is a multimodal framework for EEG and wearable-sensor
time-series classification. Its current TimeMosaic-style path learns the
Activity Graph granularity before graph rendering, combines line-plot and
Activity-Graph visual features with cross-attention, and aligns the resulting
visual representation with frozen Mantis-8M temporal features.

![NeuroSigViT method overview](assets/neurosigvit_method.jpg)

The current reproduction scripts support ADFTD, TDBRAIN, APAVA,
`Shimmer_10_session10_AFC`, and `PADS_11_task08_TouchIndex`.

## Selected protocols

| Key | Dataset | Input | Split and endpoint |
| --- | --- | --- | --- |
| `shimmer10` | `Shimmer_10_session10_AFC` | `(N,6,4096)` | subject-level 69/23/25; HC versus MildPD/ModeratePD |
| `pads11` | `PADS_11_task08_TouchIndex` | `(N,6,976)` | subject-level 280/92/97 source split; Healthy versus Parkinson uses 212/70/73 after excluding OMD |

The selected wearable protocols use a fixed data-split seed of 42. The current
TimeMosaic scripts use model seed 42. To change it, edit `--random_seed` and
the cache/result directory arguments in the selected script. Training-only statistics are used
whenever normalization is required. See `DATA_PROCESSING.md` for the
full protocol.

## Repository layout

```text
NeuroSigViT-main/
|-- main.py
|-- run_selector_comparison.py
|-- selector_host.py
|-- selector_policies/
|   |-- timemosaic.py
|   `-- pathformer.py
|-- experiment_common.py
|-- src/
|   |-- neurosigvit.py
|   |-- med_activity_graph.py
|   |-- medformer_graph/
|   |   `-- timemosaic_adaptive.py
|   |-- line_graph_cross_attention.py
|   |-- timemosaic_patch_pipeline.py
|   |-- timemosaic_graph_training.py
|   |-- patch_mindts.py
|   |-- datautils.py
|   `-- privacy.py
|-- assets/
|   `-- neurosigvit_method.jpg
|-- data/
|   `-- wearable/
|       |-- Shimmer_10_session10_AFC/
|       `-- PADS_11_task08_TouchIndex/
|-- data_loading/
|   `-- split_reference_seed42.csv
|-- scripts/
|   |-- adftd.sh
|   |-- tdbrain.sh
|   |-- apava.sh
|   |-- shimmer10.sh
|   |-- pads11.sh
|   `-- run_timemosaic_graph.sh  # compatibility entry for existing job launchers
|-- reproduction/
|   |-- prepare_assets.py
|   |-- evaluate_checkpoint.py
|   `-- verify_checkpoint_grid.py
|-- DATA_PROCESSING.md
|-- ANONYMITY.md
|-- requirements.txt
`-- LICENSE
```

The local dataset directories and links are runtime inputs. Datasets, model
checkpoints, feature caches, logs, results, backups, and experiment snapshots
remain on the server and are excluded from this source release.

This repository contains both the current pre-render adaptive Activity Graph
path and the archived post-encoding selector comparison. Machine-local dataset
and checkpoint paths for the current method are written directly in each
dataset script as command-line arguments.

## Environment

Create an environment from the repository root:

```bash
conda create -n neurosigvit python=3.11 -y
conda activate neurosigvit
python -m pip install -r requirements.txt
```

The checked server environment uses Python 3.11, PyTorch 2.7.1, CUDA 12.6,
`open_clip_torch` 2.32.0, `mantis-tsfm` 1.0.0, and `transformers` 4.33.3.
Activate `neurosigvit` before running the scripts; the server's default
non-interactive `python` is not the training environment.

## Checkpoints

The dataset scripts use these existing model entries relative to the repository:

```text
../models/CLIP-ViT-H-14-laion2B-s32B-b79K
../models/Mantis-8M
```

On another machine, edit `--vit_1_name` and `--mantis_name` in the script.

The corresponding public models are
[`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
and [`paris-noah/Mantis-8M`](https://huggingface.co/paris-noah/Mantis-8M).
Archived classifier checkpoints are optional for training. Use
`reproduction/evaluate_checkpoint.py` only when verifying an archived model;
the script creates a fresh feature cache from the selected data and encoders.

The current path writes `timemosaic_graph_checkpoint.pt`. It contains the
adaptive region gate, line-query/graph-key-value cross-attention, temporal
`concat_attn` fusion module, and classifier head. It intentionally does not
contain the frozen OpenCLIP or Mantis weights, so those two encoders must be
available when the checkpoint is used. A checkpoint from the archived
post-encoding selector has a different architecture and cannot be substituted
for this file. The earlier pre-render `concat_mlp` checkpoint schema is also
incompatible with the current `concat_attn` model and must be retrained.

## Data sources

| Dataset | Source |
| --- | --- |
| Shimmer / PDWearML | [IEEE DataPort](https://ieee-dataport.org/documents/pdwearml-leveraging-daily-activities-rapid-free-living-parkinsons-disease-severity), [PDWearML repository](https://github.com/wang-xulong/PDWearML) |
| PADS | [PhysioNet PADS v1.0.0](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) |

The original licenses and access terms apply.

### PADS11 endpoint

The selected PADS experiment uses
`PADS_11_task08_TouchIndex`, the index-finger touching task from the
Parkinson's Disease Smartwatch (PADS) dataset. Each processed sample contains
six wrist inertial channels and 976 time steps. The fixed seed-42 split is
subject-disjoint and contains 280/92/97 source subjects in the
training/validation/test partitions.

For the reported binary clinical endpoint, only Healthy and Parkinson samples
are retained: `Healthy=0` and `Parkinson=1`. Other Movement Disorders (OMD)
samples are excluded without being merged into either class. This leaves
212/70/73 samples, with class counts 47/165, 15/55, and 17/56 for
Healthy/Parkinson in the three partitions. Channel normalization statistics
are fitted on the retained training samples only and then reused for validation
and test data.

Run only this endpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/pads11.sh
```

## Running experiments

### Current pre-render TimeMosaic path

For every 64-sample outer window, the current implementation follows this
sequence:

1. Split every channel into four 16-sample regions and use a hard
   straight-through TimeMosaic-style gate to choose granularity 4, 8, or 16 for
   each region.
2. Apply the selected maps before cross-channel propagation and render exactly
   one adaptive Activity Graph. Render the line plot from the same window in
   parallel.
3. Encode both images with the shared frozen OpenCLIP backbone. Use the pooled
   line feature as Query and the 4 x 4 Activity Graph spatial tokens as 16
   Key/Value tokens in cross-attention.
4. Encode the raw window with frozen Mantis. Apply symmetric within-sample
   `N x N` InfoNCE between the fused visual patch features and temporal patch
   features.
5. Project the same visual and temporal patch features to two branch tokens,
   apply multi-head self-attention over them, flatten the attended tokens, and
   perform valid-window pooling before the classifier MLP. This is the
   repository's existing `concat_attn` interaction; the InfoNCE branch remains
   a training objective.

Only the region classifier and hard Gumbel selection pattern are adapted from
TimeMosaic's
[`models/TimeMosaic.py::AdaptivePatchEmbedding`](https://github.com/BenchCouncil/TimeMosaic/blob/214423b7f0b4653d04620814380a9301580285cc/models/TimeMosaic.py#L59-L144).
Activity-map construction, channel propagation, rendering, cross-attention,
InfoNCE, and classification are NeuroSigViT components; this path does not
embed the complete TimeMosaic forecasting model.

Each dataset has one short script containing a direct `python -u main.py`
command. From the repository root, activate the environment and choose a dataset:

```bash
conda activate neurosigvit

CUDA_VISIBLE_DEVICES=0 bash scripts/adftd.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/tdbrain.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/apava.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/shimmer10.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/pads11.sh
```

Run the selected line on an available GPU. Running all five lines executes the
datasets sequentially. The scripts use the existing `data/eeg/processed` and
`data/wearable` entries; the wearable entry points to the compatible dataset
wrapper. Point these entries at your local datasets and keep the model entries
listed above; no model/data-path exports are required.

The overridden training settings are visible in the selected `.sh` file. Batch sizes are
8 for ADFTD/TDBRAIN/APAVA, 1 for Shimmer10, and 4 for PADS11. Each script uses
seed 42, 100 epochs, and the existing early-stopping/scheduler configuration.
Unspecified options retain the defaults in `src/arguments.py`.

Additional `main.py` arguments may be appended to a script invocation. For
example, append `--mlp_lr 1e-4` to change the learning rate. When changing the
seed, also set `--feature_cache_dir` and `--result_dir` to the intended run
directories; these are explicit paths in the short scripts.

The method-defining values remain fixed: outer window/stride 64/64, adaptive
granularities 4/8/16, a 4 x 4 graph-token grid, line-Q/graph-KV attention,
Mantis patch alignment, and final visual-Mantis `concat_attn` fusion.

This path is selected only by
`--modal_interaction patch_timemosaic_graph`. Do not add
`--med_activity_adaptive_granularity`: that flag activates the historical
post-encoding graph bank and is rejected for the current path.

The existing `run_timemosaic_graph.sh` is retained unchanged for compatibility
with automated jobs. Its dataset-key invocation and dry-run behavior remain
available:

```bash
DRY_RUN=1 bash scripts/run_timemosaic_graph.sh adftd
```

Historical Activity Graph, Patch-MindTS, GPU-waiting, batch-launching, and
Router-v4 analysis scripts have been removed from the current `scripts/`
directory. Their committed versions remain recoverable from Git; historical
worktrees retain their own copies.

### Archived TimeMosaic selector-only configuration (post-encoding)

The archived TimeMosaic experiments used `selector_only_v1_shared_v5_loss`.
This is a controlled adaptation of the hard Gumbel top-1 selector from
[TimeMosaic](https://github.com/BenchCouncil/TimeMosaic/tree/214423b7f0b4653d04620814380a9301580285cc),
not a reproduction of the complete forecasting model. Only the adaptive
granularity selector changes; Activity Graph construction, frozen CLIP and
Mantis encoders, Patch-MindTS fusion, classifier, loss, and data splits remain
shared.

In that archived path, all candidate Activity Graphs are rendered and encoded
before a selector chooses among their features. It is retained for reproducing
the earlier five-dataset comparison and must not be described as the current
pre-render method. Its checkpoints and reported results do not validate the
new `patch_timemosaic_graph` architecture; the current method must be retrained
and evaluated separately.

| Setting | Archived value |
| --- | --- |
| Datasets | ADFTD, TDBRAIN, APAVA, Shimmer10, PADS11 |
| Seeds | 42, 43, 44 |
| Outer window / stride | 64 / 64 |
| Activity Graph candidates | `(4,)`, `(8,)`, `(16,)` |
| TimeMosaic selector | two-layer MLP, hard Gumbel top-1, temperature 0.5 |
| Fusion width / heads | 128 / 2 |
| Classifier | hidden width 128, 2 layers, dropout 0.1 |
| Optimizer | learning rate `3e-4`, weight decay `1e-3`, 100 epochs |
| Early stopping | `raw_primary`, 10-epoch warmup, patience 12, minimum delta 0.002 |
| LR scheduler | ReduceLROnPlateau, patience 4, factor 0.5, minimum LR `1e-6` |
| Batch size | ADFTD/TDBRAIN/APAVA 8; Shimmer10 1; PADS11 4 |

The archived entry is `run_selector_comparison.py`; its asset configuration
and exact runtime values are defined in that module and `experiment_common.py`.
`selector_host.py` installs the selected routing policy into the unchanged
`PatchMindTSFusionModule`; `selector_policies/timemosaic.py` contains the
TimeMosaic-style selector.

### Additional code paths

The source also includes MedActivity image transforms, adaptive granularity
selection, and Patch-MindTS / Router development paths.

| Entry | Purpose |
| --- | --- |
| `src/medformer_graph/timemosaic_adaptive.py` | Pre-render 4/8/16 region gate and differentiable adaptive Activity Graph renderer |
| `src/line_graph_cross_attention.py` | Pooled line Query and Activity Graph spatial Key/Value cross-attention |
| `src/timemosaic_patch_pipeline.py` | Visual-temporal InfoNCE and final `concat_attn` fusion |
| `src/timemosaic_graph_training.py` | Current feature-cache, training, evaluation, and checkpoint path |
| `scripts/adftd.sh`, `tdbrain.sh`, `apava.sh`, `shimmer10.sh`, `pads11.sh` | Direct per-dataset commands for the current method |
| `scripts/run_timemosaic_graph.sh` | Compatibility launcher for existing automated jobs |
| `src/med_activity_graph.py`, `src/medformer_graph/` | MedActivity image transforms and granularity selection |
| `src/patch_mindts.py` | Patch-MindTS and Router implementation |
| `selector_policies/timemosaic.py` | TimeMosaic-style adaptive granularity selector |
| `selector_host.py` | Controlled selector replacement on the shared Patch-MindTS host |
| `run_selector_comparison.py` | Fixed five-dataset selector-only experiment configuration |

See [MedActivity implementation notes](docs/MEDACTIVITY.md) for the image and
feature layouts of the historical transforms. The current scripts contain
their dataset, checkpoint, cache, and output paths directly.

Run scripts from the repository root. Real experiments require the corresponding
local datasets and frozen encoders; this source update does not report new
benchmark results or establish equivalence between development and snapshot runs.

## Verification

```bash
python -m compileall -q main.py src data_loading scripts selector_policies \
  reproduction selector_host.py experiment_common.py run_selector_comparison.py
for script in scripts/*.sh; do bash -n "$script"; done
```

These checks verify that the published Python sources compile and the maintained
shell launchers parse. Dataset and model behavior is verified by running the
selected experiment with the released data and frozen weights.

## Anonymous release

Follow `ANONYMITY.md`. Do not submit datasets, local links, results, feature
caches, checkpoints, logs, environment files, or Git metadata.

## License

Code is released under the terms in [LICENSE](LICENSE). Dataset use remains
subject to the original sources above.
