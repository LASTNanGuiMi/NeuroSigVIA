# NeuroSigVIA

**NeuroSigVIA** is a multimodal framework for EEG and wearable-sensor
time-series classification. Its adaptive granularity module learns the
Activity Graph granularity before graph rendering, combines line-plot and
Activity-Graph visual features with cross-attention, and aligns the resulting
visual representation with frozen Mantis-8M temporal features.

The current reproduction scripts support ADFTD, TDBRAIN, APAVA,
`Shimmer_10_session10_AFC`, and `PADS_11_task08_TouchIndex`.

## Selected protocols

| Key | Dataset | Input | Split and endpoint |
| --- | --- | --- | --- |
| `shimmer10` | `Shimmer_10_session10_AFC` | `(N,6,4096)` | subject-level 69/23/25; HC versus MildPD/ModeratePD |
| `pads11` | `PADS_11_task08_TouchIndex` | `(N,6,976)` | subject-level 280/92/97 source split; Healthy versus Parkinson uses 212/70/73 after excluding OMD |

The selected wearable protocols use a fixed data-split seed of 42. The current
method scripts default to training seeds 42, 43 and 44. Edit `SEEDS` at the top
of the method script to select training seeds; output and cache paths follow
automatically. Training-only statistics are used
whenever normalization is required. See `DATA_PROCESSING.md` for the
full protocol.

## Repository layout

```text
NeuroSigVIA-main/
|-- main.py
|-- run_baseline.py
|-- experiment_common.py
|-- third_party/medformer/
|-- src/
|   |-- README.md
|   |-- neurosigvia.py
|   |-- activity_graph.py
|   |-- temporal_granularity.py
|   |-- adaptive_activity_graph.py
|   |-- granularity_selector.py
|   |-- line_graph_cross_attention.py
|   |-- multimodal_fusion.py
|   |-- adaptive_graph_training.py
|   |-- patch_mindts.py
|   |-- mlp_classifier.py
|   |-- classifier.py
|   |-- embedding.py
|   |-- analysis.py
|   |-- mutual_knn.py
|   |-- arguments.py
|   |-- datautils.py
|   |-- utils.py
|   |-- provenance.py
|   `-- privacy.py
|-- data/
|   `-- wearable/
|       |-- Shimmer_10_session10_AFC/
|       `-- PADS_11_task08_TouchIndex/
|-- data_loading/
|   `-- split_reference_seed42.csv
|-- scripts/
|   |-- NeuroSigVIA.sh
|   |-- Medformer.sh
|   |-- Crossformer.sh
|   |-- FEDformer.sh
|   |-- Autoformer.sh
|   |-- PatchTST.sh
|   |-- Transformer.sh
|   `-- lib/experiments.sh
|-- tests/
|-- DATA_PROCESSING.md
|-- ANONYMITY.md
|-- SOURCE_NOTES.md
|-- requirements.txt
`-- LICENSE
```

The local dataset directories and links are runtime inputs. Datasets, model
checkpoints, feature caches, logs, results, backups, and experiment snapshots
remain on the server and are excluded from this source release.

The current pre-render adaptive Activity Graph components are independent
modules under `src/`; see [the source module map](src/README.md). Shared
feature extraction, fusion, classification and data utilities remain available.
Machine-local dataset and checkpoint paths for the current method are written
directly in each method script as command-line arguments.

## Environment

Create an environment from the repository root:

```bash
conda create -n neurosigvia python=3.11 -y
conda activate neurosigvia
python -m pip install -r requirements.txt
```

The checked server environment uses Python 3.11, PyTorch 2.7.1, CUDA 12.6,
`open_clip_torch` 2.32.0, `mantis-tsfm` 1.0.0, and `transformers` 4.33.3.
Activate `neurosigvia` before running the scripts; the server's default
non-interactive `python` is not the training environment.
An existing compatible server environment can continue to be used without
renaming or reinstalling it. In that case, activate that environment in place
of the new environment name used in the examples below.

## Checkpoints

The NeuroSigVIA script uses these existing model entries relative to the repository:

```text
../models/CLIP-ViT-H-14-laion2B-s32B-b79K
../models/Mantis-8M
```

On another machine, edit `--vit_1_name` and `--mantis_name` in the script.

The corresponding public models are
[`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
and [`paris-noah/Mantis-8M`](https://huggingface.co/paris-noah/Mantis-8M).

The current path writes `neurosigvia_checkpoint.pt`. It contains the
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

To run only this endpoint, set `DATASETS="pads11"` and `GPUS="0"` at the top
of `scripts/NeuroSigVIA.sh`, then run:

```bash
conda activate neurosigvia
bash scripts/NeuroSigVIA.sh
```

## Running experiments

### Adaptive granularity path

For every 64-sample outer window, the current implementation follows this
sequence:

1. Split every channel into four 16-sample regions and use a hard
   straight-through categorical gate to choose granularity 4, 8, or 16 for
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

See [source and adaptation credits](SOURCE_NOTES.md) for the gate's upstream
reference and the boundaries of its adaptation.

Each method has one script with a direct Python classification command, following Medformer's method-script layout. Each entry runs all five datasets for training seeds 42, 43 and 44:

```bash
conda activate neurosigvia
bash scripts/NeuroSigVIA.sh
```

The six comparison entries are `Medformer.sh`, `Crossformer.sh`, `FEDformer.sh`, `Autoformer.sh`, `PatchTST.sh`, and `Transformer.sh`, all in `scripts/`. Run one method at a time on the chosen GPUs. Model sources are copied unchanged from Medformer into `third_party/medformer/`, including their MIT license, commit and per-file hashes. Install the additional import dependency with `python -m pip install -r third_party/medformer/requirements.txt`.

Defaults use GPUs 0/1/2/3/4 for ADFTD/TDBRAIN/APAVA/Shimmer/PADS respectively. Each GPU processes seeds 42, 43 and 44 sequentially. The command waits for the full method batch; use tmux when disconnecting SSH:

```bash
tmux new-session -s neurosigvia
conda activate neurosigvia
bash scripts/NeuroSigVIA.sh
```

Set datasets, training seeds, GPU assignments and per-dataset batch sizes in
`DATASETS`, `SEEDS`, `GPUS` and `BATCH_SIZES` at the top of the chosen method
script. For example, use `DATASETS="adftd apava"`, `SEEDS="42"` and `GPUS="0 1"`
there to train those two datasets once. Use `GPUS="0"` to run all selected
datasets sequentially on GPU 0. Training hyperparameters, including learning
rate, epochs and early stopping, are written in the script's Python command;
edit those values in the script before launching.

To inspect the full commands without launching:

```bash
DRY_RUN=1 bash scripts/NeuroSigVIA.sh
```

Normal launches need no parameter prefixes or appended arguments. Optional
runtime settings remain available for command inspection (`DRY_RUN`), choosing
an interpreter (`PYTHON_BIN`), waiting for GPUs (`WAIT_FOR_GPUS`) and naming a
run (`RUN_TAG`). When a selected GPU is occupied, the default exits before
launching; enabling `WAIT_FOR_GPUS` keeps each dataset queued until its GPU is
free, with status `WAITING_FOR_GPU`.

Every launch creates a unique `RUN_TAG`. Results are `results/<RUN_TAG>/seed<SEED>/<DATASET>/`; the current model's training artifacts are under its `adaptive_graph/` subdirectory. Logs and status/manifest files are in `logs/<RUN_TAG>/` and `status/<RUN_TAG>/`. Existing run tags are rejected. To rerun failed jobs, edit `DATASETS` and `SEEDS` in the method script and launch again with a fresh run tag; existing checkpoints/results remain intact. Baselines do not use a feature cache.

The five datasets keep their current fixed subject assignments (`split_seed=42`), normalization, labels and full sequence lengths. Training seeds affect initialization and stochastic training. Defaults are 100 epochs with existing early stopping (warmup 10, patience 12) and batch sizes 8/8/8/1/4. Both trainers select on validation subject Macro-F1. Baselines use AdamW, balanced cross entropy, mean subject probabilities and one final test evaluation after restoring the best checkpoint. A smoke check is explicitly marked non-scientific and does not evaluate the test set.

The six baseline scripts supply a shared starting configuration, not
original-paper tuned settings or completed benchmark results. Keep
hyperparameter changes in the corresponding method script so the same
`bash scripts/<Method>.sh` command reproduces its saved configuration. The
batch scheduler derives output paths from the method, dataset, seed and run tag.

The method-defining values remain fixed: outer window/stride 64/64, adaptive
granularities 4/8/16, a 4 x 4 graph-token grid, line-Q/graph-KV attention,
Mantis patch alignment, and final visual-Mantis `concat_attn` fusion.

This path is selected only by
`--modal_interaction adaptive_granularity`. Do not add
`--med_activity_adaptive_granularity`: that flag activates the historical
post-encoding graph bank and is rejected for the current path.

The script sets `--granularity_gate_temperature`,
`--granularity_balance_weight` and `--granularity_graph_token_grid` directly.
Optional gate loading and freezing use `--granularity_gate_checkpoint` and
`--granularity_freeze_gate`; add these options to the Python command inside
`scripts/NeuroSigVIA.sh` when needed.

### Source modules

Current method components are organized by function. The Medformer baseline
and the other five baseline implementations are under `third_party/medformer/`.

| Entry | Purpose |
| --- | --- |
| `src/temporal_granularity.py` | Pre-render 4/8/16 region gate and gate checkpoint provenance |
| `src/adaptive_activity_graph.py` | Differentiable adaptive Activity Graph renderer |
| `src/line_graph_cross_attention.py` | Pooled line Query and Activity Graph spatial Key/Value cross-attention |
| `src/multimodal_fusion.py` | Visual-temporal InfoNCE and final `concat_attn` fusion |
| `src/adaptive_graph_training.py` | Current feature-cache, training, evaluation, and checkpoint path |
| `scripts/NeuroSigVIA.sh` and six baseline method scripts | Five datasets and three seeds per method |
| `src/activity_graph.py` | Shared graph-rendering primitives, fixed MedActivity transforms and candidate graph banks |
| `src/granularity_selector.py` | Feature-level selector used by shared fusion paths |
| `src/patch_mindts.py`, `src/mlp_classifier.py` | Shared window, encoder and fusion utilities, plus retained Patch-MindTS / Router paths |

The former selector-comparison and archived-checkpoint reproduction entries
have been removed from the current checkout. Their committed versions remain
recoverable from Git; historical worktrees retain their own copies. Shared
modules still imported by the maintained entry points are retained. See
[the source module map](src/README.md) for their boundaries.

Run scripts from the repository root. Real experiments require the corresponding
local datasets and frozen encoders; this source update does not report new
benchmark results or establish equivalence between development and snapshot runs.

## Verification

```bash
python -m compileall -q main.py run_baseline.py experiment_common.py src \
  data_loading tests third_party/medformer
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
