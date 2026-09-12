# NeuroSigVIA

**NeuroSigVIA** is a multimodal framework for EEG and wearable-sensor
time-series classification. Its adaptive granularity module selects
piecewise-mean waveforms before Activity Graph rendering, combines line-plot and
Activity-Graph visual features with cross-attention, and aligns the resulting
visual representation with frozen Mantis-8M temporal features.

![NeuroSigVIA method overview](assets/neurosigvit_method.jpg)

The current reproduction scripts support ADFTD, TDBRAIN, APAVA,
`Shimmer_10_session10_AFC`, and `PADS_11_task08_TouchIndex`.

## Selected protocols

| Key | Dataset | Input | Split and endpoint |
| --- | --- | --- | --- |
| `shimmer10` | `Shimmer_10_session10_AFC` | `(N,6,4096)` | subject-level 69/23/25; HC versus MildPD/ModeratePD |
| `pads11` | `PADS_11_task08_TouchIndex` | `(N,6,976)` | subject-level 280/92/97 source split; Healthy versus Parkinson uses 212/70/73 after excluding OMD |

The selected wearable protocols use a fixed data-split seed of 42. Each method
script explicitly lists a separate Python command for training seeds 42, 43
and 44, including the matching result and cache paths. Training-only statistics are used
whenever normalization is required. See `DATA_PROCESSING.md` for the
full protocol.

## Repository layout

```text
NeuroSigVIA/
|-- main.py                  # Compatibility launcher for existing queues
|-- runners/
|   |-- __init__.py
|   |-- neurosigvia.py
|   |-- baselines.py
|   `-- numeric_ablation.py
|-- third_party/medformer/
|-- third_party/timesnet/
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
|   |-- datasets.py
|   |-- experiment.py
|   `-- split_reference_seed42.csv
|-- assets/
|   `-- neurosigvit_method.jpg
|-- scripts/
|   |-- NeuroSigVIA.sh
|   |-- Medformer.sh
|   |-- Crossformer.sh
|   |-- FEDformer.sh
|   |-- Autoformer.sh
|   |-- PatchTST.sh
|   |-- Transformer.sh
|   |-- TimesNet.sh
|   |-- NeuroSigVIA_NumericOnly.sh
|   `-- lib/explicit_experiments.sh
|-- tests/
|-- DATA_PROCESSING.md
|-- ANONYMITY.md
|-- SOURCE_NOTES.md
|-- requirements.txt
`-- LICENSE
```

Training implementations live in `runners/`: `neurosigvia.py` runs the current
method, `baselines.py` runs the comparison models, and `numeric_ablation.py`
runs the paired NumericOnly ablation. Shared baseline dataset
loading lives in `data_loading/experiment.py`. The method shell scripts call
these modules with `python -m`; root `main.py` only forwards existing queued
commands to `runners.neurosigvia`.

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

To run only this endpoint, retain the PADS11 dataset block in
`scripts/NeuroSigVIA.sh` and remove or comment out the other complete dataset
blocks, including their associated `PIDS+=("$!")` lines. Set
`export CUDA_VISIBLE_DEVICES=0` inside the retained block, then run:

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
   each region. Replace each selected block with its mean to form a
   piecewise-mean waveform of the original length.
2. Apply Yang et al.'s Algorithm 1 signal order, which covers every unordered
   signal pair in adjacent positions. Algorithm 3 draws the cyclic previous,
   current and next waveforms in three columns at each position. Render one
   `360 x 360` Activity Graph and resize it to `224 x 224` for OpenCLIP.
   Render the line plot from the same window in parallel.
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

The graph layout follows Algorithms 1 and 3 of
[Yang et al. (IEEE TII, 2022)](https://doi.org/10.1109/TII.2022.3142315).
The original rasterization details are not fully specified; plot bounds,
line width and differentiable drawing are explicit implementation choices.
The granularity gate and fusion remain NeuroSigVIA additions, so these
components do not constitute a complete AGCNN reproduction. See
[source and adaptation credits](SOURCE_NOTES.md) for both paper sources and
their implementation boundaries.

Each method script lists every dataset and seed as a separate multiline
`python -u -m ...` command, with one argument per line. A dataset block is a
subshell with its own `export CUDA_VISIBLE_DEVICES`; its seed 42, 43 and 44
commands run sequentially, while dataset blocks run in parallel. NeuroSigVIA
and the six Medformer-family scripts cover all five datasets; TimesNet and
NumericOnly cover the four non-ADFTD datasets:

```bash
conda activate neurosigvia
bash scripts/NeuroSigVIA.sh
```

The seven comparison entries are `Medformer.sh`, `Crossformer.sh`, `FEDformer.sh`, `Autoformer.sh`, `PatchTST.sh`, `Transformer.sh`, and `TimesNet.sh`, all in `scripts/`. Run one method at a time on the chosen GPUs. The first six model implementations are copied unchanged from Medformer into `third_party/medformer/`; TimesNet is copied from the official Time-Series-Library into `third_party/timesnet/`. Both vendor directories retain their MIT license, pinned commit and per-file hashes. Install the additional Medformer import dependency with `python -m pip install -r third_party/medformer/requirements.txt`.

After activating the environment, run TimesNet with:

```bash
bash scripts/TimesNet.sh
```

Its defaults are TDBRAIN/APAVA/Shimmer10/PADS11 on GPUs 0/1/2/3, batch sizes
8/8/1/4, and sequential training seeds 42/43/44 on each GPU. The call chain is
`scripts/TimesNet.sh` -> `runners.baselines` ->
`third_party/timesnet/models/TimesNet.py::Model`. The adapter transposes loader
inputs from `[B,C,T]` to `[B,T,C]`, supplies an all-ones padding mask, and uses
the official classification head. It reports the six window-level metrics;
mean window probabilities per subject provide supplementary subject metrics.
The sourced `scripts/lib/explicit_experiments.sh` handles run directories,
GPU locks, logs and process management; model parameters stay in each command.

The TimesNet script records the user-requested retrospective combination of
the screenshot and subsequent sensitivity runs: dropout 0.3 for APAVA/Shimmer10
and 0.1 for TDBRAIN/PADS11. This combination was chosen by comparing reported
test Macro-F1, not by a shared validation-tuning procedure. Its remaining
configuration is unchanged: split seed 42, `d_model=128`, `d_ff=256`, two layers, AdamW learning rate
`3e-4`, weight decay `1e-3`, at most 100 epochs, and early stopping with warmup
10, patience 12 and `min_delta=0.002`. It uses `top_k=3` and `num_kernels=6`;
`n_heads=8` is accepted but unused by its convolution architecture. See
[TimesNet integration](third_party/timesnet/INTEGRATION.md) for verification and
interpreter examples.

The other six baseline launchers follow the effective configurations selected
by Medformer's `scripts/classification` at pinned source commit `446275f`:
`d_model=128`, `d_ff=256`, six encoder layers, learning rate `1e-4`, at most 100
epochs and patience 10, plus the upstream defaults of eight heads and dropout
0.1. For the overlapping subject-independent datasets, the official batch
sizes are ADFTD/TDBRAIN/APAVA = 128/32/32. Medformer additionally uses the
upstream dataset-specific patch lists, augmentations and epoch-wise SWA. The
upstream directory has no Shimmer10 or PADS11 entries, so those two retain local
batch sizes 1/4 and the Medformer fallback `2,4,8` plus no augmentation; these
fallback values are not claimed as upstream settings.

NeuroSigVIA and the six Medformer-family scripts assign GPUs 0/1/2/3/4 to ADFTD/TDBRAIN/APAVA/Shimmer/PADS respectively. Each dataset processes seeds 42, 43 and 44 sequentially. The command waits for the full method batch; use tmux when disconnecting SSH:

```bash
tmux new-session -s neurosigvia
conda activate neurosigvia
bash scripts/NeuroSigVIA.sh
```

Edit the desired dataset's `export CUDA_VISIBLE_DEVICES` and the explicit
`--batch_size`, learning rate, epochs and early-stop arguments in each of its
three Python commands. To omit a dataset, remove or comment out its complete
subshell block and associated `PIDS+=("$!")` line. To omit a seed, remove or
comment out its entire Python command, including all continuation lines.
For example, to run ADFTD and APAVA seed 42 only, retain those two dataset
blocks and their seed-42 commands, and assign one free GPU to each block.
Dataset blocks remain concurrent; assigning the same GPU does not make them
sequential unless `WAIT_FOR_GPUS=1` is enabled, and then lock acquisition
determines the order.

To inspect the full commands without launching:

```bash
DRY_RUN=1 bash scripts/NeuroSigVIA.sh
```

Normal launches need no parameter prefixes or appended arguments. Optional
runtime settings remain available for command inspection (`DRY_RUN`), choosing
an interpreter (`PYTHON_BIN`), waiting for GPUs (`WAIT_FOR_GPUS`) and naming a
run (`RUN_TAG`). When a selected GPU is occupied, that dataset stops before
training by default; enabling `WAIT_FOR_GPUS` keeps it queued until its GPU is
free, with status `WAITING_FOR_GPU`.

Every launch creates a unique `RUN_TAG`. Results are `results/<RUN_TAG>/seed<SEED>/<DATASET>/`; the current model's training artifacts are under its `adaptive_graph/` subdirectory. Logs and job-status files are in `logs/<RUN_TAG>/` and `status/<RUN_TAG>/`. Existing run tags are rejected. To rerun failed jobs, retain only their explicit commands in the method script and launch again with a fresh run tag; existing checkpoints/results remain intact. Baselines do not use a feature cache.

The five datasets keep their current fixed subject assignments (`split_seed=42`), normalization, labels and full sequence lengths. Training seeds affect initialization and stochastic training. NeuroSigVIA itself keeps batch sizes 8/8/8/1/4 and its existing early stopping; the six Medformer-family launchers use 128/32/32/1/4 and patience 10, while TimesNet keeps 8/8/1/4 on its four datasets. The study adapter uses AdamW, balanced cross entropy, ReduceLROnPlateau, explicitly selected validation window Macro-F1, supplementary mean subject probabilities and one final test evaluation after restoring the best checkpoint. Medformer selection uses its averaged model when `--swa` is enabled. A smoke check is explicitly marked non-scientific and does not evaluate the test set.

The six Medformer-family scripts transfer the upstream classification-shell
settings into NeuroSigVIA's fixed data and evaluation protocol; this is not a
copy of the upstream training loop and not a claim about published scores.
TimesNet remains a separately sourced configuration. Keep hyperparameter changes
in the corresponding method script so the same `bash scripts/<Method>.sh`
command reproduces its saved configuration. Every command spells out its
dataset, seed and output path beneath the shared run-tag directory.

`scripts/NeuroSigVIA_NumericOnly.sh` lists 12 paired commands for
TDBRAIN/APAVA/Shimmer10/PADS11 and seeds 42/43/44. Set `REFERENCE_RUN` to a
completed main-method run with its `status/<RUN_TAG>/manifest.tsv`, the paired
dataset/seed results and compatible feature caches; the script's default is a
historical server run.
The NumericOnly runner inherits model dimensions, training settings and cached
Mantis inputs from each paired reference, initializes a fresh numeric-only
model, and explicitly selects checkpoints by `window_macro_f1`. Its default
GPU exports are Shimmer10=2, PADS11=3, APAVA=4 and TDBRAIN=5; edit each block
to use available GPUs before launching:

```bash
REFERENCE_RUN=results/your_completed_main_run bash scripts/NeuroSigVIA_NumericOnly.sh
```

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

The script searches `feature_cache/` for reusable raw-window, line-plot and
Mantis features. Set `REUSE_STATIC_CACHE_DIR` to search another cache root.
Reuse checks the ordered inputs and splits, encoder checkpoints, software,
and static extraction code; online graph and gate settings are outside this
static cache identity. Two audited historical cache formats can be copied to
the new format after every raw window, validity mask, length and label matches
the current loader. Source caches are retained. Graph-derived embeddings and
old graph-model checkpoints are invalidated by the new graph implementation.

### Source modules

Current method components are organized by function. The Medformer baseline
and the other five baseline implementations are under `third_party/medformer/`.
The official TimesNet model and its required layers are under `third_party/timesnet/`.

| Entry | Purpose |
| --- | --- |
| `runners/neurosigvia.py`, `runners/baselines.py` | Current method and baseline training entry points |
| `data_loading/experiment.py` | Shared baseline data loading and subject-split integrity checks |
| `src/temporal_granularity.py` | Pre-render 4/8/16 region gate and gate checkpoint provenance |
| `src/adaptive_activity_graph.py` | Gate-selected piecewise-mean waveforms rendered with the paper's Activity Graph layout |
| `src/line_graph_cross_attention.py` | Pooled line Query and Activity Graph spatial Key/Value cross-attention |
| `src/multimodal_fusion.py` | Visual-temporal InfoNCE and final `concat_attn` fusion |
| `src/adaptive_graph_training.py` | Current feature-cache, training, evaluation, and checkpoint path |
| `scripts/NeuroSigVIA.sh` and seven baseline method scripts | Explicit commands for three seeds; TimesNet covers four non-ADFTD datasets, other scripts cover five |
| `scripts/NeuroSigVIA_NumericOnly.sh` | Four datasets and three seeds paired with a completed `REFERENCE_RUN` and its cached features |
| `third_party/timesnet/` | Pinned official TimesNet model, required layers, MIT license and integration notes |
| `src/activity_graph.py` | Algorithm 1 signal ordering and Algorithm 3 cyclic three-column waveform rendering |
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
python -m compileall -q main.py runners src data_loading tests third_party/medformer third_party/timesnet
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
