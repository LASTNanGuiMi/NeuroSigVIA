# Wearable dataset processing protocols

The two maintained clinical datasets are stored under `data/wearable/` and
launched together through `scripts/NeuroSigViT.sh`.

| Key | Dataset | Input tensor | Split | Labels | Normalization |
|---|---|---|---|---|---|
| `shimmer10` | `Shimmer_10_session10_AFC` | `(N,6,4096)` | label-stratified subject-level 60/20/20, seed 42 | `HC=0`, `MildPD/ModeratePD=1` | channel mean/std from the training split only |
| `pads11` | `PADS_11_task08_TouchIndex` | `(N,6,976)` | label-stratified subject-level 60/20/20, seed 42 | retain Healthy and Parkinson only: `Healthy=0`, `Parkinson=1`; remove OMD | channel mean/std from the training split only |

Shimmer and PADS use the subject IDs in `Meta/subject_map.csv`; the split audit
written with each run records the original and mapped labels. The fixed
data-split seed is 42 in both loaders. The current method scripts use training
seeds 42, 43 and 44 by default. Set `SEEDS` to choose training seeds; this
does not replace the fixed split seed.

Run examples:

```bash
conda activate neurosigvit
CUDA_VISIBLE_DEVICES=0 DATASETS=shimmer10 bash scripts/NeuroSigViT.sh
CUDA_VISIBLE_DEVICES=0 DATASETS=pads11 bash scripts/NeuroSigViT.sh
```

Run a selected command from the repository root on an available GPU. Set
`WEARABLE_DATA_ROOT` if the compatible dataset wrapper is stored elsewhere.
Training parameters are visible in each method script. Results and cache paths
are generated per run tag, training seed and dataset.

The commands use the active `neurosigvit` environment described in `README.md`.

Before creating an anonymous artifact, follow `ANONYMITY.md`. Local data links,
results, caches, checkpoints, and Git metadata are runtime-only and must not be
included in the submission archive.
