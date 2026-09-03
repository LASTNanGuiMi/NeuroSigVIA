# Wearable dataset processing protocols

The two maintained clinical datasets are stored under `data/wearable/` and
launched through `scripts/run_wearable_activity_graph.sh`.

| Key | Dataset | Input tensor | Split | Labels | Normalization |
|---|---|---|---|---|---|
| `shimmer10` | `Shimmer_10_session10_AFC` | `(N,6,4096)` | label-stratified subject-level 60/20/20, seed 42 | `HC=0`, `MildPD/ModeratePD=1` | channel mean/std from the training split only |
| `pads11` | `PADS_11_task08_TouchIndex` | `(N,6,976)` | label-stratified subject-level 60/20/20, seed 42 | retain Healthy and Parkinson only: `Healthy=0`, `Parkinson=1`; remove OMD | channel mean/std from the training split only |

Shimmer and PADS use the subject IDs in `Meta/subject_map.csv`; the split audit
written with each run records the original and mapped labels. The fixed
data-split seed is 42 in both loaders. The wrapper's `SEED` variable maps to
`--random_seed` (default
2022) for model initialization, training randomness, and result naming; it does
not replace the fixed split seed.

Run examples:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate neurosigvit
cd NeuroSigViT-main
DRY_RUN=1 bash scripts/run_wearable_activity_graph.sh shimmer10
DRY_RUN=1 bash scripts/run_wearable_activity_graph.sh pads11
```

Remove `DRY_RUN=1` only after checking GPU availability. `GPU`, `SEED`,
`EPOCHS`, `PATIENCE`, `RESULT_DIR`, and `FEATURE_CACHE_DIR` can be overridden as
environment variables. Set `WEARABLE_DATA_ROOT` when the dataset parent is not
`data/wearable/`. Additional `main.py` arguments can be appended after the dataset
key.

The commands assume the `neurosigvit` Conda environment described in
`README.md`. If Miniconda is installed elsewhere, adjust only the first
`source` command locally; do not put that machine-specific path into a
submitted script or document.

Before creating an anonymous artifact, follow `ANONYMITY.md`. Local data links,
results, caches, checkpoints, and Git metadata are runtime-only and must not be
included in the submission archive.
