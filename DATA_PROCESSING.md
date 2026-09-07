# Wearable dataset processing protocols

The two maintained clinical datasets are stored under `data/wearable/` and
launched together through `scripts/NeuroSigVIA.sh`.

| Key | Dataset | Input tensor | Split | Labels | Normalization |
|---|---|---|---|---|---|
| `shimmer10` | `Shimmer_10_session10_AFC` | `(N,6,4096)` | label-stratified subject-level 60/20/20, seed 42 | `HC=0`, `MildPD/ModeratePD=1` | channel mean/std from the training split only |
| `pads11` | `PADS_11_task08_TouchIndex` | `(N,6,976)` | label-stratified subject-level 60/20/20, seed 42 | retain Healthy and Parkinson only: `Healthy=0`, `Parkinson=1`; remove OMD | channel mean/std from the training split only |

Shimmer and PADS use the subject IDs in `Meta/subject_map.csv`; the split audit
written with each run records the original and mapped labels. The fixed
data-split seed is 42 in both loaders. The current method scripts use training
seeds 42, 43 and 44 by default. Edit `SEEDS` at the top of the method script
to choose training seeds; this does not replace the fixed split seed.

To run one wearable dataset on GPU 0, set `DATASETS="shimmer10"` or
`DATASETS="pads11"` and `GPUS="0"` at the top of `scripts/NeuroSigVIA.sh`,
then run from the repository root:

```bash
conda activate neurosigvia
bash scripts/NeuroSigVIA.sh
```

Choose an available GPU in the script. Training parameters are written in
each method script, with per-dataset batch sizes in `BATCH_SIZES` at the top.
If the compatible dataset wrapper is stored elsewhere, the optional
`WEARABLE_DATA_ROOT` runtime setting can select its path. Results and cache
paths are generated per run tag, training seed and dataset.

The commands use the active `neurosigvia` environment described in `README.md`.

Before creating an anonymous artifact, follow `ANONYMITY.md`. Local data links,
results, caches, checkpoints, and Git metadata are runtime-only and must not be
included in the submission archive.
