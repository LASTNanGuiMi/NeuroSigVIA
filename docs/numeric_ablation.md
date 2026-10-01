# Priority-1 numeric-only ablation: no-fusion v2

Four datasets: Shimmer10, PADS11, APAVA, TDBRAIN. Training seeds: 42, 43, 44;
fixed subject split seed: 42. ADFTD is excluded. The historical feature/config
reference is `NeuroSigVIA_paperAG_s42-43-44_20260908_003550` (completed dual-modality
main method, selected by validation **subject** macro-F1). A new protocol-matched
comparison must use dual and numeric checkpoints both selected by validation
**window** macro-F1; merely reporting old checkpoints at window level is not such
a rerun.

## Intervention and controls

The default `NumericOnlyClassifier` in `src/numeric_ablation.py` now uses:

```text
Frozen Mantis -> original ChannelAttentionPool (128 dimensions)
              -> valid-duration weighted pooling
              -> MLP: Linear(128,128), ReLU, Dropout(0.1), Linear(128,2)
```

No visual feature, zero placeholder or concat_attn module is present in this
model. Renderer, visual encoding, visual cross-attention and alignment are also
absent. This explicitly changes the architecture from the previous v1 ablation,
as requested: removing fusion changes the MLP input from256 to128. Its hidden
width, depth, activations and dropout remain unchanged.

To retain compatible initialization, construction temporarily creates a fresh
untrained full model using the reference config and paired seed, keeps its
channel pool and classifier layers, then replaces only the classifier's first
Linear with the narrower input. Discarded modules are not registered in the
returned numeric model and are never used in its forward or optimizer.

No trained main-model weights initialize the ablation. Static frozen Mantis
features are reused only after input-content, signature, label, patch metadata,
and subject-split verification. The existing data loader reconstructs ordered
subject IDs. Raw numeric content and ordered label digests must match the cache
signature. All three subject sets must be disjoint.

All optimizer, batching, classifier, early-stop and LR scheduler hyperparameters
come from the paired main run's args.json. The selection signal is explicitly
overridden by `--checkpoint-metric window_macro_f1` (the new default), and is shared
by checkpoint saving, early stopping and LR scheduling. Exact window-F1 ties keep
the earliest checkpoint, without subject-F1 or log-loss tie-breaking. The explicit
`--checkpoint-metric subject_macro_f1` option retains the historical lexicographic
subject-F1 / negative subject-log-loss selection. Only classification loss
remains: vision InfoNCE and gate-balancing losses cannot apply without vision.
The channel pool and compatible classifier layers retain their original
initialization; the resized input Linear is freshly initialized. Full shared-head
equality or stepwise identical dropout trajectories are not claimed.
The final test is evaluated once, after validation-only checkpoint selection.
Window-level metrics are computed directly from all held-out windows. The original
subject probability-averaging metrics are retained as supplementary results.
The train/validation/test partition remains subject-disjoint; changing the metric
unit never randomly redistributes one person's windows between these sets.

Report all six **window-level** metrics in the primary table: Accuracy, macro
Precision, macro Recall, macro F1, macro AUROC and macro AUPRC (the code's macro
Average Precision, not trapezoidal PR-AUC). Both paired groups are recomputed from raw
seed results as mean ± sample SD (ddof=1); do not mix with older ddof=0 tables.
Record actual differences, including ties or numeric-only improvements.

## Historical results and checkpoints

New checkpoints are tagged `neurosigvia_numeric_only_no_fusion_v2` and new runs
use the `NeuroSigVIA_NumericOnly_NoFusion` prefix. Existing zero-slot results in
`NeuroSigVIA_NumericOnly_s42-43-44_20260908_043500` remain v1, not v2 evidence.
Their files and checkpoints must not be overwritten or relabeled.

The completed no-fusion run
`NeuroSigVIA_NumericOnly_NoFusion_s42-43-44_20260908_r1` also used subject-level
validation selection. Its existing window metrics remain valid scores of that
historical checkpoint, but must not be called window-selected results. New
protocol, checkpoint, progress, history and summary artifacts explicitly record
the effective selection metric and evaluation level. The collector reads those
fields from each summary/protocol. For missing effective-selection fields it
recovers selection from actual historical protocol descriptions or explicit
`patch_checkpoint_metric` training arguments, searching parent `args.json` only
inside the corresponding run. Absent evidence is marked `unknown`, never assumed
to be subject-selected or claimed to be matched. Mixed selection between the paired
methods is warned and contrasts are descriptive only; mixed selection within a
three-seed group is not averaged.

`ZeroVisualSlotClassifier` retains the old v1 implementation only for historical
checkpoint reconstruction. `numeric_model_from_checkpoint(payload)` dispatches
on the stored v1/v2 architecture and loads strictly; unknown or mismatched
architectures are rejected. The result collector labels the two variants
separately and refuses to average a run containing mixed architectures.

## Invocation

Activate the environment described in the repository `README.md` and run all
commands from the repository root. `PYTHON_BIN` may name another interpreter.

Read-only preflight (GPU memory should be below 500 MiB on GPUs 2,3,4,5):

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
python -m unittest discover -s tests -p test_numeric_ablation.py -v
python -m unittest discover -s tests -p test_window_numeric_protocol.py -v
DRY_RUN=1 bash scripts/NeuroSigVIA_NumericOnly.sh
```

Expect unit tests to pass and the dry run to print 12 commands, each bound to one
of four GPUs. Smoke test: two complete-shape training batches on Shimmer, no
validation/test evaluation, fresh isolated output (do not overwrite a prior run):

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python -m runners.numeric_ablation --reference-run results/NeuroSigVIA_paperAG_s42-43-44_20260908_003550 --dataset shimmer10 --seed 42 --checkpoint-metric window_macro_f1 --output results/NumericOnly_NoFusion_Window_preflight_20260908 --smoke
```

Expect `SMOKE PASSED` and `smoke.json` with passed=true, classifier_input_dim=128,
uses_concat_attn=false and test_evaluations=0. This invocation is documented for
a future authorized preflight; changing the model code does not launch it.
Production (use a persistent tmux session):

```bash
bash scripts/NeuroSigVIA_NumericOnly.sh
```

The default scheduler assigns Shimmer/PADS/APAVA/TDBRAIN to GPUs 2/3/4/5,
respectively; each GPU runs its three seeds sequentially. It refuses occupied
GPUs or existing output paths. Check `status/<run>/`, per-epoch `progress.json`,
tqdm logs under `logs/<run>/`, and `numeric_ablation_summary.json`; a live process
is not a completed result. `best_checkpoint.pt` is preserved after every improved
validation checkpoint. No reference outputs or checkpoints are modified.

After or during production, substitute the printed run tag:

```bash
python scripts/summarize_numeric_ablation.py --run results/RUN_TAG --reference-run results/PAIRED_WINDOW_SELECTED_MAIN_RUN --metric-level window
```

This writes per_seed_metrics.csv, comparison.json, and comparison.md inside the
new run's `report_window/` directory, without changing source summaries or old
root-level reports. `--metric-level subject` produces a separate supplementary
`report_subject/` directory. A custom `--output-dir` is also supported. A mean ± SD
is emitted only when all three seeds for that row exist and share one selection
protocol. Every per-seed row records the actual checkpoint-selection provenance;
the reporting-level flag never changes it.
