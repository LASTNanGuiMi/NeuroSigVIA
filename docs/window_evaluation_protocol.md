# Window-primary validation and evaluation

The primary metric unit is an original dataset input example/window, **not an
internal 64-point model patch**. Subject IDs still determine the train/validation/
test assignment. No subject is randomly split across the three sets. Subject
probability averaging remains available only as supplementary evaluation.

New defaults select the checkpoint with the highest **validation window
Macro-F1**. The early-stopping monitor and LR scheduler use the same window
score. Equal window F1 retains the earlier epoch. Test is evaluated once after
restoring the validation-selected weights, never used for selection. Main-method
runs now persist both validation/test prediction NPZ files for later verification.

The explicit legacy `subject_macro_f1` option is preserved. Metadata records the
actual selection metric separately from the metric unit used in a report.
Existing data, features, learned model architecture, optimizer hyperparameters,
training seeds 42/43/44 and all old results/checkpoints are unchanged. This fixes
the evaluation/selection unit; it does **not** claim identical full Medformer
hyperparameters, tie handling, or its five-seed 41–45 reproduction protocol.

## Entry points

- Main method: `--patch_checkpoint_metric window_macro_f1` (new CLI default).
- Baselines: `--checkpoint_metric window_macro_f1` (new CLI default).
- Numeric v2: `--checkpoint-metric window_macro_f1` (new CLI default).
- Numeric report: `--metric-level window` (new default); reports must disclose
  whether each input checkpoint was selected by window or subject F1.

ADFTD remains out of scope. This correction does not itself launch or authorize
any dataset rerun; the existing main/baseline shell scripts may list ADFTD, so
do not run a full batch without first choosing the explicit non-ADFTD scope.

## Read-only/CPU regression checks

Activate the environment described in the repository `README.md`; no additional
package installation is needed. Run from the repository root:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_*protocol.py' -v
```

The tests include opposite rankings for window F1 versus subject F1 and verify
that checkpoint choice, early stop and LR scheduling follow the chosen unit.
They use tiny deterministic CPU fixtures, do not load the research test datasets,
and do not overwrite scientific experiment artifacts.

## Legacy result boundary

The 108 old runs each preserved only one best checkpoint, selected with subject
validation F1. In the observed histories, 25 saved epochs do not reach the peak
window validation F1 (16 APAVA, 9 TDBRAIN). Their missing epoch weights cannot be
reconstructed from metrics. The other 83 include ties and still do not prove an
identical window-controlled early-stop/LR trajectory.

`scripts/report_legacy_window_metrics.py` creates a **new** output directory and
refuses existing paths. It reads the fixed historical run list, recomputes saved
prediction metrics where available, and labels every row as old subject-selected
weights. Its window table is a diagnostic report, not a corrected-training result.
