# TimesNet integration and launch

The official https://github.com/thuml/TimesNet repository redirects its code to
https://github.com/thuml/Time-Series-Library. This copy retains the upstream model,
embedding, convolution blocks and MIT license; SOURCE_MANIFEST.json records the
exact revision and file hashes. The shared runner uses the classification head.

The call chain is `scripts/TimesNet.sh` ->
`python -u -m runners.baselines --model TimesNet` ->
`third_party/timesnet/models/TimesNet.py::Model`. The script explicitly lists
one multiline Python command per dataset and seed. Each dataset subshell sets
`CUDA_VISIBLE_DEVICES` and runs seeds 42/43/44 sequentially; dataset blocks
run in parallel. The sourced `scripts/lib/explicit_experiments.sh` handles
run directories, GPU locks, logs and process management without generating
model parameters. The adapter transposes `[B,C,T]` inputs to `[B,T,C]`
and passes an all-ones `[B,T]` padding mask to the official classification head.
It applies softmax to the returned logits and reports window-level Accuracy,
Macro-Precision, Macro-Recall, Macro-F1, Macro-AUROC and Macro-AUPRC. Averaged
window probabilities per subject provide supplementary subject metrics.
The other six baselines continue to use `third_party/medformer/`.

The current launcher records a user-requested retrospective combination of
the screenshot and subsequent sensitivity results: dropout 0.3 on APAVA and
Shimmer10, and 0.1 on TDBRAIN and PADS11. The combination was selected by
comparing reported test Macro-F1; it is not a configuration obtained through
a common validation-tuning procedure. Other integration settings remain unchanged:
subject split seed 42,
initialization seeds 42/43/44, AdamW (learning rate 0.0003, weight decay 0.001),
d_model 128, d_ff 256, two layers, at most 100 epochs, early-stop
warmup 10, patience 12 and min_delta 0.002. Its n_heads=8 setting is accepted but
unused by the TimesNet convolution architecture. Model-specific settings are
top_k=3 and num_kernels=6. The explicit `--checkpoint_metric window_macro_f1`
selects checkpoints using validation window Macro-F1; equal scores retain the
earlier checkpoint. Test is evaluated once after checkpoint restoration.

The four datasets and GPU assignments are TDBRAIN/GPU0, APAVA/GPU1,
Shimmer10/GPU2 and PADS11/GPU3. Batches are 8/8/1/4. Sequence lengths remain
256/256/4096/976; no downsampling is introduced. Three seeds run sequentially per
GPU. ADFTD is excluded from this launcher. tqdm reports each training epoch.
Change GPU exports and hyperparameters directly in the relevant dataset block
and its three commands. To omit a dataset, remove or comment out the complete
subshell block and its associated `PIDS+=("$!")` line; to omit a seed, remove
or comment out the complete Python command, including continuation lines.

## Environment

Follow the repository [environment setup](../../README.md#environment), activate
that environment, and run the commands below from your checkout root:

```bash
conda activate neurosigvia
```

An existing compatible environment can be used instead. TimesNet uses PyTorch
and the shared baseline dependencies; it does not require the main method's
frozen CLIP/Mantis checkpoints. The existing server locations are optional
examples, not requirements for another machine:

| Account | Checkout | Existing environment |
| --- | --- | --- |
| guoyin | `/home/guoyin/NeuroSigViT/main` | `/home/guoyin/.conda/envs/neurosigvit` |
| xzy | `/home/xuzheyuan/guoyin/NeuroSigViT/main` | `/home/xuzheyuan/miniconda3/envs/tivit_env` |

Activate the selected environment by name or full path. For non-interactive
launches, `PYTHON_BIN=/absolute/path/to/env/bin/python` selects its interpreter.

## Verification commands

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*baseline*.py'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -p test_timesnet_integration.py
python -m unittest discover -s tests -p test_explicit_launchers.py
PYTHON_BIN=python DRY_RUN=1 bash scripts/TimesNet.sh
```

Expect protocol tests and TimesNet integration tests to pass, followed by exactly
12 commands, four datasets, three seeds each. The dry-run starts no jobs.

For a GPU smoke test, use an empty result directory and a currently free GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m runners.baselines --model TimesNet --dataset tdbrain --random_seed 42 --split_seed 42 --smoke --smoke_samples_per_class 4 --progress --result_dir results/timesnet_smoke/tdbrain
```

The smoke test uses complete sequences and performs forward/backward and restored
checkpoint validation on a small training/validation row subset. Expected
metrics.json fields: status=COMPLETED, smoke=true, scientific_result=false,
test=null, test_evaluation_count=0. Repeat on APAVA, Shimmer10 and PADS11 before
launch. Smoke outputs must not be merged into scientific results.

## Formal launch

```bash
bash scripts/TimesNet.sh
```

Use a persistent tmux session for the formal launch. The shared runtime checks
GPU availability, acquires the assigned GPU lock for each job and refuses existing result
directories. RUN_TAG may be specified for a unique, traceable run name. Status,
logs, checkpoints, source hashes, args, split audits, training history, test
predictions and the six window-level test metrics use the existing layout;
subject-level metrics remain supplementary.
