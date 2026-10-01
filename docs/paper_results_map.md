# Paper results map

Status on 2026-10-01. This file records which launcher corresponds to each
group of saved paper results and how closely the launcher matches the runs that
produced them.

The index is `saved_results/provenance.csv`: 369 records, one per
table × method × dataset × seed (four datasets, seeds 42/43/44). Verify it
from the repository root:

```bash
python scripts/check_results_index.py
```

The check confirms that every indexed metric file and model file exists and
that the six test metrics in the index equal the values in the metric files.

## Result groups

| Index table | Methods | Records | Launcher | Agreement with the saved runs |
| --- | --- | ---: | --- | --- |
| Table2a, MainRerun | NeuroSigVIA | 12 + 6 | `scripts/NeuroSigVIA.sh` | Arguments equal the saved `args.json` |
| Fig2 | Medformer, PatchTST, TimesNet | 36 | `scripts/<Model>.sh` | Arguments equal the saved `args.json` (TimesNet TDBRAIN differs only by `--progress`) |
| Fig2 | TeCh | 12 | `scripts/TeCh.sh` | Per-dataset configuration equals the saved `protocol.json` |
| Fig2 | Medformer\*, PatchTST\*, TimesNet\*, TeCh\* | 48 | `scripts/TiViT_FrozenBackbone.sh` | Runner arguments equal the saved `args.json`; the saved Medformer\*/PatchTST\*/TimesNet\* runs read TiViT features extracted once per dataset, the launcher extracts them in each run |
| Table2a | w/o Visual | 12 | `scripts/Ablation_wo_Visual.sh` | Differs, see below |
| Table2a | w/o Numeric | 12 | `scripts/Ablation_wo_Numeric.sh` | Arguments equal the recorded commands |
| BackboneReplace(frozen) | TeCh, PatchTST, Medformer, TimesNet | 48 | `scripts/Ablation_NumericBackbone.sh` | Arguments equal the saved `args.json` |
| Fusion | cross_attn_gate, concat_attn, concat, masked_pretrain | 48 | `scripts/Ablation_Fusion.sh` | 30 commands equal the saved `args.json`; 18 differ as described below |
| Visual | GAF, Adaptive Heatmap, Grayscale Image, Line Plot | 48 | `scripts/Ablation_Imaging.sh` | Arguments equal the saved `args.json` |
| Layer | L1, L3, L24, L32 | 48 | `scripts/Ablation_ViTLayer.sh` | Arguments equal the saved `args.json` |
| FixedGranularity | q=4, q=8, q=16 | 36 | `scripts/Ablation_FixedGranularity.sh` | Arguments equal the saved `args.json` |

In the numeric feature-extractor table, the "CNNs" row is the TimesNet
replacement and the "Transformers" row is the mean of the Medformer, PatchTST
and TeCh replacements.

## Known differences

- **w/o Visual.** The saved runs used the legacy paired mode
  (`--reference-run`), which inherits settings and cached Mantis inputs from a
  completed main-method run; the launcher runs the standalone mode. The
  Shimmer10 and PADS11 saved runs selected checkpoints by subject Macro-F1,
  the launcher uses window Macro-F1, and the saved protocols record
  `mlp_early_stop_ema_decay=0.6` where the launcher passes `0.9`.
- **Fusion, cross_attn_gate and concat_attn on TDBRAIN, Shimmer10 and PADS11.**
  These 18 saved runs came from an earlier implementation on a second machine
  that selected the variant with `--adaptive_fusion_mode` and
  `--disable_patch_alignment`. The launcher selects the same fusion module with
  `--temporal_visual_fusion` and `--disable_alignment`. All shared arguments are
  equal except `pretrain_epochs`, which neither variant uses.
- **Settings that differ from the main method.** The fusion and imaging
  ablations use batch size 4 with window-level checkpoint selection on Shimmer10
  (the reported Line Plot row uses batch size 1; its three batch-4 runs stay in
  the index as `Line Plot (batch 4)`)
  and the default learning rate and dropout on PADS11; the main method uses
  batch size 1 with subject-level selection on Shimmer10 and tuned values on
  PADS11. On Shimmer10 every subject contributes one recording, so window and
  subject Macro-F1 have the same value and the rules differ in the tie-break.
- **Runs from a second machine.** 42 records were produced on another machine
  and copied here.
- **Scope.** `scripts/NeuroSigVIA.sh` and the seven baseline launchers also
  list ADFTD, and launchers exist for Crossformer, FEDformer, Autoformer and
  Transformer; the index contains no record for either.
- **Index contents.** The index stores the original run location and the
  machine name of each record; review these columns before publishing it.
