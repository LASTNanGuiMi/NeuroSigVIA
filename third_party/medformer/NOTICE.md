# Medformer source

Copied directly from the Medformer project, commit `446275f27b713a9f09917a6ba0bc51a18e921597`.
The six model implementations and shared layers are unchanged; SHA-256 values are in `SOURCE_MANIFEST.json`.
Original MIT copyright and license are preserved in `LICENSE`.

`run_baseline.py` is this study's training adapter. It retains NeuroSigViT's fixed subject splits, labels, input lengths and normalization, then selects checkpoints by validation subject Macro-F1. It does not copy Medformer's original data loaders or training loop.
The method scripts use study-specific initial hyperparameters (128 hidden dimensions, 2 encoder layers, 8 heads, dropout 0.1). Medformer starts with patch lengths 2/4/8 and no augmentation; PatchTST uses 16/8. These are not the original paper's dataset-tuned configurations or published scores. A final comparison should apply the same validation tuning budget to each method.
