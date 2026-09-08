# Medformer source

Copied directly from the Medformer project, commit `446275f27b713a9f09917a6ba0bc51a18e921597`.
The six model implementations and shared layers are unchanged; SHA-256 values are in `SOURCE_MANIFEST.json`.
Original MIT copyright and license are preserved in `LICENSE`.

`runners/baselines.py` is this study's training adapter. It retains NeuroSigVIA's fixed subject splits, labels, input lengths and normalization, then selects checkpoints by validation subject Macro-F1. It does not copy Medformer's original data loaders or full training loop.

The six corresponding method scripts follow the effective configurations selected by upstream `scripts/classification` (the script blobs at the pinned commit are identical to Medformer main commit `891f65b8a7e77188508fd8c56e10ccdba7fc1e18`). The scripts explicitly select 128 model dimensions, 256 feed-forward dimensions, six encoder layers, learning rate 0.0001, 100 epochs and patience 10; eight heads and dropout 0.1 are inherited upstream defaults that NeuroSigVIA records explicitly. Subject-independent ADFTD/TDBRAIN/APAVA use upstream batches 128/32/32. Medformer also uses the upstream patch lists, augmentations and SWA for those datasets.

Upstream has no Shimmer10 or PADS11 configuration. Their local batches remain 1/4 and Medformer uses the documented fallback patch lengths 2/4/8 with no augmentation. The study adapter retains its own optimizer, loss weighting, scheduler, checkpoint selection and final-test policy, so these runs are upstream-script-aligned configurations rather than reproduced published scores.
