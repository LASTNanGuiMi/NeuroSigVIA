# TeCh source

`models/TeCh.py`, `layers/Transformer_EncDec.py` and `layers/Augmentation.py` are
copied from the TeCh project (https://github.com/Levi-Ackman/TeCh), commit
`9a378cc546a5d97c871eff282148175b3c7cd75b`. The model code is unchanged; the files
carry CRLF line endings, which are the bytes every reported TeCh run recorded in
its `protocol.json`. SHA-256 values are in `SOURCE_MANIFEST.json`.

The upstream repository did not declare a license when these files were copied.
Confirm the terms with the TeCh authors before redistributing this directory.

`runners/tech.py` is this study's training adapter. It keeps NeuroSigVIA's fixed
subject splits, labels and normalization and selects checkpoints by validation
window Macro-F1; the per-dataset settings are listed in that file.
