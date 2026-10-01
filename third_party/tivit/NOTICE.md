# TiViT source

`src/tivit.py`, `src/embedding.py`, `src/classifier.py` and `src/utils.py` are
copied unchanged from TiViT (https://github.com/ExplainableML/TiViT), commit
`5faafcd04db4815bdf32a06740f6a85a52f1bff3`. SHA-256 values are in
`SOURCE_MANIFEST.json`; the original MIT license is preserved in `LICENSE`.

`runners/tivit_numeric.py` loads these modules by path and replaces only the
numeric branch of TiViT with a frozen comparison-method encoder
(`src/tivit_numeric.py`).
