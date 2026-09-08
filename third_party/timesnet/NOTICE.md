# TimesNet source provenance

The official TimesNet repository (https://github.com/thuml/TimesNet) redirects the
complete code and scripts to https://github.com/thuml/Time-Series-Library.

The vendored Python files and LICENSE are exact, unmodified copies of commit
`4e938a1767106324dd753b2a44832bf870a0252e`. SOURCE_MANIFEST.json records the
upstream URLs, byte counts, and SHA-256 hashes verified against the local copies.

The upstream MIT license and copyright notice are retained in LICENSE.

## Classification integration

The shared runner uses task_name="classification", pred_len=0, label_len=48,
and seq_len equal to the input window length. label_len is unused by the
classification path. The official forward interface accepts B x T x C input
and a B x T padding mask; use ones for fixed, completely observed windows.
The official classification head flattens T x d_model embeddings before the
final class projection. It returns logits, not probabilities.

The host project supplies its existing training, preprocessing, subject split,
seed handling, checkpoint selection, and evaluation protocol. Only the model
implementation and its required embedding/convolution modules are vendored here.

TimesNet-specific settings are top_k=3 and num_kernels=6. top_k=3 is the most
common setting in the pinned official classification script; num_kernels=6 is
the pinned run.py default (the FaceDetection script alone overrides it to 4).
These are integration choices, not claims that every upstream classification
dataset uses one identical configuration.

Sources:

- https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/scripts/classification/TimesNet.sh
- https://github.com/thuml/Time-Series-Library/blob/4e938a1767106324dd753b2a44832bf870a0252e/run.py

The upstream positional embedding has max_len=5000, covering the currently
needed lengths 64, 256, 976, and 4096 without source changes.
