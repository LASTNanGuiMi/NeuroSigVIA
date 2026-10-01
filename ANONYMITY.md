# Anonymous artifact checklist

Run the verification commands in `README.md` before preparing a review
artifact, then inspect the clean export for machine-specific paths and identity
disclosures.

The runtime writes sanitized arguments: absolute machine paths are replaced by
`<ABSOLUTE_PATH>/<basename>` in `args.json` and feature-cache signatures. The
default experiment wrappers pass relative paths, so dry-run commands are also
machine-independent.

Do not include local datasets, symbolic links used to reach them, results,
feature caches, checkpoints, logs, environment files, or Git metadata in the
review archive. Git metadata contains development history and local repository
configuration even when the checked-out source is anonymous. Use a clean export
of the reviewed source files rather than compressing the working directory.

## Path and identity check

Run from the repository root. The command lists every releasable file that
still contains a machine-specific absolute path and must print nothing:

```bash
git ls-files -co --exclude-standard -z -- . ':!ANONYMITY.md' \
  | xargs -0 grep -nIE '/home/|/Users/|/root/|/mnt/|[A-Za-z]:\\\\'
```

Also search the same file list for account names, host names and e-mail
addresses used during development before publishing.

Third-party dataset links, model identifiers, citations, and license attributions
must remain intact; they identify dependencies and prior work, not the submission
authors.
