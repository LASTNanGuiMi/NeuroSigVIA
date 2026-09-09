"""Launch one method using the shared, fixed subject cohort and all its windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SHA = 'bc83d4353ba0922a7494e2d7e8adb45a97286bf9a5b46a7ac41cfc46501e9168'


def main():
    methods = json.loads((ROOT / 'configs/adftd_subject13_methods.json').read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=methods, required=True)
    p.add_argument('--data-root', type=Path, required=True, help='Parent containing ADFTD/')
    p.add_argument('--models-root', type=Path, help='Required for NeuroSigVIA')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    manifest = ROOT / 'configs/ADFTD_subject13_seed42.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != SHA:
        p.error('Frozen cohort manifest hash differs')
    out = args.output.resolve()
    argv = [sys.executable] + methods[args.method]
    argv[argv.index('--random_seed') + 1] = str(args.seed)
    argv += ['--result_dir', str(out)]
    if args.method == 'NeuroSigVIA':
        if args.models_root is None:
            p.error('--models-root is required for NeuroSigVIA')
        argv += ['--data_dir', str(args.data_root.resolve()),
                 '--vit_1_name', str(args.models_root.resolve() / 'CLIP-ViT-H-14-laion2B-s32B-b79K'),
                 '--mantis_name', str(args.models_root.resolve() / 'Mantis-8M'),
                 '--feature_cache_dir', str(out / 'feature_cache')]
    env = os.environ.copy()
    for key in ['NEUROSIGVIA_ADFTD_SUBJECT_QUOTAS_JSON', 'NEUROSIGVIA_ADFTD_SUBSET_FRACTION']:
        env.pop(key, None)
    env.update(NEUROSIGVIA_ADFTD_SUBJECT_SELECTION_MANIFEST=str(manifest),
               NEUROSIGVIA_ADFTD_SUBJECT_SELECTION_MANIFEST_SHA256=SHA,
               NEUROSIGVIA_ADFTD_SUBJECT_SELECTION_SEED='42',
               NEUROSIGVIA_EEG_ROOT=str(args.data_root.resolve()),
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    print(shlex.join(argv), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(argv, cwd=ROOT, env=env))


if __name__ == '__main__':
    main()
