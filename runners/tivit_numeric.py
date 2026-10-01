"""TiViT with its numeric branch replaced by a Medformer / TimesNet / PatchTST / TeCh encoder.

The vision branch (official TiViT_OpenCLIP, ViT-H-14), per-branch L2
normalization, concatenation and the StandardScaler + LogisticRegression probe
are unchanged from TiViT. Only the Mantis / MOMENT embedding is replaced by the
feature-extraction part of an already trained baseline checkpoint, frozen
without further training (src/tivit_numeric.py). Data come from the fixed NeuroSigVIA
subject splits in data_loading/, the same loader used for the TiViT-only rows,
and every checkpoint must have been trained on exactly this split.

--image-mode selects the visual input with everything else unchanged:
  grayscale  official TiViT: each channel's patches stacked into a grayscale image
  lineplot   NeuroSigVIA stacked multichannel line plot, one image per sample

Reported per seed:
  numeric_head   metrics stored with the baseline checkpoint (its own classifier)
  numeric_probe  TiViT probe on the numeric embedding alone
  vision_probe   TiViT probe on the visual embedding alone
  tivit_numeric  TiViT probe on [TiViT embedding, numeric embedding]
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import warnings

import joblib
import numpy as np
import open_clip
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, TensorDataset

from data_loading.experiment import load_data
from src.neurosigvia import preprocess_stacked_multichannel_lineplot
from src.tivit_numeric import (NUMERIC_MODELS, check_vendor_source, extract_features,
                               load_frozen_encoder, sha)

ROOT = Path(__file__).resolve().parents[1]
TIVIT_UPSTREAM = ROOT / "third_party" / "tivit" / "src"


def _upstream(name):
    """Load one unmodified upstream TiViT module by path; this project already owns the package name ``src``."""
    spec = importlib.util.spec_from_file_location(f"tivit_upstream_{name}", TIVIT_UPSTREAM / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


get_classifier = _upstream("classifier").get_classifier
embed = _upstream("embedding").embed
TiViT_OpenCLIP = _upstream("tivit").TiViT_OpenCLIP
get_patch_size = _upstream("utils").get_patch_size
TIVIT_UPSTREAM_COMMIT = "5faafcd04db4815bdf32a06740f6a85a52f1bff3"
VIT_DIM = 1280
SPLITS = ('train', 'vali', 'test')
IMAGE_MODES = ('grayscale', 'lineplot')


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def code_revision():
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        return dict(commit=head, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return None


def metrics(y, pred, score, classes):
    assert score.shape == (len(y), len(classes))
    assert np.isfinite(score).all() and np.allclose(score.sum(1), 1, atol=1e-5)
    assert set(np.unique(y)) == set(classes)
    return dict(
        accuracy=float(accuracy_score(y, pred)),
        macro_precision=float(precision_score(y, pred, labels=classes, average='macro', zero_division=0)),
        macro_recall=float(recall_score(y, pred, labels=classes, average='macro', zero_division=0)),
        macro_f1=float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)),
        macro_auroc=float(np.mean([roc_auc_score(y == c, score[:, i]) for i, c in enumerate(classes)])),
        macro_auprc=float(np.mean([average_precision_score(y == c, score[:, i]) for i, c in enumerate(classes)])))


def tivit_settings(args, length):
    if args.image_mode == 'grayscale':
        # Same keys as the TiViT-only feature directories, so their features stay reusable.
        return dict(layer=args.layer, aggregation=args.aggregation,
                    patch_size=get_patch_size(args.patch_size, length)[0], stride=args.stride)
    return dict(image_mode='lineplot', renderer='NeuroSigVIA stacked multichannel line plot',
                image_size=224, layer=args.layer, aggregation=args.aggregation)


def vision_dim(args, channels):
    # Grayscale: one TiViT image per channel; line plot: all channels in one image.
    return channels * VIT_DIM if args.image_mode == 'grayscale' else VIT_DIM


@torch.no_grad()
def embed_lineplot(model, block, batch_size):
    outputs = []
    for start in range(0, len(block), batch_size):
        images = preprocess_stacked_multichannel_lineplot(block[start:start + batch_size]).to('cuda')
        hidden = model.forward_vit(images)
        outputs.append(model.aggregate_hidden_representations(hidden, model.aggregation).float().cpu().numpy())
    values = np.concatenate(outputs)
    # Same per-sample L2 normalization TiViT applies to its embeddings.
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    return values


def make_tivit(args, length):
    # Official TiViT OpenCLIP class with the locally staged ViT-H-14 weights.
    full, _, processor = open_clip.create_model_and_transforms('ViT-H-14', pretrained=args.weights)
    model = TiViT_OpenCLIP(processor, full.visual, args.layer, args.aggregation,
                           get_patch_size(args.patch_size, length)[0], args.stride)
    del full
    model.requires_grad_(False)
    assert len(model.vit.transformer.resblocks) == args.layer
    return model.eval().to('cuda')


def load_tivit_features(args, x, data_manifest, length):
    """Reuse frozen TiViT features only if data, settings and file hashes all match."""
    source = Path(args.tivit_features).expanduser().resolve()
    reference = json.loads((source / 'protocol.json').read_text())
    if reference.get('image_mode', 'grayscale') != args.image_mode:
        raise ValueError(f'{source} holds {reference.get("image_mode", "grayscale")} features, not {args.image_mode}')
    expected = tivit_settings(args, length)
    mismatched = {k: (reference.get(k), v) for k, v in expected.items() if reference.get(k) != v}
    if mismatched:
        raise ValueError(f'TiViT feature settings differ: {mismatched}')
    if reference['data_manifest'] != json.loads(json.dumps(data_manifest)):
        raise ValueError('TiViT features were extracted from a different data split or content')
    hashes = json.loads((source / 'feature_sha256.json').read_text())
    features = {}
    for split in SPLITS:
        path = source / (split + '_features.npy')
        if sha(path) != hashes[split]:
            raise ValueError(f'SHA-256 mismatch for {path}')
        features[split] = np.load(path, mmap_mode='r')
        assert features[split].shape == (len(x[split]), vision_dim(args, x[split].shape[1]))
    return features, dict(reused=True, source=str(source), sha256=hashes)


def extract_tivit_features(args, x, data_manifest, length, out):
    channels = x['train'].shape[1]
    dim = vision_dim(args, channels)
    feature_dir = out / 'tivit_features'
    feature_dir.mkdir()
    extra = dict(lineplot_source_sha256=sha(ROOT / 'src' / 'neurosigvia.py')) if args.image_mode == 'lineplot' else {}
    write_json(feature_dir / 'protocol.json', dict(**tivit_settings(args, length), data_manifest=data_manifest,
               weights_sha256=sha(args.weights), backbone='ViT-H-14 laion2b_s32b_b79k', **extra))
    model = make_tivit(args, length)
    features = {}
    for split in SPLITS:
        dst = feature_dir / (split + '_features.npy')
        arr = np.lib.format.open_memmap(dst, mode='w+', dtype=np.float32, shape=(len(x[split]), dim))
        start = time.time()
        # Chunking preserves TiViT's per-channel embedding and L2 normalization.
        for offset in range(0, len(x[split]), args.chunk_size):
            block = x[split][offset:offset + args.chunk_size]
            if args.image_mode == 'grayscale':
                loader = DataLoader(TensorDataset(block), batch_size=args.batch_size, shuffle=False, num_workers=0)
                values = embed(model, loader, 'tivit', channels, 'cuda')
            else:
                values = embed_lineplot(model, block, args.batch_size)
            assert values.shape == (len(block), dim) and np.isfinite(values).all()
            arr[offset:offset + len(block)] = values
            arr.flush()
            progress = dict(dataset=args.dataset, stage='tivit_extraction', split=split,
                            done=offset + len(block), total=len(x[split]), elapsed_seconds=time.time() - start)
            write_json(out / 'progress.json', progress)
            print('EXTRACTION_PROGRESS', json.dumps(progress), flush=True)
        features[split] = arr
    del model
    torch.cuda.empty_cache()
    hashes = {s: sha(feature_dir / (s + '_features.npy')) for s in SPLITS}
    write_json(feature_dir / 'feature_sha256.json', hashes)
    return features, dict(reused=False, source=str(feature_dir), sha256=hashes)


def check_numeric_run(run_dir, args, seed, data_manifest, y, subject_ids):
    """The checkpoint must come from this model, dataset, seed and exactly this data split."""
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    if args.numeric_model == 'TeCh':
        # TeCh runs (runners/tech.py): summary.json, config inside protocol.json.
        status = json.loads((run_dir / 'summary.json').read_text())
        batch_size = int(protocol['config']['batch_size'])
        expected = dict(method='upstream_TeCh_CoTAR', dataset=args.dataset, training_seed=seed)
        subject_key = 'subject_ids'
    else:
        status = json.loads((run_dir / 'metrics.json').read_text())
        batch_size = int(json.loads((run_dir / 'args.json').read_text())['batch_size'])
        expected = dict(model=args.numeric_model, dataset=args.dataset, random_seed=seed, smoke=False)
        subject_key = 'sample_subject_id'
    wrong = {k: (protocol.get(k), v) for k, v in expected.items() if protocol.get(k) != v}
    if wrong or status.get('status') != 'COMPLETED':
        raise ValueError(f'{run_dir} is not a completed {expected} run: {wrong or status.get("status")}')
    if protocol['data_manifest'] != json.loads(json.dumps(data_manifest)):
        raise ValueError(f'{run_dir} was trained on a different data split or content')
    stored = np.load(run_dir / 'validation_predictions.npz', allow_pickle=True)
    if not (np.array_equal(stored['y_true'], y['vali'])
            and np.array_equal(stored[subject_key].astype(str), subject_ids['vali'].astype(str))):
        raise ValueError(f'{run_dir} validation samples are not in the same order')
    return dict(run_dir=str(run_dir), validation=status['validation'], test=status['test'],
                best_epoch=status.get('best_epoch'), batch_size=batch_size,
                checkpoint_sha256=sha(run_dir / 'best_checkpoint.pt'))


def fit_probe(parts, y, seed, seed_out, name, subject_ids):
    train = np.concatenate([np.asarray(p['train']) for p in parts], axis=1)
    clf = get_classifier('logistic_regression', seed)
    fit_start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        clf.fit(train, y['train'])
    del train
    joblib.dump(clf, seed_out / f'{name}_classifier.joblib')
    result = dict(feature_dim=int(clf[0].n_features_in_), fit_seconds=time.time() - fit_start,
                  convergence_warnings=[str(w.message) for w in caught], n_iter=clf[-1].n_iter_.tolist())
    for split, label in (('vali', 'validation'), ('test', 'test')):
        features = np.concatenate([np.asarray(p[split]) for p in parts], axis=1)
        pred, scores = clf.predict(features), clf.predict_proba(features)
        np.savez_compressed(seed_out / f'{name}_{label}_predictions.npz', y_true=y[split], y_pred=pred,
                            y_score=scores, classes=clf.classes_, sample_subject_id=subject_ids[split])
        result[label] = metrics(y[split], pred, scores, clf.classes_)
    return result


def run(args):
    if '{seed}' not in args.numeric_run:
        raise ValueError('--numeric-run must contain {seed}')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    bundle, data_manifest = load_data(args.dataset)
    x = {s: getattr(bundle, s + '_loader').dataset.tensors[0] for s in SPLITS}
    y = {s: np.asarray(getattr(bundle, s + '_labels'), dtype=np.int64).ravel() for s in SPLITS}
    subject_ids = {s: np.asarray(getattr(bundle, s + '_loader').dataset.sample_subject_ids) for s in SPLITS}
    channels, length = int(x['train'].shape[1]), int(x['train'].shape[2])

    protocol = dict(
        method=f'TiViT+{args.numeric_model}', tivit_upstream_commit=TIVIT_UPSTREAM_COMMIT,
        official_source='https://github.com/ExplainableML/TiViT', code_revision=code_revision(),
        backbone='ViT-H-14 laion2b_s32b_b79k', backend='official TiViT_OpenCLIP',
        **{'image_mode': args.image_mode, **tivit_settings(args, length)},
        channels=channels, vision_feature_dim=vision_dim(args, channels),
        numeric_model=args.numeric_model, numeric_run_pattern=args.numeric_run,
        numeric_branch=('feature-extraction part of the trained baseline checkpoint (classification layer '
                        'replaced by identity), frozen, no further training'),
        numeric_batch_size=('batch size of the baseline run, loader order kept; '
                            'TimesNet output depends on batch composition'),
        fusion='concatenation of per-branch L2-normalized embeddings (TiViT late fusion)',
        classifier='StandardScaler + LogisticRegression(max_iter=500)',
        variants=dict(numeric_head='stored metrics of the baseline checkpoint (own classifier)',
                      numeric_probe='TiViT probe on numeric embedding only',
                      vision_probe='TiViT probe on the visual embedding only',
                      tivit_numeric='TiViT probe on [TiViT, numeric] embeddings'),
        data_manifest=data_manifest, metric_unit='window', split_unit='subject',
        split_seed=20260917 if args.dataset == 'apava' else 42,
        classification_seeds=args.seeds,
        seed_controls='which baseline checkpoint is used and LogisticRegression random_state',
        mantis=False, moment=False, activity_graph=False)
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'args.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    if args.tivit_features:
        tivit_features, tivit_source = load_tivit_features(args, x, data_manifest, length)
    else:
        tivit_features, tivit_source = extract_tivit_features(args, x, data_manifest, length, out)
    write_json(out / 'tivit_feature_source.json', tivit_source)

    for seed in args.seeds:
        seed_out = out / f'seed{seed}'
        seed_out.mkdir(exist_ok=False)
        run_dir = Path(args.numeric_run.format(seed=seed)).expanduser().resolve()
        write_json(out / 'progress.json', dict(dataset=args.dataset, stage='numeric_features', seed=seed))
        numeric_head = check_numeric_run(run_dir, args, seed, data_manifest, y, subject_ids)
        vendor_check = check_vendor_source(run_dir, args.numeric_model)
        # The baseline's own evaluation batch size and order (TimesNet output is batch dependent).
        batch_size = numeric_head['batch_size']
        encoder, encoder_info = load_frozen_encoder(run_dir, args.numeric_model, x['vali'], batch_size, 'cuda')
        numeric_features = {s: extract_features(encoder, x[s], batch_size, 'cuda') for s in SPLITS}
        del encoder
        torch.cuda.empty_cache()

        write_json(out / 'progress.json', dict(dataset=args.dataset, stage='probe_fitting', seed=seed))
        result = dict(dataset=args.dataset, numeric_model=args.numeric_model, seed=seed, status='COMPLETED',
                      metric_unit='window', test_evaluation_count=1, tivit_feature_source=tivit_source,
                      numeric_encoder={**encoder_info, 'vendor_sha256_checked': vendor_check},
                      numeric_head=numeric_head)
        result['numeric_probe'] = fit_probe([numeric_features], y, seed, seed_out, 'numeric_probe', subject_ids)
        result['vision_probe'] = fit_probe([tivit_features], y, seed, seed_out, 'vision_probe', subject_ids)
        result['tivit_numeric'] = fit_probe([tivit_features, numeric_features], y, seed, seed_out,
                                            'tivit_numeric', subject_ids)
        write_json(seed_out / 'metrics.json', result)
        print('SEED_COMPLETED', json.dumps(dict(dataset=args.dataset, numeric_model=args.numeric_model, seed=seed,
              image_mode=args.image_mode, test_macro_f1={v: result[v]['test']['macro_f1'] for v in
                                                         ('numeric_head', 'numeric_probe', 'vision_probe', 'tivit_numeric')})),
              flush=True)
    write_json(out / 'progress.json', dict(dataset=args.dataset, stage='COMPLETED', seeds=args.seeds))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['apava', 'tdbrain', 'shimmer10', 'pads11'], required=True)
    p.add_argument('--numeric-model', choices=NUMERIC_MODELS, required=True)
    p.add_argument('--numeric-run', required=True,
                   help='Baseline run directory per seed, with {seed}, e.g. .../Medformer/seed{seed}')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--weights', required=True, help='Local open_clip ViT-H-14 laion2b_s32b_b79k weights')
    p.add_argument('--image-mode', choices=IMAGE_MODES, default='grayscale',
                   help='grayscale: official TiViT per-channel patch image; lineplot: NeuroSigVIA stacked line plot')
    p.add_argument('--layer', type=int, default=14)
    p.add_argument('--aggregation', choices=['mean'], default='mean')
    p.add_argument('--patch-size', choices=['sqrt'], default='sqrt')
    p.add_argument('--stride', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=32, help='TiViT extraction batch size')
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--tivit-features', help='Reuse a TiViT feature directory (protocol.json, *_features.npy)')
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args()


if __name__ == '__main__':
    run(parse_args())
