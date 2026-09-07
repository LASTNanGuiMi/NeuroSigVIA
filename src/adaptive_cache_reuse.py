"""Promote verified static caches without recomputing frozen encoder features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.data import SequentialSampler

from src.adaptive_graph_training import (
    NEUROSIGVIA_CACHE_ARCHITECTURE, NEUROSIGVIA_CACHE_SCHEMA_VERSION,
    NEUROSIGVIA_STATIC_KEYS, _cache_label_array, _validate_static_bundle,
    save_adaptive_graph_feature_cache,
)
from src.patch_mindts import PATCH_TAIL_POLICY, make_temporal_patches


def _peek_signature(path):
    """Read only the small signature member while discovering candidates."""
    with zipfile.ZipFile(path) as archive:
        with archive.open('cache_signature.npy') as member:
            return str(np.lib.format.read_array(member, allow_pickle=False).item())


def _read_candidate(path, signature, architecture, labels, channels, window_size, stride):
    with np.load(path, allow_pickle=False) as cached:
        expected = {
            'schema_version': NEUROSIGVIA_CACHE_SCHEMA_VERSION,
            'architecture': architecture,
            'cache_signature': signature,
            'tail_policy': PATCH_TAIL_POLICY,
            'window_size': int(window_size),
            'stride': int(stride),
        }
        for key, value in expected.items():
            if key not in cached or cached[key].item() != value:
                raise ValueError(f'cache {key} mismatch: {path.name}')
        if not np.array_equal(cached['labels'], _cache_label_array(labels)):
            raise ValueError(f'cache labels mismatch: {path.name}')
        bundle = {key: torch.from_numpy(cached[key].copy()) for key in NEUROSIGVIA_STATIC_KEYS}
    bundle = _validate_static_bundle(bundle, expected_window_size=window_size,
                                     expected_channels=channels)
    if len(bundle['raw_windows']) != len(labels):
        raise ValueError(f'cache sample count mismatch: {path.name}')
    return bundle


def _verify_raw_against_loader(bundle, loader, window_size, stride):
    """Catch split swaps and stale input by checking every ordered raw window."""
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError('cache reuse requires a sequential loader without drop_last')
    offset = 0
    for batch in loader:
        if len(batch) not in (1, 2):
            raise ValueError('feature loader must yield (signals,) or (signals, lengths)')
        temporal = make_temporal_patches(batch[0], window_size=window_size,
                                        stride=stride, lengths=batch[1] if len(batch) == 2 else None)
        stop = offset + len(batch[0])
        expected = {
            'raw_windows': temporal.patches.detach().cpu().float(),
            'patch_mask': temporal.patch_mask.detach().cpu(),
            'valid_lengths': temporal.valid_lengths.detach().cpu(),
            'valid_fraction': temporal.valid_fraction.detach().cpu().half(),
        }
        for key, value in expected.items():
            if not torch.equal(bundle[key][offset:stop], value):
                raise ValueError(f'cache {key} does not match current ordered input at sample {offset}')
        offset = stop
    if offset != len(bundle['raw_windows']):
        raise ValueError('cache record count does not match the complete loader')


def promote_adaptive_static_caches(
    destination_dir, signature, legacy_signatures, search_root, split_loaders,
    channels, window_size, stride,
):
    """Reuse exact current signatures or explicitly audited legacy signatures.

    ``legacy_signatures`` maps exact serialized signatures to their recorded
    NPZ architecture. The caller must establish source-code equivalence before
    supplying any legacy signature. ``split_loaders`` maps train/vali/test to
    (sequential loader, labels). Original cache files are never modified.
    """
    destination = Path(destination_dir)
    root = Path(search_root).expanduser()
    receipt = {'status': 'cache_miss', 'reused_splits': [],
               'verified_raw_against_loader': False, 'sources': []}
    if not root.is_dir():
        return receipt
    accepted = dict(legacy_signatures)
    accepted[str(signature)] = NEUROSIGVIA_CACHE_ARCHITECTURE
    needed = {name: pair for name, pair in split_loaders.items()
              if pair[0] is not None and not (destination / f'adaptive_graph_{name}.npz').exists()}
    if not needed:
        return dict(receipt, status='destination_present')

    candidates = []
    # Only current and explicitly known historical filenames are considered.
    for filename in ('adaptive_graph_train.npz', 'timemosaic_graph_train.npz'):
        for path in sorted(root.rglob(filename)):
            if path.parent.resolve() == destination.resolve():
                continue
            try:
                recorded = _peek_signature(path)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                continue
            if recorded in accepted:
                candidates.append((recorded != str(signature), path.parent, filename.rsplit('_train', 1)[0], recorded))
    candidates.sort(key=lambda candidate: (candidate[0], str(candidate[1])))
    for name, (loader, labels) in needed.items():
        for _, source_dir, prefix, recorded in candidates:
            source = source_dir / f'{prefix}_{name}.npz'
            if not source.is_file():
                continue
            bundle = _read_candidate(source, recorded, accepted[recorded], labels,
                                     channels, window_size, stride)
            _verify_raw_against_loader(bundle, loader, window_size, stride)
            target = destination / f'adaptive_graph_{name}.npz'
            save_adaptive_graph_feature_cache(target, bundle, labels, signature,
                                               window_size=window_size, stride=stride)
            del bundle
            receipt['reused_splits'].append(name)
            receipt['sources'].append({
                'split': name, 'cache_key': source_dir.name,
                'signature_sha256': hashlib.sha256(recorded.encode()).hexdigest(),
                'architecture': accepted[recorded],
                'source_code_manifest_sha256': json.loads(recorded).get('feature_code_identity', {}).get('manifest_sha256'),
            })
            print(f'Reused verified static cache: split={name}, source_key={source_dir.name}', flush=True)
            break
    if receipt['reused_splits']:
        receipt['status'] = 'reused' if len(receipt['reused_splits']) == len(needed) else 'partially_reused'
        receipt['verified_raw_against_loader'] = True
    return receipt
