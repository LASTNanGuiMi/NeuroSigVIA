"""Immutable frozen Heatmap/Mantis cache, separate from historical line caches."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import numpy as np
import torch
from torch.utils.data import SequentialSampler
from tqdm import tqdm

ARCHITECTURE = 'neurosigvia_heatmap_static_v1'

def augment_heatmap_signature(signature, image_representation, heatmap_patch_size):
    metadata = json.loads(signature)
    code = Path(__file__).parent
    metadata['heatmap_ablation'] = {
        'architecture': ARCHITECTURE,
        'image_representation': image_representation,
        'heatmap_patch_size': int(heatmap_patch_size),
        'image_size': 224,
        'normalization': 'per_window_valid_prefix_all_channels_minmax',
        'colormap': 'viridis',
        'visual_storage_key': 'line_tokens_is_heatmap_embedding_compatibility_adapter',
        'source_sha256': {name: hashlib.sha256((code/name).read_bytes()).hexdigest()
                          for name in ['heatmap_ablation.py', 'heatmap_cache.py', 'imaging_ablation.py']},
    }
    from src.imaging_ablation import REPRESENTATIONS, rendering_policy
    if image_representation in REPRESENTATIONS:
        metadata['heatmap_ablation']['rendering'] = rendering_policy(image_representation)
        metadata['heatmap_ablation']['normalization'] = 'see_rendering_policy'
        metadata['heatmap_ablation']['colormap'] = 'see_rendering_policy'
    return json.dumps(metadata, sort_keys=True)

def get_heatmap_static_split(split_name, loader, labels, vision_model, mantis_model,
                            device, *, window_size, stride, encode_batch_size,
                            feature_cache_dir, feature_cache_signature,
                            expected_channels, image_representation, heatmap_patch_size):
    from src.adaptive_graph_training import _validate_static_bundle, _cache_label_array, NEUROSIGVIA_STATIC_KEYS
    from src.heatmap_ablation import extract_heatmap_feature_batch
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError('Heatmap extraction requires sequential input to preserve label order')
    metadata = {
        'architecture': ARCHITECTURE, 'schema_version': 1,
        'signature': str(feature_cache_signature),
        'window_size': int(window_size), 'stride': int(stride),
        'channels': int(expected_channels), 'image_representation': image_representation,
        'heatmap_patch_size': int(heatmap_patch_size),
        'visual_feature_storage_key': 'line_tokens',
        'visual_feature_semantics': 'frozen_heatmap_embedding',
    }
    if feature_cache_dir and not str(feature_cache_signature or '').strip():
        raise ValueError('Signed feature cache required')
    path = (Path(feature_cache_dir)/f'heatmap_{split_name}.npz') if feature_cache_dir else None
    lock = None
    try:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = path.with_suffix('.lock').open('a+')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path and path.exists():
            with np.load(path, allow_pickle=False) as archive:
                if json.loads(str(archive['metadata'].item())) != metadata:
                    raise ValueError(f'Heatmap cache metadata mismatch: {path}')
                if not np.array_equal(archive['labels'], _cache_label_array(labels)):
                    raise ValueError(f'Heatmap cache labels mismatch: {path}')
                bundle = {key: torch.from_numpy(archive[key].copy()) for key in NEUROSIGVIA_STATIC_KEYS}
            bundle = _validate_static_bundle(bundle, expected_window_size=window_size,
                                            expected_channels=expected_channels)
            if len(bundle['raw_windows']) != len(labels):
                raise ValueError('Cached sample count mismatch')
            print(f'Loaded frozen Heatmap/Mantis cache: {path}', flush=True)
            return bundle
        batches = {key: [] for key in NEUROSIGVIA_STATIC_KEYS}
        for batch in tqdm(loader, desc=f'Extract {image_representation} {split_name}', leave=False):
            if len(batch) == 1:
                signals, lengths = batch[0], None
            elif len(batch) == 2:
                signals, lengths = batch
            else:
                raise ValueError('Expected (signals,) or (signals, lengths)')
            features = extract_heatmap_feature_batch(
                signals, vision_model, mantis_model, device,
                window_size=window_size, stride=stride, encode_batch_size=encode_batch_size,
                lengths=lengths, image_representation=image_representation,
                heatmap_patch_size=heatmap_patch_size)
            for key in batches:
                batches[key].append(features[key])
        if not batches['raw_windows']:
            raise ValueError('Empty feature split')
        bundle = _validate_static_bundle(
            {key: torch.cat(value, dim=0) for key,value in batches.items()},
            expected_window_size=window_size, expected_channels=expected_channels)
        if len(bundle['raw_windows']) != len(labels):
            raise ValueError('Extracted sample count mismatch')
        if path:
            temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp.npz')
            try:
                np.savez_compressed(temporary,
                    metadata=np.asarray(json.dumps(metadata,sort_keys=True)),
                    labels=_cache_label_array(labels),
                    **{key: value.detach().cpu().numpy() for key,value in bundle.items()})
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            print(f'Saved frozen Heatmap/Mantis cache: {path}', flush=True)
        return bundle
    finally:
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
