"""从历史多模态 NPZ 懒加载五个视觉字段；不读取数值特征，不重新提取。"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.vision_ablation import VISUAL_STATIC_KEYS, validate_visual_bundle
from src.patch_fusion import make_temporal_patches, PATCH_TAIL_POLICY
from src.adaptive_graph_training import NEUROSIGVIA_CACHE_SCHEMA_VERSION, NEUROSIGVIA_CACHE_ARCHITECTURE

def load_visual_cache(path, labels, signature, *, window_size, stride, expected_channels):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Reuse-only cache missing: {path}; extraction is forbidden')
    with np.load(path, allow_pickle=False) as archive:
        expected=dict(schema_version=NEUROSIGVIA_CACHE_SCHEMA_VERSION,
            architecture=NEUROSIGVIA_CACHE_ARCHITECTURE, cache_signature=signature,
            tail_policy=PATCH_TAIL_POLICY, window_size=window_size, stride=stride)
        for key,value in expected.items():
            if archive[key].item()!=value:
                raise ValueError('Cache metadata mismatch: '+key)
        if not np.array_equal(archive['labels'],labels):
            raise ValueError('Cache labels/order mismatch')
        values={key:torch.from_numpy(archive[key].copy()) for key in VISUAL_STATIC_KEYS}
    validate_visual_bundle(values,expected_channels=expected_channels,window_size=window_size)
    return values

def verify_raw_order(values, loader, window_size, stride):
    # [B,C,T] 重新切成 [B,N,C,64]，逐值验证对应关系；不运行任何特征编码器。
    offset=0
    for batch in DataLoader(loader.dataset,batch_size=64,shuffle=False,num_workers=0):
        temporal=make_temporal_patches(batch[0],window_size=window_size,stride=stride,
            lengths=batch[1] if len(batch)>1 else None)
        expected=dict(raw_windows=temporal.patches.cpu().float(),patch_mask=temporal.patch_mask.cpu(),
            valid_fraction=temporal.valid_fraction.cpu().half(),valid_lengths=temporal.valid_lengths.cpu())
        count=len(batch[0])
        for key,actual in expected.items():
            if not torch.equal(values[key][offset:offset+count],actual):
                raise ValueError('Raw cache identity/order mismatch: '+key)
        offset+=count
    if offset!=len(values['raw_windows']):
        raise ValueError('Cache sample count mismatch')
