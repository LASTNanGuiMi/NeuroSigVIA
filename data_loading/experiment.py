"""Shared fixed-split data loading and integrity checks for NeuroSigVIA baselines."""
import contextlib
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.datautils import get_eeg_medformer_dataloaders, get_wearable_dataloaders

ROOT=Path(__file__).resolve().parents[1]
EEG_ROOT=Path(os.environ.get('NEUROSIGVIA_EEG_ROOT',ROOT/'data'/'eeg')).expanduser()
WEARABLE_ROOT=Path(os.environ.get('NEUROSIGVIA_WEARABLE_ROOT',ROOT/'data'/'wearable')).expanduser()

DATASETS={'adftd':('ADFTD','eeg',None,8),'tdbrain':('TDBRAIN','eeg',None,8),
 'apava':('APAVA','eeg',None,8),'shimmer10':('Shimmer_10_session10_AFC','wearable','shimmer_hc_vs_pd',1),
 'pads11':('PADS_11_task08_TouchIndex','wearable','pads_pd_vs_hc',4)}

def load_data(key,smoke=False):
    name,kind,mode,batch=DATASETS[key]
    # Loaders print aggregates normally. Suppress any incidental source notices.
    with contextlib.redirect_stdout(io.StringIO()):
        if kind=='eeg':
            bundle=get_eeg_medformer_dataloaders(name,SimpleNamespace(data_dir=str(EEG_ROOT),batch_size=batch))
        else:
            bundle=get_wearable_dataloaders(name,SimpleNamespace(data_dir=str(WEARABLE_ROOT),batch_size=batch,wearable_label_mode=mode))
    if smoke:
        for split in ('train','vali','test'):
            old=getattr(bundle,split+'_loader'); labels=getattr(bundle,split+'_labels')
            indices=np.concatenate([np.flatnonzero(labels==c)[:2] for c in np.unique(labels)])
            ds=TensorDataset(old.dataset.tensors[0][indices,:,:256])
            ds.sample_subject_ids=np.asarray(old.dataset.sample_subject_ids)[indices]
            ds.split_name=split
            setattr(bundle,split+'_loader',DataLoader(ds,batch_size=2,shuffle=False,num_workers=0))
            setattr(bundle,split+'_labels',np.asarray(labels)[indices])
    manifest={}; idsets=[]
    for split in ('train','vali','test'):
        loader=getattr(bundle,split+'_loader'); labels=getattr(bundle,split+'_labels')
        x=loader.dataset.tensors[0]; ids=np.asarray(loader.dataset.sample_subject_ids)
        if len(x)!=len(labels) or len(ids)!=len(x): raise ValueError('Sample metadata mismatch')
        h=hashlib.sha256()
        for chunk in x.split(512):
            if not bool(torch.isfinite(chunk).all()): raise ValueError('Non-finite source values')
            h.update(chunk.numpy().tobytes())
        h.update(np.asarray(labels,dtype=np.int64).tobytes())
        h.update('\n'.join(map(str,ids)).encode())
        idsets.append(set(ids.tolist()))
        manifest[split]={'shape':list(x.shape),'classes':len(np.unique(labels)),
                         'subjects':len(idsets[-1]),'data_label_order_sha256':h.hexdigest()}
    overlap=[len(idsets[0]&idsets[1]),len(idsets[0]&idsets[2]),len(idsets[1]&idsets[2])]
    if any(overlap): raise ValueError('Subject overlap')
    manifest['subject_overlap']=overlap
    return bundle,manifest
