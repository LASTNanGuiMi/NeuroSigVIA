"""Shared data, pretrained-weight, and graph configuration for selector runs.

The experiment protocol is identical to the archived TimeMosaic comparison.
Machine-specific paths are supplied through environment variables so the
public source remains portable:

``NEUROSIGVIT_EEG_ROOT``
    Directory containing the processed ADFTD, TDBRAIN, and APAVA bundles.
``NEUROSIGVIT_AAAI27_ROOT``
    Directory containing the ``AAAI_Data`` folder for Shimmer10 and PADS11.
``NEUROSIGVIT_CLIP_PATH``
    Local CLIP-ViT-H-14 checkpoint directory.
``NEUROSIGVIT_MANTIS_PATH``
    Local Mantis-8M checkpoint directory.
"""
import contextlib,hashlib,io,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from src.datautils import get_eeg_medformer_dataloaders,get_aaai27_dataloaders
ROOT=Path(__file__).resolve().parent
EEG_ROOT=Path(os.environ.get('NEUROSIGVIT_EEG_ROOT',ROOT/'data'/'eeg')).expanduser()
AAAI27_ROOT=Path(os.environ.get('NEUROSIGVIT_AAAI27_ROOT',ROOT/'data'/'Neuro')).expanduser()
VISION_PATH=Path(os.environ.get(
    'NEUROSIGVIT_CLIP_PATH',ROOT/'pretrained'/'CLIP-ViT-H-14-laion2B-s32B-b79K'
)).expanduser()
MANTIS_PATH=Path(os.environ.get(
    'NEUROSIGVIT_MANTIS_PATH',ROOT/'pretrained'/'Mantis-8M'
)).expanduser()

DATASETS={'adftd':('ADFTD','eeg',None,8),'tdbrain':('TDBRAIN','eeg',None,8),
 'apava':('APAVA','eeg',None,8),'shimmer10':('Shimmer_10_session10_AFC','aaai27','shimmer_hc_vs_pd',1),
 'pads11':('PADS_11_task08_TouchIndex','aaai27','pads_pd_vs_hc',4)}

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)

def digest_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def weight_manifest():
    result={}
    for name,root in [('clip',VISION_PATH),('mantis',MANTIS_PATH)]:
        files=[p for p in sorted(root.iterdir()) if p.is_file() and p.suffix in {'.pt','.pth','.bin','.safetensors','.json'}]
        if not any(p.suffix!='.json' for p in files): raise ValueError('Missing pinned pretrained weights')
        result[name]={p.name:digest_file(p) for p in files}
    return result

def load_data(key,smoke=False):
    name,kind,mode,batch=DATASETS[key]
    # Loaders print aggregates normally. Suppress any incidental source notices.
    with contextlib.redirect_stdout(io.StringIO()):
        if kind=='eeg':
            bundle=get_eeg_medformer_dataloaders(name,SimpleNamespace(data_dir=str(EEG_ROOT),batch_size=batch))
        else:
            bundle=get_aaai27_dataloaders(name,SimpleNamespace(data_dir=str(AAAI27_ROOT),batch_size=batch,aaai27_label_mode=mode))
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

def benchmark_graphs(bank,example,device):
    example=example[:1,:,:64].to(device)
    bank.eval()
    with torch.no_grad():
        images=bank(example)
        if images.shape!=(1,3,3,224,224) or not torch.isfinite(images).all():
            raise ValueError('Invalid graph bank')
        if images.min()<0 or images.max()>1: raise ValueError('Image range')
        torch.testing.assert_close(images[:,:,0],images[:,:,1],rtol=0,atol=0)
        torch.testing.assert_close(images[:,:,1],images[:,:,2],rtol=0,atol=0)
        differences=[float((images[:,i]-images[:,j]).abs().mean()) for i,j in [(0,1),(0,2),(1,2)]]
        if not all(v>1e-7 for v in differences): raise ValueError('Duplicated scale candidates')
        for _ in range(2): bank(example)
        torch.cuda.synchronize(); start=time.monotonic()
        for _ in range(10): bank(example)
        torch.cuda.synchronize()
        return {'input':[1,int(example.shape[1]),64],'image_shape':list(images.shape),
                'mean_ms_per_window':(time.monotonic()-start)*100,
                'candidate_pairwise_image_l1':differences}
