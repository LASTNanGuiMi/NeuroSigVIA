"""四数据集仅视觉消融：复用旧缓存，窗口级验证选择，新建分类器。"""
import argparse
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['apava','tdbrain','shimmer10','pads11'], required=True)
    p.add_argument('--seed', type=int, choices=[42, 43, 44], required=True)
    for name in ['output', 'reference-run', 'vit-1-name']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--checkpoint-metric', choices=['window_macro_f1'], required=True)
    p.add_argument('--preflight-only', action='store_true')
    floats = {'mlp-lr': 3e-4, 'mlp-weight-decay': 1e-3, 'mlp-dropout': .1,
        'granularity-gate-temperature': .5, 'granularity-balance-weight': .001,
        'activity-graph-line-width': 1., 'activity-graph-vertical-margin': .05,
        'mlp-early-stop-min-delta': .002, 'mlp-early-stop-ema-decay': .6,
        'mlp-lr-scheduler-factor': .5, 'mlp-lr-scheduler-min-lr': 1e-6}
    ints = {'batch-size': 8, 'vit-1-layer': 14, 'fusion-dim': 128, 'fusion-heads': 2,
        'mlp-hidden-dim': 128, 'mlp-num-layers': 2, 'mlp-epochs': 100,
        'outer-patch-size': 64, 'outer-patch-stride': 64, 'visual-encode-batch-size': 4,
        'granularity-graph-token-grid': 4, 'activity-graph-canvas-size': 360,
        'mlp-early-stop-warmup-epochs': 10, 'mlp-early-stop-min-epochs': 0,
        'mlp-early-stop-patience': 12, 'mlp-lr-scheduler-patience': 4}
    for name, default in floats.items(): p.add_argument('--' + name, type=float, default=default)
    for name, default in ints.items(): p.add_argument('--' + name, type=int, default=default)
    for name, value in {'aggregation':'mean', 'image-mode':'multiscale_activity_graph',
        'eeg-protocol':'medformer_code_exact',
        'eeg-normalization':'per_window_per_channel_standard_scaler_ddof0',
        'mlp-class-weight':'balanced', 'mlp-early-stop-strategy':'raw_primary',
        'mlp-lr-scheduler':'reduce_on_plateau'}.items():
        p.add_argument('--' + name, choices=[value], default=value)
    return p


def vision_weight_identity(path):
    """与配对主方法 _checkpoint_identity(..., digest_mode='full') 完全相同。"""
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        (x for x in path.rglob('*') if x.is_file()), key=lambda x:x.relative_to(path).as_posix())
    manifest, total = hashlib.sha256(), 0
    for file in files:
        relative = file.name if path.is_file() else file.relative_to(path).as_posix()
        size = file.stat().st_size
        total += size
        manifest.update(relative.encode()); manifest.update(str(size).encode('ascii'))
        manifest.update(sha(file).encode('ascii'))
    return dict(basename=path.name, exists=path.exists(), file_count=len(files),
        total_size=total, full_manifest_sha256=manifest.hexdigest())


def main():
    args=parser().parse_args()
    from src.visual_reference import load_reference
    # This reconstructs labels, subject order and raw patches, not feature embeddings.
    reference=load_reference(args.reference_run,args.seed,args.dataset)
    signature=reference['cache_signature']
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(dataset=args.dataset,random_seed=args.seed,patch_checkpoint_metric=args.checkpoint_metric,
        patch_alignment_weight=0.,mantis=False,moment=None,visual_only=True,
        class_names=[str(x) for x in reference['classes'].tolist()],
        vision_source_identity=signature['vit_1_identity'])
    # Preserve static-renderer, encoder, optimizer and batch settings. The explicit
    # window metric follows current scripts; historical cache training used subject F1.
    for key in config:
        if key in reference['args'] and key not in {'dataset','random_seed','patch_checkpoint_metric','patch_alignment_weight','mantis','moment','vit_1_name'}:
            need(config[key]==reference['args'][key], 'Historical feature/training argument changed: '+key)
    need(vision_weight_identity(args.vit_1_name)==signature['vit_1_identity'],'Frozen vision weights differ')
    for package in ('torch','torchvision','numpy','open_clip_torch','pillow'):
        need(metadata.version(package)==signature['runtime_identity']['packages'][package],'Cache runtime differs: '+package)
    admission=dict(provenance=reference['provenance'],configuration=config,
        historical_checkpoint_metric=reference['args']['patch_checkpoint_metric'],
        current_checkpoint_metric=args.checkpoint_metric,feature_extraction=False,mantis_array_read=False,
        checkpoint_warm_start=False,static_features_used=['raw_windows','line_tokens','patch_mask','valid_fraction','valid_lengths'])
    print('VISUAL_CACHE_VERIFIED '+json.dumps(admission),flush=True)
    if args.preflight_only:
        print('PREFLIGHT_PASSED; training_started=False; gpu_allocated=False',flush=True)
        return
    import torch
    from src.neurosigvia import get_neurosigvia
    from src.vision_ablation_training import train_vision_only
    need(torch.cuda.is_available(),'Assigned CUDA GPU required')
    need(not args.output.exists(),'Refusing existing experiment output')
    args.output.mkdir(parents=True)
    (args.output/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    print('Loading frozen visual encoder only',flush=True)
    vision=get_neurosigvia(str(args.vit_1_name),args.vit_1_layer,args.aggregation,None,None,image_mode=args.image_mode).to('cuda:0')
    vision.requires_grad_(False);vision.eval()
    contract=signature['static_encoder_contract']
    need(f'{type(vision).__module__}.{type(vision).__qualname__}'==contract['vision_wrapper_class'],'Vision wrapper differs')
    need(getattr(vision,'layer_idx',None)==args.vit_1_layer and getattr(vision,'aggregation',None)==args.aggregation,'Encoder settings differ')
    # 新视觉模型不装载主方法权重，也没有 Mantis、InfoNCE、concat-attn 或零向量占位。
    train_vision_only(reference['splits'],vision,config,args.output,args.seed,'cuda:0')

if __name__=='__main__':main()
