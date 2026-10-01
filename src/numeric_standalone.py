"""Explicit numeric-only inputs; no historical result or visual encoder dependency."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.patch_fusion import make_temporal_patches, _extract_mantis_channel_tokens


def add_arguments(parser):
    parser.add_argument('--mantis-name', type=Path)
    integers = dict(outer_patch_size=64, outer_patch_stride=64, encode_batch_size=4,
                    fusion_dim=128, channel_hidden_dim=64, mlp_hidden_dim=128,
                    mlp_num_layers=2, mlp_epochs=100, mlp_early_stop_warmup_epochs=10,
                    mlp_early_stop_min_epochs=0, mlp_early_stop_patience=12,
                    mlp_lr_scheduler_patience=4)
    floats = dict(mlp_dropout=.1, mlp_lr=3e-4, mlp_weight_decay=1e-3,
                  mlp_early_stop_min_delta=.002, mlp_early_stop_ema_decay=.9,
                  mlp_lr_scheduler_factor=.5, mlp_lr_scheduler_min_lr=1e-6)
    for values, kind in ((integers, int), (floats, float)):
        for key, default in values.items():
            parser.add_argument('--'+key.replace('_', '-'), type=kind, default=default)
    parser.add_argument('--mlp-class-weight', choices=['balanced', 'none'], default='balanced')
    parser.add_argument('--mlp-early-stop-strategy', default='raw_primary')
    parser.add_argument('--mlp-lr-scheduler', default='reduce_on_plateau')


@torch.no_grad()
def extract_batch(signals, lengths, encoder, args, device):
    patches = make_temporal_patches(signals, window_size=args.outer_patch_size,
                                    stride=args.outer_patch_stride, lengths=lengths)
    b, n, c, length = patches.patches.shape
    indices = patches.patch_mask.reshape(-1).nonzero(as_tuple=False).flatten().cpu()
    windows = patches.patches.reshape(b*n, c, length).detach().cpu().float()
    valid_lengths = patches.valid_lengths.reshape(-1).cpu()
    tokens = _extract_mantis_channel_tokens(windows[indices], valid_lengths[indices],
                                           encoder, device, args.encode_batch_size).cpu()
    padded = torch.zeros(b*n, c, tokens.shape[-1])
    padded.index_copy_(0, indices, tokens.float())
    return dict(mantis_channel_tokens=padded.reshape(b,n,c,-1).half(),
                patch_mask=patches.patch_mask.cpu(), valid_fraction=patches.valid_fraction.cpu().half())


def load_inputs(args):
    if args.mantis_name is None or args.batch_size is None:
        raise ValueError('Independent mode requires --mantis-name and --batch-size')
    if not args.mantis_name.is_dir():
        raise FileNotFoundError(args.mantis_name)
    if not torch.cuda.is_available():
        raise RuntimeError('this invocation requires an assigned CUDA GPU')
    from data_loading.experiment import load_data
    from mantis.architecture import Mantis8M
    from src.patch_fusion import _labels_to_indices, _map_labels
    bundle, manifest = load_data(args.dataset, smoke=False)
    _, classes, mapping = _labels_to_indices(bundle.train_labels)
    encoder = Mantis8M(device='cuda:0').from_pretrained(str(args.mantis_name)).to('cuda:0')
    encoder.requires_grad_(False).eval()
    splits = {}
    # Smoke extracts only two full-shape train batches and never extracts/evaluates test.
    for split in (('train',) if args.smoke else ('train','vali','test')):
        source = getattr(bundle, split+'_loader')
        loader = DataLoader(source.dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=source.collate_fn)
        batches = []
        for index, batch in enumerate(tqdm(loader, desc='Numeric features '+split)):
            if args.smoke and index >= 2:
                break
            if len(batch) not in (1,2):
                raise ValueError('Expected signals and optional lengths')
            batches.append(extract_batch(batch[0], batch[1] if len(batch)==2 else None,
                                         encoder, args, 'cuda:0'))
        features = {key:torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        count = len(features['patch_mask'])
        features['labels'] = np.asarray(_map_labels(getattr(bundle,split+'_labels')[:count],mapping),dtype=np.int64)
        features['subject_ids'] = np.asarray(source.dataset.sample_subject_ids)[:count].copy()
        splits[split] = features
    del encoder
    torch.cuda.empty_cache()
    cfg = {key:value for key,value in vars(args).items() if key.startswith('mlp_')}
    cfg.update(batch_size=args.batch_size,patch_checkpoint_metric=args.checkpoint_metric)
    shape = splits['train']['mantis_channel_tokens'].shape
    model = dict(numeric_standalone=True,temporal_dim=shape[-1],num_channels=shape[-2],
                 num_classes=len(classes),fusion_dim=args.fusion_dim,channel_hidden_dim=args.channel_hidden_dim,
                 classifier_hidden_dim=args.mlp_hidden_dim,classifier_num_layers=args.mlp_num_layers,dropout=args.mlp_dropout)
    provenance = dict(configuration_source='explicit_cli',reference_summary=None,
                      source_loader_manifest=manifest,mantis_name=str(args.mantis_name.resolve()),
                      outer_patch_size=args.outer_patch_size,outer_patch_stride=args.outer_patch_stride,
                      encode_batch_size=args.encode_batch_size,reference_weights_loaded_into_model=False)
    return dict(args=cfg,splits=splits,classes=classes,model_config=model,provenance=provenance)
