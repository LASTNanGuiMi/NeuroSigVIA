#!/usr/bin/env python3
"""Only compare adaptive granularity selection; render and encode once/dataset."""
import argparse,fcntl,gc,json,os,time,traceback
from pathlib import Path
import torch
from experiment_common import (ROOT,DATASETS,VISION_PATH,MANTIS_PATH,load_data,
    write_json,digest_file,canonical_hash,weight_manifest,benchmark_graphs)
from src.neurosigvit import get_neurosigvit
from src.medformer_graph import TemporalGranularityGraphBank
from src import patch_mindts as original
from src.utils import set_random_seed
from selector_host import install_selector_host,SelectorComparisonModel

def source_manifest():
    paths=list((ROOT/'src').rglob('*.py'))
    paths += [p for p in (ROOT/'data_loading').rglob('*') if p.is_file() and p.suffix in {'.py','.csv'}]
    paths+=list((ROOT/'selector_policies').glob('*.py'))
    paths += [ROOT/'selector_host.py',ROOT/'experiment_common.py',Path(__file__)]
    return {'files':{str(path.relative_to(ROOT)):digest_file(path) for path in sorted(paths)},
       'upstream':{'timemosaic':{'url':'https://github.com/BenchCouncil/TimeMosaic','commit':'214423b7f0b4653d04620814380a9301580285cc'},
       'pathformer':{'url':'https://github.com/decisionintelligence/pathformer','commit':'ea85d82932215e171357da47b3bc82d502344758'}},
       'adaptation':'selection-mechanism-only on common projected line query; original ActivityGraph unchanged'}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=('current','timemosaic','pathformer'),required=True)
    p.add_argument('--dataset',choices=tuple(DATASETS),required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--visual-batch-size',type=int,default=16)
    return p.parse_args()

def main():
    args=parse_args(); folder=args.output_root/args.dataset/args.method/f'seed_{args.seed}'
    folder.mkdir(parents=True,exist_ok=True)
    lock=open(folder/'.runner.lock','a+'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic()
    identity={'method':args.method,'dataset':args.dataset,'seed':args.seed,'smoke':args.smoke,
       'source_manifest':source_manifest(),
       'requested_run_configuration':{'epochs':args.epochs,'visual_batch_size':args.visual_batch_size,
           'smoke':args.smoke,'protocol':'selector_only_v1_shared_v5_loss'}}
    def status(phase,**extra):
        write_json(folder/'status.json',{**identity,'phase':phase,'pid':os.getpid(),'updated_at':time.time(),**extra})
        print(json.dumps({'event':'phase','phase':phase,**extra}),flush=True)
    try:
        previous=json.loads((folder/'metrics.json').read_text()) if (folder/'metrics.json').exists() else None
        torch.set_num_threads(2); set_random_seed(args.seed)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        device=torch.device('cuda:0'); torch.cuda.reset_peak_memory_stats()
        if args.smoke: torch.autograd.set_detect_anomaly(True,check_nan=True)
        status('loading'); bundle,data=load_data(args.dataset,args.smoke)
        identity['data']=data; identity['weights_manifest']=weight_manifest()
        if previous and previous.get('status')=='completed':
            if any(previous.get(k)!=v for k,v in identity.items()): raise ValueError('Completed run contract mismatch')
            status('completed'); return
        write_json(folder/'data_manifest.json',data)
        channels=bundle.train_loader.dataset.tensors[0].shape[1]
        timings={'data_load_seconds':time.monotonic()-started,'frontend_pretrain_seconds':0.}
        status('loading_backbones'); begin=time.monotonic()
        vision=get_neurosigvit(model_name=str(VISION_PATH),model_layer=14,aggregation='mean',
            stride=None,patch_size=None,image_mode='med_activity_graph',
            med_activity_patch_lengths=(4,8,16),med_activity_adaptive_granularity=True,
            med_activity_granularity_bank=((4,),(8,),(16,))).to(device)
        # There is deliberately no replacement or mutation of the graph bank.
        if not isinstance(vision.med_activity_granularity_bank,TemporalGranularityGraphBank):
            raise AssertionError('Original ActivityGraph bank must be preserved')
        from mantis.architecture import Mantis8M
        mantis=Mantis8M(device=device).from_pretrained(str(MANTIS_PATH)).to(device)
        vision.requires_grad_(False).eval(); mantis.requires_grad_(False).eval()
        timings['backbone_load_seconds']=time.monotonic()-begin
        graph_proof=benchmark_graphs(vision.med_activity_granularity_bank,bundle.train_loader.dataset.tensors[0],device)
        write_json(folder/'unchanged_graph_benchmark.json',graph_proof)
        # Image/features have NO policy or seed dependency. Content and original
        # extraction source are pinned; use a fresh, common comparison cache.
        feature_source={k:v for k,v in identity['source_manifest']['files'].items() if k.startswith(('src/','data_loading/'))}
        signature=canonical_hash({'schema':'selector_common_features_v1','source':feature_source,
            'data':data,'weights':identity['weights_manifest'],'window':64,'stride':64,
            'bank':[[4],[8],[16]],'vision_layer':14,'aggregation':'mean','dtype':'float16','smoke':args.smoke})
        cache=args.output_root/'feature_cache'/args.dataset/signature[:20]
        cache.mkdir(parents=True,exist_ok=True)
        status('feature_extraction',cache_signature=signature)
        begin=time.monotonic(); cache_hits={}
        with (cache/'.feature.lock').open('a+') as cache_lock:
            fcntl.flock(cache_lock,fcntl.LOCK_EX)
            for split in ('train','vali','test'):
                cache_hits[split]=(cache/f'patch_{split}.npz').is_file()
                original._get_patch_feature_split(split,getattr(bundle,split+'_loader'),getattr(bundle,split+'_labels'),
                    vision,mantis,device,64,64,args.visual_batch_size,cache,signature,3)
                gc.collect()
        timings['feature_extraction_or_load_seconds']=time.monotonic()-begin
        feature_files={p.name:digest_file(p) for p in sorted(cache.glob('patch_*.npz'))}
        if len(feature_files)!=3: raise ValueError('Incomplete common feature cache')
        status('classifier_training')
        install_selector_host(args.method)
        training=dict(channels=channels,device=device,batch_size=2 if args.smoke else DATASETS[args.dataset][3],
            random_seed=args.seed,val_ratio=.2,hidden_dim=128,num_layers=2,dropout=.1,lr=3e-4,
            weight_decay=1e-3,epochs=2 if args.smoke else args.epochs,early_stop_patience=0 if args.smoke else 12,
            fusion_dim=128,fusion_heads=2,alignment_dim=256,alignment_temperature=.1,alignment_weight=.1,
            outer_patch_size=64,outer_patch_stride=64,visual_encode_batch_size=args.visual_batch_size,
            class_weight='balanced',vision_model=vision,mantis_model=mantis,
            feature_cache_dir=cache,feature_cache_signature=signature,granularity_temperature=1.,
            granularity_balance_weight=0.,granularity_entropy_weight=0.,granularity_mix_shrinkage_weight=0.,
            granularity_prior_kl_weight=0.,granularity_usage_floor=0.,granularity_entropy_floor=0.,
            granularity_entropy_ceiling=1.,granularity_router_mode='adaptive_v5',checkpoint_metric='window_macro_f1',
            artifact_dir=folder/'artifacts',router_top_k=1 if args.method=='timemosaic' else 2,
            router_training_noise_std=0.,router_local_weight=.5,router_relation_hidden_dim=16,
            router_relation_residual_scale=.25,router_key_adapter_scale=.1,router_value_adapter_scale=.1,
            router_route_budget_weight=.005,router_load_balance_weight=.005,
            early_stop_strategy='raw_primary',early_stop_min_epochs=0,early_stop_warmup_epochs=0 if args.smoke else 10,
            early_stop_min_delta=.002,lr_scheduler_type='reduce_on_plateau',lr_scheduler_patience=4,
            lr_scheduler_factor=.5,lr_scheduler_min_lr=1e-6)
        serial={k:v for k,v in training.items() if k not in {'vision_model','mantis_model','device','feature_cache_dir','artifact_dir'}}
        serial.update(selector_policy=args.method,selector_only=True,extra_pretraining=False)
        write_json(folder/'training_config.json',serial)
        begin=time.monotonic()
        val,test,_,_=original.train_patch_mindts_classifier(
            train_loader=bundle.train_loader,train_labels=bundle.train_labels,
            val_loader=bundle.vali_loader,val_labels=bundle.vali_labels,
            test_loader=bundle.test_loader,test_labels=bundle.test_labels,**training)
        torch.cuda.synchronize(); timings['classifier_training_and_eval_seconds']=time.monotonic()-begin
        timings['total_wall_seconds']=time.monotonic()-started
        checkpoint=torch.load(folder/'artifacts/patch_mindts_checkpoint.pt',map_location='cpu',weights_only=True)
        head=SelectorComparisonModel(**checkpoint['model_constructor_configuration'])
        head.load_state_dict(checkpoint['model_state_dict'],strict=True)
        selector_specific_parameters=sum(p.numel() for n,p in head.named_parameters() if p.requires_grad and (
            'selector_policy_module' in n or (args.method=='current' and any(part in n for part in
                ('granularity_attention.key_projection','v5_relation_scorer','v5_key_adapter_delta')))))
        common_query_parameters=sum(p.numel() for p in head.granularity_attention.query_projection.parameters() if p.requires_grad)
        complexity={'downstream_trainable_parameters':sum(p.numel() for p in head.parameters() if p.requires_grad),
            'selector_parameters':selector_specific_parameters+common_query_parameters,
            'selector_specific_parameters':selector_specific_parameters,
            'common_query_projection_parameters':common_query_parameters,'frontend_pretrain_parameters':0,
            'frozen_vision_parameters':sum(p.numel() for p in vision.parameters()),
            'mantis_parameters':sum(p.numel() for p in mantis.parameters()),
            'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
        metrics={**identity,'status':'completed','validation_metrics':{k:float(v) for k,v in val.items()},
            'test_metrics':{k:float(v) for k,v in test.items()},'complexity':complexity,'timing':timings,
            'cache_hit_at_entry':cache_hits,'cache_signature':signature,'common_feature_files_sha256':feature_files,
            'initial_common_state_sha256':checkpoint['model_configuration']['initial_common_state_sha256'],
            'selected_epoch':checkpoint.get('selected_epoch'),'model_configuration':checkpoint['model_configuration'],
            'scientific_scope':'adaptive granularity selection only; unchanged graph rendering; shared loss; not complete upstream models'}
        write_json(folder/'metrics.json',metrics); status('completed')
    except BaseException as exc:
        frames=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(exc.__traceback__)]
        status('failed',error_type=type(exc).__name__,frames=frames); raise
    finally: lock.close()

if __name__=='__main__': main()
