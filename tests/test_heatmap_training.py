import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_paper_activity_graph import TinyFrozenVision
from src.adaptive_graph_training import train_neurosigvia_classifier, load_neurosigvia_checkpoint
from src.heatmap_cache import get_heatmap_static_split

class TinyContractVision(TinyFrozenVision):
    def __init__(self):
        super().__init__()
        self.vit._embeds = lambda x: x
        self.vit.transformer = torch.nn.Module()
        self.vit.transformer.resblocks = torch.nn.ModuleList()

def make_bundle():
    return dict(raw_windows=torch.randn(4,2,4,64),line_tokens=torch.randn(4,2,8).half(),
        mantis_channel_tokens=torch.randn(4,2,4,6).half(),
        patch_mask=torch.ones(4,2,dtype=torch.bool),valid_fraction=torch.ones(4,2).half(),
        valid_lengths=torch.full((4,2),64,dtype=torch.long))

class HeatmapTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_full_trainer_test_once_and_checkpoint_reload(self):
        for mode in ['multivariate_heatmap','patch_heatmap']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                vision=TinyContractVision()
                loader=DataLoader(TensorDataset(torch.randn(4,4,128)),batch_size=2)
                labels=np.array([0,1,0,1])
                bundle=make_bundle()
                with patch('src.heatmap_cache.get_heatmap_static_split',return_value=bundle):
                    train_neurosigvia_classifier(
                        loader,labels,loader,labels,4,'cpu',2,42,.2,8,2,0.,.001,.001,2,2,
                        8,2,4,.1,.1,vision_model=vision,mantis_model=torch.nn.Linear(2,2),
                        val_loader=loader,val_labels=labels,selector_balance_weight=0,
                        checkpoint_metric='window_macro_f1',pretrain_epochs=0,
                        artifact_dir=directory,image_representation=mode,heatmap_patch_size=8)
                summary=json.loads((Path(directory)/'adaptive_graph_summary.json').read_text())
                self.assertEqual(summary['test_evaluations'],1)
                self.assertEqual(summary['fusion_input'],'aligned_projected_features')
                self.assertFalse(summary['visual_cross_attention_enabled'])
                self.assertFalse(summary['granularity_selection_enabled'])
                restored,payload=load_neurosigvia_checkpoint(Path(directory)/'neurosigvia_checkpoint.pt',vision_model=vision)
                self.assertEqual(payload['protocol']['image_representation'],mode)
                self.assertEqual(payload['protocol']['fusion_input'],'aligned_projected_features')
                self.assertFalse(payload['protocol']['activity_graph_generated_online'])
                self.assertTrue((Path(directory)/'test_predictions.npz').exists())
                with torch.no_grad():
                    logits,_=restored(bundle['raw_windows'],bundle['line_tokens'].float(),
                        bundle['mantis_channel_tokens'].float(),bundle['patch_mask'],
                        bundle['valid_fraction'].float(),bundle['valid_lengths'],vision)
                self.assertEqual(tuple(logits.shape),(4,2))

    def test_cache_reuse_and_fail_closed_on_signature_change(self):
        with tempfile.TemporaryDirectory() as directory:
            loader=DataLoader(TensorDataset(torch.randn(4,4,128)),batch_size=4)
            labels=np.array([0,1,0,1]); bundle=make_bundle()
            kwargs=dict(window_size=64,stride=64,encode_batch_size=2,
                feature_cache_dir=directory,feature_cache_signature='signed-v1',
                expected_channels=4,image_representation='patch_heatmap',heatmap_patch_size=8)
            with patch('src.heatmap_ablation.extract_heatmap_feature_batch',return_value=bundle) as extract:
                first=get_heatmap_static_split('train',loader,labels,None,None,'cpu',**kwargs)
                self.assertEqual(extract.call_count,1)
            with patch('src.heatmap_ablation.extract_heatmap_feature_batch',side_effect=AssertionError('cache miss')):
                second=get_heatmap_static_split('train',loader,labels,None,None,'cpu',**kwargs)
            torch.testing.assert_close(first['line_tokens'],second['line_tokens'])
            with self.assertRaisesRegex(ValueError,'metadata mismatch'):
                get_heatmap_static_split('train',loader,labels,None,None,'cpu',
                    **dict(kwargs,feature_cache_signature='other'))

if __name__=='__main__':unittest.main()
