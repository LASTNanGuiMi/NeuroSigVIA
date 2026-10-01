import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from src import visual_cache_reference as cache
from src import vision_ablation_training as training
from test_vision_ablation_training import tiny_splits,TinyVision,TinyVisualOnly

class FourDatasetContracts(unittest.TestCase):
    def test_binary_training_probability_columns_and_no_numeric(self):
        splits=tiny_splits()
        for split in splits.values():split['labels'][split['labels']==2]=1
        with tempfile.TemporaryDirectory() as directory,patch.object(training,'VisionOnlyClassifier',TinyVisualOnly):
            result=training.train_vision_only(splits,TinyVision(),dict(dataset='apava',class_names=['0','1'],
                mlp_epochs=1,batch_size=2,mlp_lr_scheduler='none'),directory,42,'cpu')
            self.assertEqual(result['dataset'],'apava');self.assertEqual(result['class_names'],['0','1'])
            self.assertFalse(result['uses_mantis']);self.assertEqual(result['test_evaluations'],1)
            with np.load(Path(directory)/'test_predictions.npz') as a:self.assertEqual(a['y_score'].shape,(6,2))
    def test_missing_cache_never_reextracts(self):
        with self.assertRaises(FileNotFoundError):cache.load_visual_cache('/nonexistent/cache.npz',[], 'sig',window_size=64,stride=64,expected_channels=2)
    def test_cache_loader_cannot_access_numeric_array(self):
        values=tiny_splits()['train'];labels=values['labels'];metadata=dict(schema_version=cache.NEUROSIGVIA_CACHE_SCHEMA_VERSION,
            architecture=cache.NEUROSIGVIA_CACHE_ARCHITECTURE,cache_signature='sig',tail_policy=cache.PATCH_TAIL_POLICY,window_size=64,stride=64)
        class Archive:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __getitem__(self,key):
                if key=='labels':return labels
                if key in metadata:return np.asarray(metadata[key])
                if key in cache.VISUAL_STATIC_KEYS:return values[key].numpy()
                raise AssertionError('Forbidden array: '+key)
        with patch.object(cache.Path,'is_file',return_value=True),patch.object(cache.np,'load',return_value=Archive()):
            actual=cache.load_visual_cache('fake',labels,'sig',window_size=64,stride=64,expected_channels=2)
            self.assertEqual(set(actual),set(cache.VISUAL_STATIC_KEYS))
    def test_permuted_raw_cache_is_rejected(self):
        x=torch.arange(3*2*128,dtype=torch.float32).reshape(3,2,128)
        loader=DataLoader(TensorDataset(x),batch_size=2)
        t=cache.make_temporal_patches(x,window_size=64,stride=64)
        v=dict(raw_windows=t.patches,patch_mask=t.patch_mask,valid_fraction=t.valid_fraction.half(),valid_lengths=t.valid_lengths)
        cache.verify_raw_order(v,loader,64,64)
        v['raw_windows']=v['raw_windows'].flip(0)
        with self.assertRaisesRegex(ValueError,'identity/order'):cache.verify_raw_order(v,loader,64,64)

if __name__=='__main__':unittest.main()
