import sys, tempfile, unittest
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_paper_activity_graph import TinyFrozenVision
from src.adaptive_graph_training import (NeuroSigVIAClassifier, _run_epoch, _run_fusion_pretraining,
    load_neurosigvia_checkpoint, _encoder_contract, NEUROSIGVIA_ARCHITECTURE, NEUROSIGVIA_CHECKPOINT_SCHEMA_VERSION)

class FusionTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        n=4
        self.loader=DataLoader(TensorDataset(torch.randn(n,2,4,64),torch.randn(n,2,8),
            torch.randn(n,2,4,6),torch.ones(n,2,dtype=torch.bool),torch.ones(n,2),
            torch.full((n,2),64),torch.tensor([0,1,0,1]),torch.arange(n)),batch_size=2)
        self.vision=TinyFrozenVision()
    def model(self,mode):
        return NeuroSigVIAClassifier(8,6,4,2,fusion_dim=8,fusion_heads=2,
            classifier_hidden_dim=8,alignment_dim=4,dropout=0,
            temporal_visual_fusion=mode,alignment_enabled=False,mask_prob=1.)
    def test_masked_pretrain_updates_only_fusion_and_restores_flags(self):
        m=self.model('masked_pretrain')
        before={k:v.clone() for k,v in m.state_dict().items()}
        flags={k:p.requires_grad for k,p in m.named_parameters()}
        h=_run_fusion_pretraining(m,self.loader,self.vision,'cpu',epochs=1,lr=.001,
            weight_decay=.001,visual_encode_batch_size=4,graph_spatial_grid_size=4)
        self.assertEqual(h[0]['updates'],2)
        changed=[k for k,v in m.state_dict().items() if not torch.equal(v,before[k])]
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith('fusion.temporal_visual_fusion.') for k in changed),changed)
        self.assertEqual(flags,{k:p.requires_grad for k,p in m.named_parameters()})
    def test_supervised_gate_gradients_zero_alignment_and_checkpoint_reload(self):
        for mode in ('concat','masked_pretrain'):
            with self.subTest(mode=mode):
                m=self.model(mode)
                stats=_run_epoch(m,self.loader,self.vision,torch.optim.AdamW(m.parameters(),lr=.001),
                    torch.nn.CrossEntropyLoss(reduction='none'),'cpu',alignment_weight=0,
                    selector_balance_weight=.001,visual_encode_batch_size=4,
                    graph_spatial_grid_size=4,vision_gradient_checkpointing=True)
                self.assertEqual(stats['alignment_loss'],0.)
                self.assertFalse(any('alignment.' in n for n,_ in m.named_parameters()))
                payload=dict(schema_version=NEUROSIGVIA_CHECKPOINT_SCHEMA_VERSION,
                    architecture=NEUROSIGVIA_ARCHITECTURE,model_constructor_configuration=m.get_config(),
                    model_state_dict=m.state_dict(),external_encoder_contracts={'vision':_encoder_contract(self.vision,'vision')})
                with tempfile.TemporaryDirectory() as d:
                    p=Path(d)/'model.pt';torch.save(payload,p)
                    restored,_=load_neurosigvia_checkpoint(p,vision_model=self.vision)
                    self.assertEqual(restored.get_config(),m.get_config())
                    for k,v in restored.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k])
if __name__=='__main__':unittest.main()
