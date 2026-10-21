"""The zero-step exception must not weaken checkpoint compatibility."""
import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from scripts.unified_query_initial_state import restore_trainable_state,validate_zero_initial

class Model(nn.Module):
    architecture_name='test'
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(to_dict=lambda: {'dim':4})
        self.frozen_omega=SimpleNamespace(source_identity={'sha':'omega'})
        self.sam_proposals=None
        self.core=nn.Module()
        self.core.base=nn.Linear(4,4)
        self.core.iou_head=nn.Sequential(nn.LayerNorm(4),nn.Linear(4,4),nn.GELU(),nn.Linear(4,1))

class InitialStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);self.model=Model()
        self.base={k:torch.full_like(v,.25) for k,v in self.model.state_dict().items() if not k.startswith('core.iou_head.')}
        self.saved=dict(model=self.base,meta=dict(architecture='test',config={'dim':4},manifest_sha256='train',heldout_sha256='val',world_size=8,omega_identity={'sha':'omega'},sam_identity=None),progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer={'state':{}},rank_rng=[{}]*8)
    def check(self,saved):
        validate_zero_initial(saved,self.model,manifest_sha256='train',heldout_sha256='val',world=8)
    def test_new_complete_head_is_only_exception(self):
        self.check(self.saved)
        hidden=self.model.core.iou_head[1].weight.clone()
        restore_trainable_state(self.model,self.base,allow_new_iou_head=True)
        for k,v in self.base.items():torch.testing.assert_close(self.model.state_dict()[k],v)
        torch.testing.assert_close(self.model.core.iou_head[1].weight,hidden)
        torch.testing.assert_close(self.model.core.iou_head(torch.randn(2,4)).sigmoid(),torch.full((2,1),.5))
        self.assertGreater(int(torch.count_nonzero(hidden)),0)
    def test_export_and_resume_require_complete_head(self):
        with self.assertRaises(ValueError):restore_trainable_state(self.model,self.base)
        full=copy.deepcopy(self.model.state_dict())
        restore_trainable_state(self.model,full)
        self.check(dict(self.saved,model=full))
    def test_unrelated_missing_extra_shape_and_partial_head_rejected(self):
        cases=[]
        missing=dict(self.base);missing.pop('core.base.bias');cases.append(missing)
        cases.append(dict(self.base,**{'unexpected':torch.zeros(1)}))
        cases.append(dict(self.base,**{'core.base.weight':torch.zeros(1)}))
        cases.append(dict(self.base,**{'core.iou_head.0.weight':torch.ones(4)}))
        for state in cases:
            with self.subTest(keys=list(state)):
                with self.assertRaises(ValueError):self.check(dict(self.saved,model=state))
    def test_world_data_sources_progress_and_adam_still_strict(self):
        for key,value in [('world_size',4),('manifest_sha256','wrong'),('omega_identity',{})]:
            bad=copy.deepcopy(self.saved);bad['meta'][key]=value
            with self.assertRaises(ValueError):self.check(bad)
        for key,value in [('rank_rng',[{}]),('progress',dict(epoch=0,update_in_epoch=1,full_updates=1)),('optimizer',{'state':{0:{'step':1}}})]:
            with self.assertRaises(ValueError):self.check(dict(self.saved,**{key:value}))

if __name__=='__main__':unittest.main()
