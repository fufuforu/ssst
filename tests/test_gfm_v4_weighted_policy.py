import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from scripts.dual_stream_training_policy import CONFIG,understanding_weight
from scripts.gfm_v4_direct_joint_policy import configure_direct_joint
from scripts.gfm_v4_weighted_policy import configure,validate_initial,SOURCE_CODE


class WeightedPolicyTests(unittest.TestCase):
    def fixture(self):
        config=dict(CONFIG);configure_direct_joint(config)
        saved=dict(config=config,recipe=config['recipe'],execution_git_sha=SOURCE_CODE,world_size=8)
        target=copy.deepcopy(saved);configure(target['config']);target['recipe']=target['config']['recipe']
        blob=dict(progress=dict(short_updates=0,full_updates=0,epoch=0,update_in_epoch=0,phase='full',epoch_counts=[0]*3,total_counts=[0]*3),optimizer={'state':{}},rank_rng=[None]*8)
        return saved,target,blob

    def test_weight_only_metadata_change_and_ramp(self):
        saved,target,blob=self.fixture();validate_initial(saved,target,blob)
        self.assertAlmostEqual(.05*understanding_weight(0,1043),.05/1043)
        self.assertEqual(.05*understanding_weight(1042,1043),.05)
        self.assertEqual(target['config']['reconstruction_lr'],saved['config']['reconstruction_lr'])
        for mutation in ('trained','optimizer','rate'):
            t=copy.deepcopy(target);b=copy.deepcopy(blob)
            if mutation=='trained':b['progress']['full_updates']=1
            if mutation=='optimizer':b['optimizer']['state']={0:{}}
            if mutation=='rate':t['config']['understanding_lr']=2e-4
            with self.assertRaises(ValueError):validate_initial(saved,t,b)

    def test_joint_backward_keeps_understanding_gradient_into_reconstruction(self):
        from scripts.train_dual_stream_v1 import Task
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.rec=nn.Parameter(torch.tensor(1.));self.under=nn.Parameter(torch.tensor(2.))
            def step_loss(self,batch,*,step,understanding_weight):
                rec=self.rec.square();under=(self.rec+self.under).square()
                return {'prediction':None},dict(loss_recon=rec,loss_understanding=under,loss=rec+understanding_weight*under)
        model=Model()
        with patch('scripts.train_dual_stream_v1.scalar_metrics',return_value={}):
            loss,_=Task(model)({},0,False,.05)
        loss.backward()
        self.assertAlmostEqual(float(loss),1.45,places=6)
        self.assertAlmostEqual(float(model.rec.grad),2.3,places=6)
        self.assertAlmostEqual(float(model.under.grad),.3,places=6)


if __name__=='__main__':unittest.main()
