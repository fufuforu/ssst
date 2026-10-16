import copy
import json
import math
import unittest
from unittest.mock import patch
import torch
from scripts.dual_stream_training_policy import understanding_weight,configure_joint_warmup,validate_joint_warmup_source,load_optimizer_preserving_moments_and_new_rates
from scripts.score_dual_stream_v1 import native_table_values,json_safe


class JointWarmupTests(unittest.TestCase):
    def test_ramp_is_bounded_and_uses_persisted_global_joint_clock(self):
        self.assertAlmostEqual(understanding_weight(0,1043),1/1043)
        self.assertAlmostEqual(understanding_weight(250,1043),251/1043)
        self.assertEqual(understanding_weight(1042,1043),1.)
        self.assertEqual(understanding_weight(4171,1043),1.)
        self.assertEqual(understanding_weight(0,0),1.)
        with self.assertRaises(ValueError):understanding_weight(-1,1043)

    def test_task_weights_understanding_but_does_not_freeze_shared_reconstruction(self):
        from scripts.train_dual_stream_v1 import Task
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.rec=torch.nn.Parameter(torch.ones(1));self.under=torch.nn.Parameter(torch.ones(1))
            def step_loss(self,batch,*,step,understanding_weight):
                rec=self.rec.square().sum();under=(self.rec+self.under).square().sum()
                return {'prediction':None},dict(loss_recon=rec,loss=rec+understanding_weight*under)
        model=Model()
        with patch('scripts.train_dual_stream_v1.scalar_metrics',return_value={}):
            loss,_=Task(model)({},0,False,.25);loss.backward()
        self.assertAlmostEqual(float(loss),2.)
        self.assertAlmostEqual(float(model.rec.grad),3.)
        self.assertAlmostEqual(float(model.under.grad),1.)

    def test_adam_moments_survive_but_source_learning_rate_does_not_override_target(self):
        p=torch.nn.Parameter(torch.ones(2))
        old=torch.optim.AdamW([dict(params=[p],lr=1e-4,base_lr=1e-4,family='reconstruction')])
        p.sum().backward();old.step();saved=copy.deepcopy(old.state_dict())
        q=torch.nn.Parameter(p.detach().clone())
        new=torch.optim.AdamW([dict(params=[q],lr=1e-5,base_lr=1e-5,family='reconstruction')])
        load_optimizer_preserving_moments_and_new_rates(new,saved)
        self.assertEqual(new.param_groups[0]['base_lr'],1e-5);self.assertEqual(new.param_groups[0]['lr'],1e-5)
        for key,value in old.state[p].items():self.assertTrue(torch.equal(new.state[q][key],value))

    def fixture(self):
        recipe='omega_gfm_semantic_pool_v3_adapt2_joint4_v1'
        saved=dict(execution_git_sha='75f6ce182b4ee79e75485db541daf0ffc85f91a9',recipe=recipe,world_size=4,accumulation=2,global_batch=8,data='same',
            config=dict(recipe=recipe,adaptation_epochs=2,reconstruction_lr=1e-4,geometry_refinement_lr=1e-5,understanding_lr=1e-4,full_monitor_every_updates=250))
        target=copy.deepcopy(saved);configure_joint_warmup(target['config']);target.update(recipe=target['config']['recipe'],world_size=8,accumulation=1,execution_git_sha='b'*40)
        progress=dict(short_updates=306,adapt_epoch=2,adapt_updates=2086,epoch=0,full_updates=0,phase='full')
        return saved,target,progress

    def test_explicit_four_to_eight_rank_transition_accepts_only_adaptation_boundary(self):
        saved,target,progress=self.fixture();validate_joint_warmup_source(saved,target,progress)
        self.assertEqual(target['config']['reconstruction_lr']/target['config']['understanding_lr'],.1)
        with self.assertRaises(ValueError):validate_joint_warmup_source(saved,target,dict(progress,full_updates=1))
        with self.assertRaises(ValueError):validate_joint_warmup_source(saved,dict(target,data='wrong'),progress)
        with self.assertRaises(ValueError):validate_joint_warmup_source(saved,dict(target,world_size=4,accumulation=2),progress)

    def test_invalid_native_depth_is_preserved_as_invalid_not_filled(self):
        raw={k:.5 for k in ('absrel','rmse','psnr','ssim','lpips')};raw['absrel']=float('nan')
        for view in ('context','target'):raw.update({view+'_miou':.5,view+'_pq':.5,view+'_map':{'map':.5}})
        values=native_table_values(raw);self.assertEqual(len(values),13);self.assertTrue(math.isnan(values[0]))
        self.assertIsNone(values[8]);self.assertIsNone(values[12])
        invalid=[];safe=json_safe(raw,invalid=invalid)
        self.assertIsNone(safe['absrel']);self.assertEqual(invalid,['$.absrel'])
        self.assertEqual(safe['psnr'],raw['psnr']);json.dumps(safe,allow_nan=False)

if __name__=='__main__':unittest.main()
