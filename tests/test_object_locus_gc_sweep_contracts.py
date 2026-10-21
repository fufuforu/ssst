import unittest
import numpy as np
import torch
from scripts.object_locus_gc_sweep_runtime import (
    ARMS,SOURCE_EXPOSURES,SOURCE_SHA256,build_model,build_optimizer,combine_gradients,
    family,load_source_blob,lr_multiplier,make_plan,state_equal)

class ObjectLocusGCSweepContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.opt,_=build_model('cpu')
        cls.peer,_,_=build_model('cpu')
        cls.optimizer=build_optimizer(cls.model)

    def test_checkpoint_metadata_strict_source_and_identical_arm_initialization(self):
        blob=load_source_blob()
        self.assertEqual(blob['epoch'],6);self.assertEqual(blob['completed_updates'],6258);self.assertEqual(blob['completed_exposures'],50064)
        self.assertEqual(SOURCE_SHA256,'68de912a60340f65d675a521c514641845c43822657f07d5851770c96cbf912a')
        self.assertTrue(state_equal(self.model,self.peer));self.assertTrue(all(p.requires_grad for p in self.model.parameters()))
        del blob

    def test_optimizer_exact_coverage_fresh_adamw_groups(self):
        expected={id(p) for p in self.model.parameters() if p.requires_grad};actual=[id(p) for g in self.optimizer.param_groups for p in g['params']]
        self.assertEqual(len(actual),len(set(actual)));self.assertEqual(set(actual),expected);self.assertEqual(self.optimizer.state,{})
        for g in self.optimizer.param_groups:
            self.assertEqual(g['lr'],g['peak_lr'])
            if g['name'].startswith('reconstruction'):self.assertEqual(g['weight_decay'],0)
            if g['name'].endswith('_decay'):self.assertEqual(g['weight_decay'],.05)

    def test_saved_source_plan_identity_fixed_sweep_plan_and_scene_independence(self):
        manifest,plan=make_plan();self.assertEqual(len(manifest['expanded_train_windows']),1008)
        self.assertEqual(len({w['scene'] for w in manifest['expanded_train_windows']}),128)
        counts=np.zeros(1008,dtype=np.int32)
        for epoch in range(8):
            perm=np.random.default_rng(42+epoch).permutation(1008)
            self.assertEqual(len(np.unique(perm)),1008)
            for update in range(126):counts[perm[update*8:(update+1)*8]]+=1
        self.assertTrue(np.all(counts==8));self.assertEqual(plan['window_exposure_counts'],counts.tolist())
        self.assertEqual(plan['dev_val_scene_intersections']['dev8'],[]);self.assertEqual(plan['dev_val_scene_intersections']['val32'],[])
        self.assertEqual(SOURCE_EXPOSURES+8064,58128)

    def test_gradient_formula_alpha_only_scales_reconstruction_understanding_gradient(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));q=torch.nn.Parameter(torch.tensor([3.,4.]))
        rec=(p.sum()+2*q.sum()).square();under=(2*p[0]-q.sum()).square()
        names=['reconstruction.weight','panoptic.object.weight'];params=[p,q]
        gr=torch.autograd.grad(rec,params,retain_graph=True);gu=torch.autograd.grad(under,params)
        for alpha in ARMS.values():
            got=combine_gradients(params,names,gr,gu,alpha)
            torch.testing.assert_close(got[0],gr[0]+alpha*gu[0]);torch.testing.assert_close(got[1],gr[1]+gu[1])

    def test_schedule_endpoints_and_local_warmup_source_step_conventions(self):
        self.assertAlmostEqual(lr_multiplier(0),.04);self.assertAlmostEqual(lr_multiplier(24),1.0);self.assertAlmostEqual(lr_multiplier(1007),.1)
        self.assertEqual(SOURCE_EXPOSURES+8*0,50064);self.assertEqual(SOURCE_EXPOSURES+8*1007,58120)
        self.assertEqual(SOURCE_EXPOSURES+8064,58128)

if __name__=='__main__':unittest.main()
