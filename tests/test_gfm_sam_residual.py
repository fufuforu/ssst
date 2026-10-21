import copy
import math
import unittest
from unittest.mock import patch
import torch
from torch import nn
from tokengs.models.geometric_sam_initialization import GeometricSAMQueryInitializer,ResidualGeometricSAMQueryInitializer
from scripts.gfm_sam_residual_policy import configure,validate_initial,restore_shared_initial,NEW
from tests.test_gfm_v4_isolated_policy import IsolatedPolicyTests


class ResidualTests(unittest.TestCase):
    def test_seed_is_small_normalized_residual_with_direct_base_and_content_gradients(self):
        torch.manual_seed(2);init=ResidualGeometricSAMQueryInitializer()
        base=torch.randn(256,requires_grad=True)*.02;base.retain_grad()
        seed=torch.randn(256,requires_grad=True)
        q=init.compose_seed(base,seed);delta=q-base
        self.assertAlmostEqual(float(delta.norm()/base.norm()),.1,places=4)
        torch.testing.assert_close(q,init.compose_seed(base,seed*3+2),atol=1e-7,rtol=1e-5)
        q[0].backward()
        for gradient in (base.grad,seed.grad,init.residual_gate_logit.grad):
            self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0.)
        with torch.no_grad():init.residual_gate_logit.fill_(-100)
        torch.testing.assert_close(init.compose_seed(base,seed),base)

    def test_candidates_centers_scales_and_fallback_values_unchanged(self):
        torch.manual_seed(4);old=GeometricSAMQueryInitializer();new=ResidualGeometricSAMQueryInitializer()
        new.load_state_dict(old.state_dict(),strict=False)
        memory=torch.randn(1,2,256,2,2);points=torch.randn(1,2,4,4,3);mu=torch.randn(1,10,3)
        proposals=[[[dict(score=1.,mask=torch.ones(4,4,dtype=torch.bool))],[]]]
        a=old(memory,proposals,points,mu);b=new(memory,proposals,points,mu)
        self.assertEqual(a[3][0]['proposal_queries'],b[3][0]['proposal_queries']);self.assertEqual(b[3][0]['proposal_queries'],1)
        for x,y in zip(a[1:3],b[1:3]):torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(a[0][:,1:],b[0][:,1:],rtol=0,atol=0)
        self.assertFalse(torch.equal(a[0][:,:1],b[0][:,:1]))
        b[0][0,0,0].backward()
        self.assertGreater(float(new.fallback.grad[0].norm()),0.)
        self.assertGreater(float(new.content_projection[1].weight.grad.norm()),0.)
        self.assertGreater(float(new.point_weight[2].weight.grad.norm()),0.)

    def test_source_metadata_remains_locked(self):
        saved,_,blob=IsolatedPolicyTests().fixture();saved['config']['required_gradient_families']=['geometric_query_pooling']
        target=copy.deepcopy(saved);configure(target['config']);target['recipe']=target['config']['recipe']
        validate_initial(saved,target,blob)
        for key,value in [('sam_residual_initial_gate',.2),('understanding_lr',.2)]:
            bad=copy.deepcopy(target);bad['config'][key]=value
            with self.assertRaises(ValueError):validate_initial(saved,bad,blob)
        bad=copy.deepcopy(blob);bad['progress']['full_updates']=1
        with self.assertRaises(ValueError):validate_initial(saved,target,bad)

    def test_shared_weights_exact_and_only_gate_new_empty_adam(self):
        class Model(nn.Module):
            def __init__(self,residual=False):
                super().__init__();self.query_initializer=nn.Module();self.query_initializer.fallback=nn.Parameter(torch.randn(2,3))
                if residual:self.query_initializer.residual_gate_logit=nn.Parameter(torch.tensor(0.))
        def groups(m):return [dict(params=list(m.parameters()),family='queries',lr=.001,base_lr=.001,weight_decay=.05)]
        old=Model();new=Model(True);opt=torch.optim.AdamW(groups(new));sourceopt=torch.optim.AdamW(groups(old))
        blob=dict(model=old.state_dict(),optimizer=sourceopt.state_dict())
        with patch('scripts.gfm_geometric_transition.template',return_value=old):
            report=restore_shared_initial(new,opt,blob,groups)
            torch.testing.assert_close(new.query_initializer.fallback,old.query_initializer.fallback,rtol=0,atol=0)
            self.assertEqual(report['new_parameters'],1);self.assertEqual(report['new_tensor_names'],[NEW]);self.assertFalse(opt.state)
            self.assertAlmostEqual(float(new.query_initializer.residual_gate_logit.sigmoid()),.1,places=6)
            blob['model']['unexpected']=torch.ones(1)
            with self.assertRaises(ValueError):restore_shared_initial(new,opt,blob,groups)


if __name__=='__main__':unittest.main()
