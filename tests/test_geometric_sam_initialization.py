"""Shape-aware query initialization, fixed selection and old-model compatibility."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
from tokengs.models.geometric_sam_initialization import GeometricSAMQueryInitializer
from tokengs.models.gfm_dual_stream_components import SemanticSAMQueryInitializer


class GeometricSAMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(25)
        self.init=GeometricSAMQueryInitializer()
        y,x=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
        self.xyz=torch.stack((x,y,x*.3+y*.2),-1).float()
        self.mask=torch.ones(8,8,dtype=torch.bool)
        self.fm=torch.randn(256,4,4)

    def test_local_shape_translation_scale_invariant_and_shape_sensitive(self):
        a=self.init.pool_region(self.fm,self.mask,self.xyz)
        b=self.init.pool_region(self.fm,self.mask,self.xyz*3+torch.tensor([5.,-2.,7.]))
        self.assertTrue(torch.allclose(a,b,atol=2e-6,rtol=2e-6))
        xyz=self.xyz.clone();xyz[...,2]=torch.sin(xyz[...,0])*4
        c=self.init.pool_region(self.fm,self.mask,xyz)
        self.assertGreater(float((a-c).norm()),1e-3)

    def test_single_pixel_invalid_points_and_empty_proposals(self):
        xyz=self.xyz.clone();xyz[0,0]=torch.nan
        mask=torch.zeros_like(self.mask);mask[0,:2]=True
        a=self.init.pool_region(self.fm,mask,xyz)
        mask[0,0]=False
        self.assertTrue(torch.equal(a,self.init.pool_region(self.fm,mask,xyz)))
        self.assertTrue(torch.isfinite(a).all())
        # Complete forward: an all-invalid proposal is skipped, a tiny valid
        # proposal survives, and all remaining slots use learned fallbacks.
        bad=torch.zeros_like(mask);bad[0,0]=True
        q,c,s,logs=self.init(self.fm[None,None],[[[dict(mask=bad,score=.99),dict(mask=mask,score=.9)]]],
                             xyz[None,None],torch.randn(1,1024,3))
        self.assertEqual(logs[0]['proposal_queries'],1)
        self.assertEqual(logs[0]['learned_queries'],99)
        self.assertTrue(all(torch.isfinite(t).all() for t in (q,c,s)))
        q,_,_,logs=self.init(self.fm[None,None],[[[]]],xyz[None,None],torch.randn(1,1024,3))
        self.assertEqual(logs[0]['proposal_queries'],0)
        self.assertTrue(torch.equal(q[0],self.init.norm(self.init.fallback)))

    def test_gradients_reach_semantic_adapter_and_all_pooling_branches(self):
        adapter=nn.Conv2d(6,256,1)
        fm=adapter(torch.randn(1,6,4,4))[0]
        xyz=self.xyz.clone().requires_grad_()
        q,_,_,_=self.init(fm[None,None],[[[dict(mask=self.mask,score=.9)]]],
                          xyz[None,None],torch.randn(1,1024,3))
        (q[0,0]*torch.randn(256)).sum().backward()
        self.assertGreater(float(adapter.weight.grad.norm()),0.)
        for name in ('local_shape.0.weight','point_weight.2.weight','shape_projection.weight','content_projection.1.weight'):
            grad=dict(self.init.named_parameters())[name].grad
            self.assertIsNotNone(grad,name)
            self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.norm()),0.,name)
        self.assertIsNone(xyz.grad)

    def test_old_v3_pooling_is_numerically_unchanged(self):
        mask=self.mask.clone();mask[:3]=False
        weights=F.interpolate(mask.float()[None,None],size=(4,4),mode='area')[0,0]
        expected=(self.fm*weights[None]).sum((1,2))/weights.sum().clamp_min(1e-8)
        actual=SemanticSAMQueryInitializer().pool_region(self.fm,mask,self.xyz)
        self.assertTrue(torch.equal(expected,actual))

    def test_bounded_sampling_is_deterministic(self):
        init=GeometricSAMQueryInitializer(max_region_points=7)
        a=init.pool_region(self.fm,self.mask,self.xyz)
        self.assertTrue(torch.equal(a,init.pool_region(self.fm,self.mask,self.xyz)))
        with self.assertRaises(ValueError):GeometricSAMQueryInitializer(0)

    def test_model_changes_only_initializer_and_registers_v4(self):
        from tokengs.options import config_defaults
        from tokengs.models import model_registry
        from tokengs.models.object_locus_gfm_dual_stream_v3 import LocusGSGFMDualStreamV3
        from tokengs.models.object_locus_gfm_dual_stream_v4 import LocusGSGFMDualStreamV4
        opt=config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4']
        with patch('tokengs.models.object_locus_dual_stream_v1.FrozenOmega',side_effect=lambda *a,**k:nn.Identity()),patch('tokengs.models.object_locus_dual_stream_v1.FrozenSAM',side_effect=lambda *a,**k:nn.Identity()),torch.device('meta'):
            old=LocusGSGFMDualStreamV3(config_defaults['train_siu3r_object_locus_gfm_dual_stream_v3'])
            new=LocusGSGFMDualStreamV4(opt)
        a=old.state_dict();b=new.state_dict()
        self.assertFalse(set(a)-set(b))
        self.assertTrue(all(a[k].shape==b[k].shape for k in a))
        additions=set(b)-set(a)
        self.assertTrue(additions)
        self.assertTrue(all(k.startswith(('query_initializer.local_shape.','query_initializer.point_weight.','query_initializer.shape_projection.')) for k in additions))
        self.assertIs(model_registry[opt.model_type],LocusGSGFMDualStreamV4)
        self.assertEqual(new.write_strength,0.)


if __name__=='__main__':unittest.main()
