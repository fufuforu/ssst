import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests.test_unified_query_v1 import config,opt,cloud,FakeOmega,FakeSAM,FakeRenderer
from tokengs.models.geometry_controlled_queries import (
    bounded_anchor_update,feature_grid_rays,GeometryBiasedUnifiedCore)
from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.superpoint_structural_variants import (
    LocusGSBoundedSuperpointDual,LocusGSGeometryBiasedUnified,build_structural_model)


class GeometryControlTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_bounded_update_zero_extreme_and_scale(self):
        mu=torch.randn(2,4,3,dtype=torch.float64)
        rho=torch.linspace(-30,30,8,dtype=torch.float64).reshape(2,4)
        epsilon=1e-4;radius=F.softplus(rho)+epsilon
        center,new=bounded_anchor_update(mu,rho,torch.zeros_like(mu),torch.zeros_like(rho),epsilon)
        torch.testing.assert_close(center,mu);torch.testing.assert_close(new,rho)
        raw=torch.randn_like(mu)*1e5;raw_r=torch.randn_like(rho)*1e5
        center,new=bounded_anchor_update(mu,rho,raw,raw_r,epsilon)
        ratio=(F.softplus(new)+epsilon)/radius
        self.assertTrue(torch.isfinite(new).all())
        self.assertTrue(((center-mu).norm(dim=-1)<=.1*radius+1e-12).all())
        self.assertTrue((ratio<=torch.exp(torch.tensor(.1))+1e-7).all())
        self.assertTrue((ratio>=torch.exp(torch.tensor(-.1))-1e-7).all())
        scale=7.;positive=F.softplus(rho)*scale
        scaled_rho=positive+torch.log(-torch.expm1(-positive))
        c2,r2=bounded_anchor_update(mu*scale,scaled_rho,raw,raw_r,epsilon*scale)
        torch.testing.assert_close(c2,center*scale)
        torch.testing.assert_close(F.softplus(r2)+epsilon*scale,(F.softplus(new)+epsilon)*scale)

    def test_bounded_update_has_both_geometry_gradients(self):
        raw=torch.randn(1,3,3,requires_grad=True);r=torch.randn(1,3,requires_grad=True)
        center,rho=bounded_anchor_update(torch.zeros_like(raw),torch.zeros_like(r),raw,r,.001)
        (center.sum()+F.softplus(rho).sum()).backward()
        self.assertGreater(float(raw.grad.norm()),0.)
        self.assertGreater(float(r.grad.norm()),0.)

    def test_feature_rays_pixel_centers_and_view_row_column_order(self):
        pose=torch.eye(4).repeat(1,2,1,1);pose[:,1,0,3]=.3
        k=torch.tensor([[8.,0.,4.],[0.,8.,4.],[0.,0.,1.]]).repeat(1,2,1,1)
        moment,direction=feature_grid_rays(pose,k,image_hw=(8,8),grid_hw=(2,2))
        expected=F.normalize(torch.tensor([[-.25,-.25,1.],[.25,-.25,1.],[-.25,.25,1.],[.25,.25,1.]]),dim=-1)
        torch.testing.assert_close(direction[0,:4],expected)
        torch.testing.assert_close(direction[0,4:],expected)
        torch.testing.assert_close(moment[0,:4],torch.zeros_like(expected))
        torch.testing.assert_close(moment[0,4:],torch.cross(torch.tensor([.3,0.,0.]).expand_as(expected),expected,dim=-1))

    def test_soft_bias_geometry_scale_gradients_and_zero_gate_equivalence(self):
        core=GeometryBiasedUnifiedCore(opt(),config())
        q=torch.randn(2,3,16);features=torch.randn(2,5,16)
        mu=(torch.randn(2,3,3)*.2).requires_grad_();radii=torch.full((2,3),.3,requires_grad=True)
        directions=F.normalize(torch.randn(2,5,3),dim=-1)
        moment=torch.cross(torch.randn(2,5,3)*.1,directions,dim=-1)
        memory=dict(features=features,rays=(moment,directions))
        b=core.attention_bias(mu,radii,moment,directions,0)
        torch.testing.assert_close(b,core.attention_bias(mu*5,radii*5,moment*5,directions,0))
        self.assertTrue(torch.isfinite(b).all());self.assertLessEqual(float(b.max()),0.)
        self.assertFalse(torch.equal(b,core.attention_bias(mu+.2,radii,moment,directions,0)))
        out=core.decode_query(core.blocks[0],q,memory,mu,radii,0)
        out.square().mean().backward()
        for value in (mu.grad,radii.grad,core.ray_gate_raw.grad):
            self.assertTrue(torch.isfinite(value).all());self.assertGreater(float(value.norm()),0.)
        with torch.no_grad():core.ray_gate_raw.fill_(-float('inf'))
        expected=core.blocks[0](q,features)
        torch.testing.assert_close(core.decode_query(core.blocks[0],q,memory,mu,radii,0),expected,rtol=1e-5,atol=1e-6)

    def test_checkpoint_replay_and_next_layer_receive_updated_geometry(self):
        a=GeometryBiasedUnifiedCore(opt(),replace(config(),checkpoint_blocks=True,depth=3))
        b=GeometryBiasedUnifiedCore(opt(),replace(config(),checkpoint_blocks=False,depth=3));b.load_state_dict(a.state_dict())
        with torch.no_grad():
            for model in (a,b):
                for head in model.geometry:head.bias.copy_(torch.tensor([.2,-.3,.1,.4]))
        memory=dict(features=torch.randn(1,5,16),rays=(torch.randn(1,5,3)*.01,F.normalize(torch.randn(1,5,3),dim=-1)))
        scene=cloud();seen=[];original=a.attention_bias
        def capture(mu,r,mo,d,i):
            seen.append((i,mu.detach().clone(),r.detach().clone()))
            return original(mu,r,mo,d,i)
        with patch.object(a,'attention_bias',side_effect=capture):out_a=a(copy.deepcopy(scene),memory)
        out_b=b(copy.deepcopy(scene),memory)
        self.assertEqual([x[0] for x in seen],[0,1,2])
        torch.testing.assert_close(seen[1][1],out_a['states'][0]['mu'])
        torch.testing.assert_close(seen[2][2],out_a['states'][1]['radii'])
        for out in (out_a,out_b):
            (out['gaussians'].square().mean()+out['gaussian_mask_logits'].square().mean()).backward()
        torch.testing.assert_close(out_a['gaussians'],out_b['gaussians'])
        for (name,p),(_,other) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(p.grad is None,other.grad is None,name)
            if p.grad is not None:torch.testing.assert_close(p.grad,other.grad,rtol=1e-4,atol=1e-6,msg=name)

    def test_legacy_unified_state_and_forward_are_unchanged(self):
        base=UnifiedQueryCore(opt(),config());new=GeometryBiasedUnifiedCore(opt(),config())
        missing=new.load_state_dict(base.state_dict(),strict=False)
        self.assertEqual(missing.missing_keys,['ray_gate_raw']);self.assertFalse(missing.unexpected_keys)
        with torch.no_grad():new.ray_gate_raw.fill_(-float('inf'))
        scene=cloud();memory=torch.randn(1,5,16)
        a=base(scene,memory);b=new(scene,dict(features=memory,rays=(torch.zeros(1,5,3),torch.randn(1,5,3))))
        for key in ('gaussians','gaussian_mask_logits','thing_logits19'):torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-6)

    def test_unified_context_pipeline_and_parameter_coverage(self):
        model=build_structural_model('unified_ray_bias',opt(),config(),omega=FakeOmega(),renderer=FakeRenderer())
        self.assertIsInstance(model,LocusGSGeometryBiasedUnified)
        out=model.generate(torch.rand(1,2,3,256,256))
        (out['gaussians'].square().mean()+out['gaussian_mask_logits'].square().mean()).backward()
        self.assertIsNotNone(model.core.ray_gate_raw.grad)
        self.assertTrue(torch.isfinite(out['gaussians']).all())
        groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertIn(id(model.core.ray_gate_raw),{id(p) for p in groups['shared']})

    def test_a_bounded_end_to_end_bounds_and_head_gradients(self):
        cfg=replace(config(),dim=256,heads=8,depth=12,superpoint_source='esam_sam',matching_mode='source_many_to_one')
        model=LocusGSBoundedSuperpointDual(opt().evolve(dec_patch_size=8,img_size=(8,8)),config=cfg,omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer())
        seen=[];original=model.core.anchor_decoder.update_geometry
        def capture(h,mu,rho,hm,hr):
            center,new=original(h,mu,rho,hm,hr)
            with torch.no_grad():
                radius=model.core.anchor_decoder.activated_radius(rho)
                self.assertTrue(((center-mu).norm(dim=-1)<=.100001*radius).all())
            seen.append(center.shape)
            return center,new
        with patch.object(model.core.anchor_decoder,'update_geometry',side_effect=capture):
            out=model.generate(torch.rand(1,2,3,256,256))
        self.assertEqual(len(seen),12)
        (out['gaussians'].square().mean()+out['gaussian_mask_logits'].square().mean()).backward()
        for heads in (model.core.anchor_decoder.refine_mu,model.core.anchor_decoder.refine_rho):
            self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for p in heads.parameters()))
        self.assertTrue(torch.isfinite(out['gaussians']).all())


if __name__=='__main__':unittest.main()
