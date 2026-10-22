import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from torch import nn
from tokengs.options import config_defaults
from tokengs.models.olda4_initialization_ablation import V4Initialization,OldA4GeometricObjectLayer,LocusGSOldA4V4Init,LocusGSOldA4NoCA
from tokengs.models.dynamic_panoptic_controller import DynamicObjectLayer
from tokengs.models.object_locus_panoptic_v1_controller import geometry_bias


class InitializationTests(unittest.TestCase):
    def test_exact_v4_formula_and_each_new_input(self):
        torch.manual_seed(42);module=V4Initialization(config_defaults['train_siu3r_object_locus_gfm_dual_stream_v4'])
        mu=torch.randn(1,1024,3);radius=torch.rand(1,1024)+.1;color=torch.rand(1,1024,3);visual=torch.randn(1,1024,2048)
        seed=dict(centers=mu,radii=radius,color=color,visual=visual)
        origin=mu.mean(1,keepdim=True);ell=(mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05)
        expected=module.gs_tokens[None].clone()+module.anchor_content(torch.cat(((mu-origin)/ell[:,None,None],(radius/ell[:,None])[...,None],color),-1))+module.anchor_visual_projection(visual)
        out=module(seed);torch.testing.assert_close(out,expected,rtol=0,atol=0)
        for key in seed:
            changed=dict(seed);changed[key]=seed[key]+torch.randn_like(seed[key])*.05
            self.assertFalse(torch.equal(module(changed),out),key)
        out.square().mean().backward()
        for p in module.parameters():self.assertGreater(float(p.grad.norm()),0.)


class NoCALayerTests(unittest.TestCase):
    def test_geometry_normalized_fm_still_read_and_anchor_features_unused(self):
        torch.manual_seed(7);layer=DynamicObjectLayer();layer.__class__=OldA4GeometricObjectLayer;del layer.anchor_attention
        # Zero output projections are normal at initialization; activate only in this test.
        nn.init.normal_(layer.image_attention.O.weight,std=.01)
        h=torch.randn(1,5,1024);a=torch.randn(1,5,256,requires_grad=True);mu=torch.randn(1,5,3)
        q=torch.randn(1,4,256);c=torch.randn(1,2,3);s=torch.ones(1,2,3)*.7;ell=torch.ones(1)
        image=torch.randn(1,8,256,requires_grad=True);mask=nn.Linear(256,256);fa=torch.randn(1,5,256)
        def run(aa,im,mm):return layer(h,aa,mm,q,c,s,ell,im,mask,fa,0)
        x=run(a,image,mu);y=run(a*100,image,mu)
        expected=geometry_bias(mu,c,s).softmax(-1)[:,None]
        torch.testing.assert_close(x['evidence_attention'],expected,rtol=0,atol=0)
        torch.testing.assert_close(expected.sum(-1),torch.ones(1,1,4))
        for k in ('q','c','s'):torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
        grads=torch.autograd.grad(x['q'].square().sum(),(a,image),allow_unused=True)
        self.assertIsNone(grads[0]);self.assertGreater(float(grads[1].norm()),0)
        self.assertFalse(torch.equal(run(a,image,mu+.1)['c'],x['c']))


class RealCoreTests(unittest.TestCase):
    def _check(self,constructor):
        from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
        from tokengs.models.unified_query_loss import unified_instance_loss
        from tokengs.models.input_types import ModelInputDecoder,ModelSupervision
        cfg=replace(config(),dim=256,heads=8,depth=12,superpoint_source='esam_sam',matching_mode='source_many_to_one')
        torch.manual_seed(42)
        model=constructor(opt().evolve(dec_patch_size=8,img_size=(8,8)),config=cfg,omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer())
        self.assertEqual(model.supervised_layers,(6,8,10,12));self.assertEqual(model.layer_weights,[.1,.2,.3,.4])
        self.assertFalse(hasattr(model.core,'understanding_adapters'))
        for layer in model.core.panoptic.layers.values():
            if hasattr(layer,'anchor_attention'):nn.init.normal_(layer.anchor_attention.O.weight,std=.01)
            nn.init.normal_(layer.image_attention.O.weight,std=.01)
        nn.init.normal_(model.core.panoptic.child_mlp[-1].weight,std=.01)
        def small_memory(layers,rgb):
            m=model.core.semantic_memory
            patches=sum(w*proj(x.float().mean(-2)) for w,proj,x in zip(m.layer_logits.softmax(0),m.projections,layers))
            grid=patches.reshape(2,256,1,1).expand(-1,-1,8,8)
            return m.refine(grid+m.local(torch.nn.functional.interpolate(rgb.flatten(0,1),size=(8,8),mode='bilinear',align_corners=False))).reshape(1,2,256,8,8)
        rgb=torch.rand(1,2,3,256,256)
        original=model.core.initialize_reconstruction_anchors
        def reduced(*args,**kwargs):
            result=original(*args,**kwargs)
            if constructor is LocusGSOldA4V4Init:
                self.assertEqual(result['tokens'].shape,(1,1024,1024))
                for k in ('tokens','centers','radii'):result[k]=result[k][:,:6]
                for k in ('anchor_to_superpoint','anchor_slot_ids','anchor_point_indices'):result[k]=result[k][:6]
            return result
        with patch.object(model.core.semantic_memory,'forward',side_effect=small_memory),patch.object(model.core,'initialize_reconstruction_anchors',side_effect=reduced):
            out=model.generate(rgb)
            self.assertEqual(out['num_thing_queries'],2)
            self.assertEqual(out['gaussians'].shape,(1,384 if constructor is LocusGSOldA4V4Init else 128,14))
            expected=model.core.initializer.understanding(out['cloud'])
            torch.testing.assert_close(out['q_pre'],expected['query'])
            # Seed continues to learn the OLD shared projection, not semantic FM.
            seed_loss=(out['q_pre']*torch.randn_like(out['q_pre'])).sum()
            shared=list(model.gfm_projection.parameters());fm=list(model.core.semantic_memory.parameters())
            grads=torch.autograd.grad(seed_loss,shared+fm,retain_graph=True,allow_unused=True)
            self.assertGreater(sum(float(g.norm()) for g in grads[:len(shared)] if g is not None),0)
            self.assertTrue(all(g is None for g in grads[len(shared):]))
            batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
            lu,_=unified_instance_loss(out,batch,cfg)
            groups=model.parameter_groups();allids=[id(p) for ps in groups.values() for p in ps]
            self.assertEqual(len(allids),len(set(allids)));self.assertEqual(set(allids),{id(p) for p in model.parameters() if p.requires_grad})
            lu.backward(retain_graph=True)
            for group in ('reconstruction','understanding','shared'):
                self.assertGreater(sum(float(p.grad.norm()) for p in groups[group] if p.grad is not None),0,group)
            model.zero_grad(set_to_none=True)
            decoder=ModelInputDecoder(cam_view=torch.eye(4).expand(1,2,4,4),intrinsics=torch.tensor([4.,4.,4.,4.]).expand(1,2,4))
            supervision=ModelSupervision(images_output=torch.rand(1,2,3,8,8),masks_output=torch.ones(1,2,1,8,8),has_mask=torch.tensor([False]),rays_os=torch.zeros(1,2,3,8,8),rays_ds=torch.ones(1,2,3,8,8))
            lr,*_=model._layer_objective(out['states'],decoder,supervision);self.assertTrue(torch.isfinite(lr));lr.backward()
            self.assertGreater(sum(float(p.grad.norm()) for p in groups['reconstruction'] if p.grad is not None),0)

    def test_v4_init_joint_actual_loss_routing(self):self._check(LocusGSOldA4V4Init)
    def test_olda4_no_ca_joint_actual_loss_routing(self):self._check(LocusGSOldA4NoCA)


class BatchContractTests(unittest.TestCase):
    def test_old_four_view_contract_rejects_six_views(self):
        from scripts.smoke_olda4_ablation import validate_batch_contract
        window=dict(context=[0,1],novel=[2,3])
        batch=dict(images_input=torch.empty(1,2,3,256,256),images_all=torch.empty(1,4,3,256,256),images_output=torch.empty(1,2,3,256,256),frame_ids=torch.tensor([[0,1,2,3]]))
        self.assertEqual(validate_batch_contract(batch,window)['supervised_views'],4)
        batch['images_all']=torch.empty(1,6,3,256,256)
        with self.assertRaises(ValueError):validate_batch_contract(batch,window)
