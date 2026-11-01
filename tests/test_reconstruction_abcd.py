"""Independent forward, gradient and resource-contract checks for ABCD."""
import unittest
import torch
from unittest.mock import patch
from tokengs.models.reconstruction_abcd import activate_geometry,activate_appearance,variants
from tokengs.rendering.sh1 import sh1_rgb,SH1Renderer
from scripts.reconstruction_abcd_policy import expanded
from tokengs.models.dual_stream_initialization import seed_anchors

class Contracts(unittest.TestCase):
    def test_expansion_preserves_old_rows_and_pads_only_new_moments(self):
        old=torch.randn(1024,1024)
        result=expanded(old,'parameter')
        torch.testing.assert_close(result[:1024],old,rtol=0,atol=0)
        self.assertFalse(torch.equal(result[1024:2048],old))
        torch.testing.assert_close(result,expanded(old,'parameter'),rtol=0,atol=0)
        moment=expanded(old,'moment');self.assertEqual(int(moment[1024:].count_nonzero()),0)
        torch.testing.assert_close(moment[:1024],old,rtol=0,atol=0)

    def test_pool_fallback_shared_and_unique(self):
        y,x=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
        p=torch.stack((x,y,torch.ones_like(x)),-1).float()[None,None].repeat(1,2,1,1,1);p[:,1,:, :,2]=2
        confidence=torch.ones(1,2,8,8);rgb=torch.rand(1,2,3,8,8)
        a=seed_anchors(p,confidence,rgb,count=8,candidate_side=2,return_source_indices=True,minimum_candidates=64)
        b=seed_anchors(p,confidence,rgb,count=64,candidate_side=2,return_source_indices=True,minimum_candidates=64)
        self.assertEqual(b[3].unique().numel(),64)
        torch.testing.assert_close(a[3],b[3][:,:8],rtol=0,atol=0)
        with self.assertRaises(ValueError):seed_anchors(p[:,:,:1,:1],confidence[:,:,:1,:1],rgb,count=8,minimum_candidates=64)

    def test_sh1_matches_analytic_view_dependence_and_propagates_gradient(self):
        sh=torch.randn(1,2,4,3,requires_grad=True)*.01
        d=torch.nn.functional.normalize(torch.randn(1,2,3),dim=-1)
        expected=.5+.28209479177387814*sh[...,0,:]+.4886025119029199*(-d[...,1,None]*sh[...,1,:]+d[...,2,None]*sh[...,2,:]-d[...,0,None]*sh[...,3,:])
        actual=sh1_rgb(sh,d)
        torch.testing.assert_close(actual,expected.clamp_min(0.))
        self.assertFalse(torch.equal(actual,sh1_rgb(sh,-d)))
        grad,=torch.autograd.grad(actual.sum(),sh);self.assertTrue(torch.all(grad.abs().sum(-1)>0))

    def test_sh_renderer_calls_same_backend_with_per_camera_rgb(self):
        from tokengs.options import config_defaults
        opt=config_defaults['train_siu3r_unified_query_v1'];renderer=SH1Renderer(opt)
        g=torch.zeros(1,2,23);g[...,2]=1;g[...,20]=.1
        cam=torch.eye(4).repeat(1,2,1,1);cam[:,1,3,0]=-1.
        captured=[]
        def fake(self,gs,cam,bg,intr):
            captured.append(gs.clone());return {'images_pred':gs[...,11:14].mean(1)[:,None,:,None,None]}
        with patch('tokengs.rendering.gs.GaussianRenderer.render',new=fake):
            out=renderer.render(g,cam,intrinsics=torch.ones(1,2,4))
        self.assertEqual(out['images_pred'].shape,(1,2,3,1,1));self.assertFalse(torch.equal(captured[0][...,11:],captured[1][...,11:]))
        torch.testing.assert_close(captured[0][...,:11],g[...,:11],rtol=0,atol=0)

    def test_D_same_gaussians_feed_semantics_and_both_loss_directions(self):
        from dataclasses import replace
        from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
        from tokengs.models.unified_query_loss import unified_instance_loss
        from tokengs.models.input_types import ModelInputDecoder,ModelSupervision
        cfg=replace(config(),dim=256,heads=8,depth=12,superpoint_source='esam_sam',matching_mode='source_many_to_one')
        class GeometryRenderer(FakeRenderer):
            def render(self,g,cam_view,bg_color,intrinsics):
                out=super().render(g,cam_view,bg_color,intrinsics)
                # Synthetic differentiable position response; actual gsplat
                # projection and geometry derivatives belong to the GPU gate.
                out['images_pred']=out['images_pred']+.01*g[...,:3].mean()
                return out
        model=variants()['reconstruction_d'](opt().evolve(dec_patch_size=8,img_size=(8,8)),config=cfg,omega=FakeOmega(),sam=FakeSAM(),renderer=GeometryRenderer())
        # Only shrink the query batch for this CPU integration check; all actual
        # geometry/appearance/semantic layers, activations and losses run.
        model.core.count=6
        model.core.gs_tokens=torch.nn.Parameter(model.core.gs_tokens[:6].detach().clone())
        model.core.appearance.tokens=torch.nn.Parameter(model.core.appearance.tokens[:6].detach().clone())
        for layer in model.core.panoptic.layers.values():torch.nn.init.normal_(layer.image_attention.O.weight,std=.01)
        torch.nn.init.normal_(model.core.panoptic.child_mlp[-1].weight,std=.01)
        def small_memory(layers,rgb):
            m=model.core.semantic_memory
            patches=sum(w*proj(x.float().mean(-2)) for w,proj,x in zip(m.layer_logits.softmax(0),m.projections,layers))
            grid=patches.reshape(2,256,1,1).expand(-1,-1,8,8)
            return m.refine(grid+m.local(torch.nn.functional.interpolate(rgb.flatten(0,1),size=(8,8),mode='bilinear',align_corners=False))).reshape(1,2,256,8,8)
        with patch.object(model.core.semantic_memory,'forward',side_effect=small_memory):
            out=model.generate(torch.rand(1,2,3,256,256))
            self.assertEqual(out['gaussians'].shape,(1,384,23))
            torch.testing.assert_close(out['gaussians'],out['states'][-1]['gaussians'],rtol=0,atol=0)
            torch.testing.assert_close(out['gaussian_parent_anchor'],torch.arange(6).repeat_interleave(64))
            groups=model.parameter_groups()
            batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
            lu,_=unified_instance_loss(out,batch,cfg);lu.backward(retain_graph=True)
            def energy(ps):return sum(float(p.grad.square().sum()) for p in ps if p.grad is not None)
            self.assertEqual(energy(groups['reconstruction']),0.)
            self.assertGreater(energy(groups['understanding']),0.)
            model.zero_grad(set_to_none=True)
            decoder=ModelInputDecoder(cam_view=torch.eye(4).expand(1,2,4,4),intrinsics=torch.tensor([4.,4.,4.,4.]).expand(1,2,4))
            supervision=ModelSupervision(images_output=torch.rand(1,2,3,8,8),masks_output=torch.ones(1,2,1,8,8),has_mask=torch.tensor([False]),rays_os=torch.zeros(1,2,3,8,8),rays_ds=torch.ones(1,2,3,8,8))
            loss,*_=model._layer_objective(out['states'],decoder,supervision);loss.backward()
            self.assertGreater(energy(groups['reconstruction']),0.)
            self.assertEqual(energy(groups['understanding']),0.)
            self.assertEqual(energy(groups['shared']),0.)
            self.assertGreater(float(model.core.geometry_head.weight.grad.norm()),0.)
            self.assertGreater(float(model.core.appearance.output.weight.grad.norm()),0.)

    def test_geometry_activation_no_double_offset_and_alpha_contract(self):
        raw=torch.randn(1,64,10)*.01;p,s,r=activate_geometry(raw)
        torch.testing.assert_close(p[...,:2],raw[...,:2].sign()*raw[...,:2].abs().expm1())
        torch.testing.assert_close(p[...,2]-1,raw[...,2].sign()*raw[...,2].abs().expm1(),atol=1e-7,rtol=1e-5)
        self.assertTrue(torch.all((s>0)&(s<=.075)))
        torch.testing.assert_close(r.norm(dim=-1),torch.ones(1,64))
        color,alpha=activate_appearance(torch.zeros(1,64,13));self.assertEqual(color.shape,(1,64,12))
        torch.testing.assert_close(alpha,torch.full((1,64,1),torch.sigmoid(torch.tensor(-2.))))
        self.assertEqual(set(variants()),{'reconstruction_'+a for a in 'abcd'})

if __name__=='__main__':unittest.main()
