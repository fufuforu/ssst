"""Coupling mechanisms: controlled gradients, adjoint visibility and losses."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
from torch import nn

from tests.test_unified_query_v1 import config,opt,FakeOmega
from tests.test_unified_instance_grounding import SpatialFeatureRenderer
from tokengs.models.unified_instance_grounding import LocusGSInstanceGroundedUnified
from tokengs.models.unified_instance_coupling import (
    LocusGSInstanceLift,LocusGSInstanceBoundary,LocusGSInstanceBidirectional,
    instance_boundary_weights,boundary_rgb_loss)
from tokengs.models.object_locus_panoptic_v1_lift import lift_features
from tokengs.models.input_types import ModelInputDecoder


def make(cls):
    return cls(opt().evolve(img_size=(8,8)),config=config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())


class CouplingTests(unittest.TestCase):
    def test_probe_diagnostics_use_gt_only_after_generation_and_preserve_rng(self):
        model=make(LocusGSInstanceBoundary)
        rgb=torch.rand(1,2,3,256,256);p=model.generate(rgb)
        labels=torch.ones(1,2,8,8,dtype=torch.long);labels[...,4:]=2
        batch=dict(instance_label_all=labels,semantic_label_all=torch.full_like(labels,2),
                   images_input=rgb[:,:,:,:8,:8])
        p['final_targets']=dict(valid_pixels=torch.ones(1,2,8,8,dtype=torch.bool),
            gt_pixel_masks=[torch.stack((labels[0]==1,labels[0]==2))])
        state=torch.get_rng_state();metrics=model.probe_diagnostics(p,batch)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(metrics['instance_feature_evaluated_gt'],2)
        self.assertGreater(metrics['context_gt_boundary_pixels'],0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

    def test_boundaries_are_symmetric_local_and_detached(self):
        x=torch.zeros(1,2,8,16,16);x[:,:,0,:,:8]=1;x[:,:,1,:,8:]=1;x.requires_grad_()
        w=instance_boundary_weights(x)
        self.assertFalse(w.requires_grad)
        self.assertGreater(float(w[...,7:9].mean()),.99)
        self.assertLess(float(w[...,:4].mean()),.05)
        torch.testing.assert_close(w,w.flip(-1))
        rgb=torch.rand(1,2,3,16,16,requires_grad=True)
        loss,_=boundary_rgb_loss(rgb,torch.zeros_like(rgb),x)
        loss.backward();self.assertIsNone(x.grad);self.assertGreater(float(rgb.grad.norm()),0.)

    def test_uniform_boundary_weights_reduce_to_uniform_charbonnier_and_no_wrap(self):
        x=torch.ones(1,2,8,8,8);rgb=torch.rand(1,2,3,8,8)
        loss,w=boundary_rgb_loss(rgb,torch.zeros_like(rgb),x)
        torch.testing.assert_close(w,w.mean().expand_as(w))
        torch.testing.assert_close(loss,(rgb.square()+1e-6).sqrt().mean())
        x[:,:,:,:,0]*=-1;w=instance_boundary_weights(x)
        self.assertGreater(float(w[...,0].mean()),.99);self.assertLess(float(w[...,-1].mean()),.05)
        with self.assertRaises(ValueError):boundary_rgb_loss(rgb[:,:,:,:,:7],rgb,x)

    def test_all_common_initial_weights_rng_groups_and_boundary_native_predictions(self):
        models=[];rng=[]
        for cls in (LocusGSInstanceGroundedUnified,LocusGSInstanceLift,LocusGSInstanceBoundary,LocusGSInstanceBidirectional):
            torch.manual_seed(42);models.append(make(cls));rng.append(torch.get_rng_state())
        for model,state in zip(models[1:],rng[1:]):
            self.assertTrue(torch.equal(state,rng[0]))
            for name,p in models[0].state_dict().items():torch.testing.assert_close(model.state_dict()[name],p,rtol=0,atol=0)
            groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        rgb=torch.rand(1,2,3,256,256);a=models[0].generate(rgb);b=models[2].generate(rgb)
        for key in ('gaussians','gaussian_mask_logits','thing_logits19','rendered_instance_features'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        for model in models:
            from scripts.train_unified_query_v1 import trainable_state,restore_state
            restore_state(model,trainable_state(model))
        with self.assertRaises(ValueError):restore_state(models[1],trainable_state(models[0]))

    def test_lift_actual_feature_gradient_and_no_geometry_gradient(self):
        model=make(LocusGSInstanceLift);p=model.generate(torch.rand(1,2,3,256,256))
        self.assertGreater(float(p['lift_residual'].norm()),0.)
        a,b=torch.autograd.grad(p['lift_residual'].square().mean(),
            (model.gfm_projection[1].weight,p['gaussians']),allow_unused=True,retain_graph=True)
        self.assertGreater(float(a.norm()),0.);self.assertIsNone(b)
        p['gaussian_membership'].mean().backward()
        self.assertGreater(float(model.core.lift_residual[1].weight.grad.norm()),0.)
        self.assertGreater(sum(float(q.grad.abs().sum()) for q in model.core.activation_head.parameters() if q.grad is not None),0.)
        self.assertTrue(all(q.grad is None for q in model.frozen_omega.parameters()))
        model.eval()
        with torch.no_grad():
            out=model.generate(torch.rand(1,2,3,256,256))
            self.assertTrue(torch.isfinite(out['region_mass']).all())

    def test_adjoint_matches_pixel_dot_product_and_invisible_gaussians_have_no_evidence(self):
        class LinearRenderer(nn.Module):
            def render_feature_channels(self,g,f,cam,intr):
                a=torch.tensor([[1.,.5,0.],[0.,.5,0.]])
                image=torch.einsum('pn,bnc->bcp',a,f).reshape(1,1,f.shape[-1],1,2)
                return dict(images_pred=image,alphas_pred=torch.ones(1,1,1,1,2))
        renderer=LinearRenderer();g=torch.randn(1,3,14,requires_grad=True)
        pixels=torch.tensor([[[[[2.,4.]],[[6.,8.]]]]],requires_grad=True)
        decoder=ModelInputDecoder(cam_view=torch.eye(4).reshape(1,1,4,4),intrinsics=torch.ones(1,1,4))
        evidence,gate,mass=lift_features(pixels,g,decoder,renderer)
        torch.testing.assert_close(evidence,torch.tensor([[[2.,6.],[3.,7.],[0.,0.]]]))
        self.assertEqual(float(gate[0,2]),0.)
        evidence.sum().backward();self.assertIsNone(g.grad)
        torch.testing.assert_close(pixels.grad,torch.tensor([[[[[1.5,.5]],[[1.5,.5]]]]]))

    def test_boundary_loss_uses_context_only_independent_weight_and_ramp(self):
        model=make(LocusGSInstanceBidirectional);rgb=torch.rand(1,2,3,256,256)
        batch=dict(images_input=rgb,images_all=rgb,semantic_label_all=torch.full((1,3,8,8),2),
                   instance_label_all=torch.ones(1,3,8,8,dtype=torch.long),has_mask=False,masks_all=torch.ones(1,2,1,256,256))
        batch['instance_label_all'][:,:,:,-4:]=2
        def calibration(images,p):return dict(c2w=torch.eye(4).expand(1,2,4,4),intrinsics=torch.ones(1,2,4),intrinsics_matrix=torch.eye(3).expand(1,2,3,3))
        def objective(states,decoder,supervision):
            value=model.activation_head(states[-1]['tokens'],states[-1]['mu'],states[-1]['radii']).square().mean()
            return value,{f'psnr_layer{model.supervised_layers[-1]}':value.detach()*0},None,None,None
        # Synthetic RGB renderer has an 8x8 output; use the corresponding
        # context target for this loss without changing the real input contract.
        import tokengs.models.unified_instance_coupling as coupling
        real=coupling.boundary_rgb_loss
        def small_rgb(pred,target,features,recipe):
            return real(pred,torch.nn.functional.interpolate(target.flatten(0,1),size=(8,8)).reshape(1,2,3,8,8),features,recipe)
        with patch.object(model,'calibrate_targets',side_effect=calibration),patch.object(model,'_layer_objective',side_effect=objective),patch.object(coupling,'boundary_rgb_loss',side_effect=small_rgb):
            for weight,coefficient in ((0.,0.),(.025,.01),(.05,.02)):
                _,m=model.step_loss(batch,understanding_weight=weight)
                self.assertAlmostEqual(m['boundary_rgb_effective_weight'],coefficient)
                torch.testing.assert_close(m['loss'],m['loss_recon_base']+coefficient*m['loss_boundary_rgb']+weight*m['loss_understanding'])
                m['loss'].backward();model.zero_grad(set_to_none=True)


if __name__=='__main__':unittest.main()
