import copy
from dataclasses import asdict,replace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeRenderer
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from tokengs.models.unified_instance_grounding import (
    RECIPE,LocusGSInstanceGroundedUnified,rendered_instance_grounding_loss)
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from tokengs.models.superpoint_structural_variants import build_structural_model
from scripts.export_object_locus_v3_set_official import assemble_panoptic


def batch_labels():
    return dict(semantic_label_all=torch.full((1,2,1,4),2),
        instance_label_all=torch.tensor([[[[1,1,2,2]],[[1,1,2,2]]]]))


def ideal_features():
    x=torch.zeros(1,2,8,1,4);x[:,:,0,:,:2]=1;x[:,:,1,:,2:]=1
    return x


class SpatialFeatureRenderer(FakeRenderer):
    """CPU test oracle: differentiable local mixing, not a CUDA renderer test."""
    def render_feature_channels(self,g,features,cam_view,intrinsics):
        b,v=cam_view.shape[:2]
        x=torch.linspace(-1,1,8,device=g.device)
        view=torch.linspace(-.2,.2,v,device=g.device)
        support=g[...,0]+torch.linspace(-1,1,g.shape[1],device=g.device)[None]
        weights=(-5*(x[None,None,None,:]-support[:,None,:,None]-view[None,:,None,None]).square()).softmax(2)
        alpha=g[...,3].mean(1)[:,None,None,None,None].expand(b,v,1,8,8)
        image=torch.einsum('bvnw,bnc->bvcw',weights,features)[:,:,:,None,:].expand(b,v,features.shape[-1],8,8)
        return dict(images_pred=image*alpha,alphas_pred=alpha)


class GroundingLossTests(unittest.TestCase):
    def test_pull_push_cross_have_expected_values(self):
        target=build_context_instance_targets(batch_labels());ideal=ideal_features()
        loss,m=rendered_instance_grounding_loss(ideal,target)
        self.assertEqual(float(loss),0.);self.assertEqual(m['instance_feature_cross_pairs'],2)
        collapsed=ideal.clone();collapsed[:,:,1]=0;collapsed[:,:,0]=1
        loss,m=rendered_instance_grounding_loss(collapsed,target)
        torch.testing.assert_close(loss,torch.tensor(1.));torch.testing.assert_close(m['loss_instance_push'],torch.tensor(1.))
        flipped=ideal.clone();flipped[:,1]*=-1
        loss,m=rendered_instance_grounding_loss(flipped,target)
        torch.testing.assert_close(m['loss_instance_cross'],torch.tensor(1.5))
        torch.testing.assert_close(loss,torch.tensor(3.))
        spread=ideal.clone();spread[:,:,:,:,1]*=-1;spread[:,:,:,:,3]*=-1
        loss,m=rendered_instance_grounding_loss(spread,target)
        torch.testing.assert_close(m['loss_instance_pull'],torch.tensor(.5))
        torch.testing.assert_close(loss,torch.tensor(2.))

    def test_view_id_batch_and_pixel_replication_invariance(self):
        torch.manual_seed(7);x=torch.randn(1,2,8,1,4)
        batch=batch_labels();targets=build_context_instance_targets(batch)
        a,_=rendered_instance_grounding_loss(x,targets)
        renamed=copy.deepcopy(batch);renamed['instance_label_all']=torch.where(batch['instance_label_all']==1,101,3)
        b,_=rendered_instance_grounding_loss(x,build_context_instance_targets(renamed));torch.testing.assert_close(a,b)
        reversed_batch={k:v.flip(1) for k,v in batch.items()}
        c,_=rendered_instance_grounding_loss(x.flip(1),build_context_instance_targets(reversed_batch));torch.testing.assert_close(a,c)
        repeated={k:v.repeat_interleave(3,-1) for k,v in batch.items()}
        d,_=rendered_instance_grounding_loss(x.repeat_interleave(3,-1),build_context_instance_targets(repeated));torch.testing.assert_close(a,d)
        twice={k:v.repeat(2,1,1,1) for k,v in batch.items()}
        e,_=rendered_instance_grounding_loss(x.repeat(2,1,1,1,1),build_context_instance_targets(twice));torch.testing.assert_close(a,e)

    def test_ignore_and_stuff_are_excluded(self):
        batch=batch_labels();batch['semantic_label_all'][:,:,:,0]=20;batch['semantic_label_all'][:,:,:,-1]=0
        targets=build_context_instance_targets(batch);x=torch.randn(1,2,8,1,4)
        a,m=rendered_instance_grounding_loss(x,targets)
        altered=x.clone();altered[:,:,:,:,0]=100;altered[:,:,:,:,-1]=-100
        b,_=rendered_instance_grounding_loss(altered,targets);torch.testing.assert_close(a,b)
        self.assertEqual(m['instance_feature_supervised_pixels'],4)

    def test_empty_one_instance_missing_views_and_zero_features_are_finite(self):
        for kind in ('empty','single','missing','zeros'):
            batch=batch_labels()
            if kind=='empty':batch['semantic_label_all'].zero_()
            elif kind=='single':batch['instance_label_all'].fill_(1)
            elif kind=='missing':batch['semantic_label_all'][:,1].zero_()
            x=(torch.zeros(1,2,8,1,4) if kind=='zeros' else torch.randn(1,2,8,1,4)).requires_grad_()
            loss,m=rendered_instance_grounding_loss(x,build_context_instance_targets(batch))
            self.assertTrue(torch.isfinite(loss));loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
            if kind=='empty':self.assertEqual(float(loss),0.)
            if kind=='missing':self.assertEqual(m['instance_feature_cross_pairs'],0)

    def test_loss_gradient_and_shape_validation(self):
        x=torch.randn(1,2,8,1,4,requires_grad=True)
        target=build_context_instance_targets(batch_labels())
        loss,_=rendered_instance_grounding_loss(x,target);loss.backward()
        self.assertGreater(float(x.grad.norm()),0.)
        for bad in (x[:,:1],x[:,:,:7],x[:,:,:,:,:3]):
            with self.assertRaises(ValueError):rendered_instance_grounding_loss(bad,target)
        bad=x.detach().clone();bad[...,0]=float('nan')
        with self.assertRaises(FloatingPointError):rendered_instance_grounding_loss(bad,target)


class GroundedUnifiedTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def model(self):
        return build_structural_model('unified_instance_grounding',opt().evolve(img_size=(8,8)),config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())

    def test_constructor_rng_common_parameters_and_native_outputs_are_unchanged(self):
        old=LocusGSUnifiedQueryV1(opt().evolve(img_size=(8,8)),config=config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())
        rng=torch.get_rng_state();torch.manual_seed(42);new=self.model()
        self.assertIsInstance(new,LocusGSInstanceGroundedUnified)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in old.state_dict().items():torch.testing.assert_close(new.state_dict()[name],value,rtol=0,atol=0)
        rgb=torch.rand(1,2,3,256,256);a=old.generate(rgb);b=new.generate(rgb)
        for key in ('gaussians','gaussian_mask_logits','thing_logits19','region_mass','semantic_scores','instance_eligible'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        for x,y in zip(assemble_panoptic(a),assemble_panoptic(b)):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(b['gaussian_instance_features'].shape[-1],8)
        self.assertEqual(b['rendered_instance_features'].shape,(1,2,8,8,8))

    def test_auxiliary_reaches_shared_features_and_geometry_but_not_classifier(self):
        model=self.model();out=model.generate(torch.rand(1,2,3,256,256))
        batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        batch['instance_label_all'][:,:,:,-4:]=2
        loss,_=rendered_instance_grounding_loss(out['rendered_instance_features'],build_context_instance_targets(batch))
        out['gaussians'].retain_grad();loss.backward()
        self.assertGreater(float(out['gaussians'].grad[...,:3].norm()),0.)
        for module in (model.core.instance_head,model.core.child_feature,model.core.blocks,model.core.activation_head,model.gfm_projection):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for p in model.core.classifier.parameters()))
        self.assertTrue(all(p.grad is None for p in model.core.query_mask.parameters()))
        self.assertTrue(all(p.grad is None for p in model.frozen_omega.parameters()))
        groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertTrue({id(p) for p in model.core.instance_head.parameters()}<={id(p) for p in groups['understanding']})

    def test_signed_features_render_without_probability_clamp_and_arbitrary_views(self):
        model=self.model();out=model.generate(torch.rand(1,2,3,256,256))
        with torch.no_grad():
            model.core.instance_head[1].weight.zero_();model.core.instance_head[1].bias.fill_(-2.)
        out=model.generate(torch.rand(1,2,3,256,256))
        self.assertTrue((out['rendered_instance_features']<0).all())
        rendered=model.render_generated_at(out,torch.eye(4).expand(1,3,4,4),torch.ones(1,3,4))
        self.assertEqual(rendered['rendered_instance_features'].shape,(1,3,8,8,8))

    def test_reconstruction_gradient_is_unchanged_and_auxiliary_head_is_independent(self):
        model=self.model();out=model.generate(torch.rand(1,2,3,256,256))
        out['gaussians'].square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.core.instance_head.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.core.activation_head.parameters()))

    def test_step_loss_effective_weight_and_no_novel_view_supervision(self):
        model=self.model();rgb=torch.rand(1,2,3,256,256)
        batch=dict(images_input=rgb,images_all=rgb,semantic_label_all=torch.full((1,3,8,8),2),
            instance_label_all=torch.ones(1,3,8,8,dtype=torch.long),has_mask=False,masks_all=torch.ones(1,2,1,256,256))
        batch['instance_label_all'][:,:,:,-4:]=2
        def calibration(images,p):return dict(c2w=torch.eye(4).expand(1,2,4,4),intrinsics=torch.ones(1,2,4),intrinsics_matrix=torch.eye(3).expand(1,2,3,3))
        def objective(states,decoder,supervision):
            value=model.activation_head(states[-1]['tokens'],states[-1]['mu'],states[-1]['radii']).square().mean()
            return value,{f'psnr_layer{model.supervised_layers[-1]}':value.detach()*0},None,None,None
        with patch.object(model,'calibrate_targets',side_effect=calibration),patch.object(model,'_layer_objective',side_effect=objective):
            _,metrics=model.step_loss(batch,understanding_weight=.05)
            torch.testing.assert_close(metrics['loss'],metrics['loss_recon']+.05*metrics['loss_understanding_base']+.0005*metrics['loss_instance_grounding'])
            self.assertAlmostEqual(metrics['instance_feature_effective_weight'],.0005)
            self.assertEqual(metrics['instance_feature_supervised_pixels'],128)
            metrics['loss'].backward()
            self.assertGreater(float(model.core.instance_head[1].weight.grad.norm()),0.)

    def test_checkpoint_roundtrip_and_recipe_guard(self):
        from scripts.train_unified_query_v1 import trainable_state,restore_state
        from scripts.unified_query_initial_state import validate_zero_initial
        model=self.model();saved_state=trainable_state(model)
        self.assertTrue(any(k.startswith('core.instance_head.') for k in saved_state))
        other=self.model();restore_state(other,saved_state)
        for k,v in trainable_state(other).items():torch.testing.assert_close(v,saved_state[k],rtol=0,atol=0)
        meta=dict(architecture=model.architecture_name,config=model.config.to_dict(),manifest_sha256='train',heldout_sha256='heldout',world_size=1,
            omega_identity=model.frozen_omega.source_identity,sam_identity=None,**model.extra_checkpoint_metadata)
        saved=dict(meta=meta,progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer=dict(state={}),rank_rng=[{}])
        def validate(x):validate_zero_initial(x,model,manifest_sha256='train',heldout_sha256='heldout',world=1)
        validate(saved)
        bad=copy.deepcopy(saved);bad['meta']['instance_grounding_recipe']['push_margin']=0.
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(saved);bad['meta'].pop('instance_grounding_recipe')
        with self.assertRaises(ValueError):validate(bad)
        base=LocusGSUnifiedQueryV1(opt(),config=config(),omega=FakeOmega(),renderer=FakeRenderer())
        with self.assertRaises(ValueError):restore_state(other,trainable_state(base))


if __name__=='__main__':unittest.main()
