import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
from tokengs.models.direct_gaussian_mask import DirectGaussianMaskField
from tokengs.models.superpoint_structural_variants import LocusGSSuperpointDual,LocusGSDirectGaussian
from tokengs.models.dynamic_panoptic_controller import DynamicPanopticController


class StructuralTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_legacy_object_layer_matches_original_at_100_queries(self):
        from tokengs.models.object_locus_panoptic_v1_controller import RegisteredObjectLayer
        from tokengs.models.dynamic_panoptic_controller import DynamicObjectLayer
        original=RegisteredObjectLayer();dynamic=DynamicObjectLayer();dynamic.load_state_dict(original.state_dict())
        args=(torch.randn(1,5,1024),torch.randn(1,5,256),torch.randn(1,5,3),torch.randn(1,102,256),
            torch.randn(1,100,3),torch.ones(1,100,3)*.2,torch.ones(1),torch.randn(1,7,256),torch.nn.Identity(),torch.randn(1,5,256),100)
        a,b=original(*args),dynamic(*args)
        for key in ('q','c','s','route','joint_delta','anchor_mask_logits'):torch.testing.assert_close(a[key],b[key],rtol=1e-5,atol=1e-6)

    def test_direct_field_permutation_chunking_and_geometry_gradients(self):
        field=DirectGaussianMaskField(16,hidden=8,chunk=3)
        q=torch.randn(1,3,16,requires_grad=True);g=torch.randn(1,7,14,requires_grad=True)
        with torch.no_grad():g[...,4:7].abs_().add_(.01)
        centers=torch.randn(1,3,3);scale=torch.tensor(.5)
        a=field(q,g,centers,scale)
        pi=torch.tensor([4,0,3,1,6,2,5]);qi=torch.tensor([2,0,1])
        b=field(q[:,qi],g[:,pi],centers[:,qi],scale)
        torch.testing.assert_close(b,a[:,pi][:,:,qi])
        a.square().mean().backward();self.assertGreater(float(q.grad.norm()),0.);self.assertGreater(float(g.grad[...,:3].norm()),0.)
        field.chunk=100;c=field(q,g,centers,scale);torch.testing.assert_close(a,c)

    def test_dynamic_legacy_controller_uses_all_queries_beyond_100(self):
        controller=DynamicPanopticController();out=controller.classify(torch.randn(1,107,256))
        self.assertEqual(out['thing_logits19'].shape,(1,105,19))
        layer=controller.layers['L6'];n=3
        output=layer(torch.randn(1,5,1024),torch.randn(1,5,256),torch.randn(1,5,3),torch.randn(1,n+2,256),
            torch.randn(1,n,3),torch.ones(1,n,3)*.2,torch.ones(1),torch.randn(1,7,256),torch.nn.Identity(),torch.randn(1,5,256),0)
        self.assertEqual(output['q'].shape,(1,5,256));self.assertEqual(output['c'].shape,(1,3,3));self.assertEqual(output['route'].shape,(1,5,6))

    def test_direct_model_predicts_semantics_instances_and_retains_geometry(self):
        cfg=replace(config(),superpoint_source='esam_sam',matching_mode='source_many_to_one')
        model=LocusGSDirectGaussian(opt(),config=cfg,omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer())
        out=model.generate(torch.rand(1,2,3,256,256));n=out['num_thing_queries']
        self.assertEqual(out['gaussian_semantics'].shape,(1,n*4,20));self.assertEqual(out['semantic_scores'].shape,(1,2,20,8,8))
        self.assertEqual(out['gaussian_membership'].shape,(1,n*4,n+2))
        self.assertFalse(hasattr(model.core,'query_mask'));self.assertFalse(hasattr(model.core,'child_feature'))
        loss=out['gaussian_mask_logits'].square().mean()+out['semantic_scores'][:,:,2].mean();loss.backward()
        self.assertGreater(float(model.core.instance_field.parameters_from_query.weight.grad.norm()),0.)
        self.assertGreater(float(model.core.gaussian_semantic_head.weight.grad.norm()),0.)
        groups=model.parameter_groups();ids=[id(p) for group in groups.values() for p in group]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})

    def test_direct_semantic_loss_participates_in_joint_training(self):
        model=LocusGSDirectGaussian(opt(),config=config(),omega=FakeOmega(),renderer=FakeRenderer())
        rgb=torch.rand(1,2,3,256,256)
        batch=dict(images_input=rgb,images_all=rgb,semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long),has_mask=False,masks_all=torch.ones(1,2,1,256,256))
        def calibration(images,p):return dict(c2w=torch.eye(4).expand(1,2,4,4),intrinsics=torch.ones(1,2,4),intrinsics_matrix=torch.eye(3).expand(1,2,3,3))
        def objective(states,decoder,supervision):
            value=model.activation_head(states[-1]['tokens'],states[-1]['mu'],states[-1]['radii']).square().mean()
            return value,{f'psnr_layer{model.supervised_layers[-1]}':value.detach()*0},None,None,None
        with patch.object(model,'calibrate_targets',side_effect=calibration),patch.object(model,'_layer_objective',side_effect=objective):
            _,metrics=model.step_loss(batch,understanding_weight=.05);self.assertGreater(float(metrics['loss_semantic_direct_ce']),0.)
            metrics['loss'].backward();self.assertGreater(float(model.core.gaussian_semantic_head.weight.grad.norm()),0.)

    def test_legacy_dual_end_to_end_context_only_seeds_readout_and_checkpoint(self):
        cfg=replace(config(),dim=256,heads=8,depth=12,superpoint_source='esam_sam',matching_mode='source_many_to_one')
        model=LocusGSSuperpointDual(opt().evolve(dec_patch_size=8,img_size=(8,8)),config=cfg,omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer())
        out=model.generate(torch.rand(1,2,3,256,256));n=out['num_thing_queries']
        self.assertEqual(n,2);self.assertEqual(out['gaussians'].shape,(1,n*64,14));self.assertEqual(out['gaussian_membership'].shape,(1,n*64,n+2))
        self.assertEqual(out['states'][-1]['tokens'].shape[-1],1024)
        loss=out['gaussians'].square().mean()+out['thing_logits19'].square().mean()+out['gaussian_membership'].mean();loss.backward()
        for initializer in (model.core.initializer.reconstruction,model.core.initializer.understanding):
            self.assertGreater(float(initializer.point_encoder.layers[0][0].weight.grad.norm()),0.)
        groups=model.parameter_groups();ids=[id(p) for group in groups.values() for p in group]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
        self.assertGreater(float(model.gfm_projection[1].weight.grad.norm()),0.)


if __name__=='__main__':unittest.main()
