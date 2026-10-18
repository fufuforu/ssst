import copy
from dataclasses import asdict,replace
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F

from tests.test_unified_query_v1 import config,opt,cloud,FakeOmega
from tests.test_unified_instance_grounding import SpatialFeatureRenderer
from tokengs.models.direct_instance_attributes import RECIPE,DirectInstanceAttributeCore,LocusGSDirectInstanceAttributes
from tokengs.models.unified_instance_grounding import LocusGSInstanceGroundedUnified,rendered_instance_grounding_loss
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.instance_clustering import RECIPE as CLUSTER_RECIPE


def labels():
    data=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
    data['instance_label_all'][:,:,:,-4:]=2
    return data


class DirectAttributeTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def model(self):
        return build_structural_model('unified_direct_instance_attributes',opt().evolve(img_size=(8,8)),config(),
            omega=FakeOmega(),renderer=SpatialFeatureRenderer())

    def test_shared_initialization_and_reconstruction_preserved(self):
        base=LocusGSInstanceGroundedUnified(opt().evolve(img_size=(8,8)),config=config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())
        rng=torch.get_rng_state();torch.manual_seed(42);new=self.model()
        self.assertIsInstance(new,LocusGSDirectInstanceAttributes);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for name,value in new.state_dict().items():
            if name in base.state_dict():torch.testing.assert_close(value,base.state_dict()[name],rtol=0,atol=0)
        rgb=torch.rand(1,2,3,256,256);a=base.generate(rgb);b=new.generate(rgb)
        for key in ('gaussians','thing_logits19','unified_query'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        self.assertIs(b['gaussian_instance_attributes'],b['gaussian_instance_features'])
        self.assertEqual(b['gaussians'].shape[-1],14)
        for name in ('instance_head','child_feature','child_index','query_mask','stuff_feature'):
            self.assertFalse(hasattr(new.core,name))

    def test_each_child_has_independent_direct_attribute_channels(self):
        core=DirectInstanceAttributeCore(opt(),config());q=torch.zeros(1,3,16)
        with torch.no_grad():
            core.instance_attribute_head[1].weight.zero_()
            core.instance_attribute_head[1].bias.copy_(torch.arange(32).float()-16)
        r=core.instance_readout(q)
        self.assertEqual(r['gaussian_instance_attributes'].shape,(1,12,8))
        torch.testing.assert_close(r['gaussian_instance_attributes'][0,:4],torch.arange(32).reshape(4,8).float()-16)
        self.assertEqual(r['query_instance_prototypes'].shape,(1,3,8));self.assertEqual(r['gaussian_membership'].shape,(1,12,5))

    def test_attributes_do_not_depend_on_explicit_geometry_or_geometry_heads(self):
        core=DirectInstanceAttributeCore(opt(),config());c=cloud();memory=torch.randn(1,5,16);ids=torch.tensor([0,0,1,1])
        a=core(c,memory,ids)
        grads=torch.autograd.grad(a['gaussian_instance_attributes'].square().mean(),
            [a['gaussians'],a['states'][-1]['mu'],a['states'][-1]['radii']]+list(core.activation_head.parameters())+list(core.geometry.parameters()),allow_unused=True)
        self.assertTrue(all(g is None for g in grads))
        with torch.no_grad():
            core.activation_head.deconv.bias.add_(.1)
            for head in core.geometry:head.bias.add_(.1)
        b=core(c,memory,ids)
        self.assertFalse(torch.equal(a['gaussians'],b['gaussians']))
        for key in ('gaussian_instance_attributes','query_instance_prototypes','gaussian_mask_logits'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_all_query_cosine_membership_and_stuff_contract(self):
        core=DirectInstanceAttributeCore(opt(),config());q=torch.randn(1,3,16);r=core.instance_readout(q)
        e=F.normalize(r['gaussian_instance_attributes'],dim=-1);p=F.normalize(r['query_instance_prototypes'],dim=-1)
        expected=10*(e@p.transpose(1,2))
        torch.testing.assert_close(r['gaussian_mask_logits'][...,:3],expected)
        torch.testing.assert_close(r['gaussian_membership'],r['gaussian_mask_logits'].sigmoid())
        self.assertTrue(torch.isfinite(r['gaussian_membership']).all())
        with torch.no_grad():
            core.instance_prototype_head[1].weight.zero_();core.instance_prototype_head[1].bias.fill_(1.)
        r=core.instance_readout(q)
        torch.testing.assert_close(r['gaussian_membership'][...,0],r['gaussian_membership'][...,1])
        # Duplicate prototypes are permitted; independent sigmoids need not sum to 1.
        self.assertFalse(torch.allclose(r['gaussian_membership'].sum(-1),torch.ones(1,12)))

    def test_membership_gradients_reach_attributes_prototypes_and_shared_decoder(self):
        model=self.model();p=model.generate(torch.rand(1,2,3,256,256))
        (p['region_mass'].square().mean()+p['thing_logits19'].square().mean()).backward()
        for module in (model.core.instance_attribute_head,model.core.instance_prototype_head,model.core.stuff_attribute_head,model.core.blocks,model.gfm_projection):
            self.assertGreater(sum(float(x.grad.abs().sum()) for x in module.parameters() if x.grad is not None),0.)
        self.assertIsNotNone(model.core.instance_logit_scale.grad);self.assertIsNotNone(model.core.instance_logit_bias.grad)
        self.assertTrue(all(x.grad is None for x in model.frozen_omega.parameters()))

    def test_feature_loss_reaches_direct_attributes_and_geometry_but_not_prototype(self):
        model=self.model();p=model.generate(torch.rand(1,2,3,256,256));p['gaussians'].retain_grad()
        loss,_=rendered_instance_grounding_loss(p['rendered_instance_features'],build_context_instance_targets(labels()))
        loss.backward()
        for module in (model.core.instance_attribute_head,model.core.activation_head,model.core.geometry,model.core.blocks,model.gfm_projection):
            self.assertGreater(sum(float(x.grad.abs().sum()) for x in module.parameters() if x.grad is not None),0.)
        self.assertGreater(float(p['gaussians'].grad[...,:3].norm()),0.)
        self.assertTrue(all(x.grad is None for x in model.core.instance_prototype_head.parameters()))
        self.assertTrue(all(x.grad is None for x in model.core.stuff_attribute_head.parameters()))

    def test_rgb_gradient_does_not_update_instance_output_heads(self):
        model=self.model();p=model.generate(torch.rand(1,2,3,256,256));p['gaussians'].square().mean().backward()
        for head in (model.core.instance_attribute_head,model.core.instance_prototype_head,model.core.stuff_attribute_head):
            self.assertTrue(all(x.grad is None for x in head.parameters()))
        self.assertTrue(any(x.grad is not None for x in model.core.activation_head.parameters()))

    def test_signed_render_and_both_inference_methods_use_same_attributes(self):
        model=self.model().eval()
        with torch.no_grad():
            model.core.instance_attribute_head[1].weight.zero_();model.core.instance_attribute_head[1].bias.fill_(-2.)
        p=model.generate(torch.rand(1,2,3,256,256));self.assertTrue((p['rendered_instance_features']<0).all())
        before={n:x.detach().clone() for n,x in model.named_parameters()};versions={n:x._version for n,x in model.named_parameters()}
        prototype=model.infer_instances(p,method='prototype')
        cluster=model.infer_instances(p,method='cluster',cluster_recipe=replace(CLUSTER_RECIPE,min_cluster_size=4,min_samples=2,n_jobs=1))
        self.assertEqual(prototype['raw_membership_masks'].shape[:2],(1,2))
        self.assertEqual(prototype['instance_eligible'].ndim,1)
        self.assertEqual(cluster['instance_labels'].shape,(1,2,8,8));self.assertTrue(cluster['class_agnostic'])
        for row in cluster['cluster_reports'][0]:self.assertFalse(row['gt_used']);self.assertTrue(row['post_render_l2'])
        for n,x in model.named_parameters():
            torch.testing.assert_close(x,before[n],rtol=0,atol=0);self.assertEqual(x._version,versions[n]);self.assertIsNone(x.grad)
        novel=model.infer_instances(p,method='cluster',cam_view=torch.eye(4).expand(1,3,4,4),intrinsics=torch.ones(1,3,4))
        self.assertEqual(novel['instance_labels'].shape,(1,3,8,8))
        novel=model.infer_instances(p,method='prototype',cam_view=torch.eye(4).expand(1,3,4,4),intrinsics=torch.ones(1,3,4))
        self.assertEqual(novel['instance_labels'].shape,(1,3,8,8))
        from scripts.evaluate_direct_instance_attributes import evaluate_readout_outputs
        metrics=evaluate_readout_outputs(dict(prototype=prototype,cluster=cluster),build_context_instance_targets(labels()))
        self.assertEqual(set(metrics),{'prototype','prototype_raw','prototype_nms','cluster'})
        for v in range(2):
            self.assertEqual(metrics['cluster'][v]['gt_count'],2)
            self.assertLessEqual(metrics['prototype_nms'][v]['pred_count'],metrics['prototype_raw'][v]['pred_count'])
        with self.assertRaises(ValueError):model.infer_instances(p,method='other')
        with self.assertRaises(ValueError):model.infer_instances(p,cam_view=torch.eye(4))

    def test_step_loss_recipe_effective_weight_groups_and_all_parameter_updates(self):
        model=self.model();rgb=torch.rand(1,2,3,256,256);batch=dict(labels(),images_input=rgb,images_all=rgb,has_mask=False,masks_all=torch.ones(1,2,1,256,256))
        def calibration(images,p):return dict(c2w=torch.eye(4).expand(1,2,4,4),intrinsics=torch.ones(1,2,4),intrinsics_matrix=torch.eye(3).expand(1,2,3,3))
        def objective(states,decoder,supervision):
            value=model.activation_head(states[-1]['tokens'],states[-1]['mu'],states[-1]['radii']).square().mean()
            return value,{f'psnr_layer{model.supervised_layers[-1]}':value.detach()*0},None,None,None
        with patch.object(model,'calibrate_targets',side_effect=calibration),patch.object(model,'_layer_objective',side_effect=objective):
            _,m=model.step_loss(batch,understanding_weight=.05)
        torch.testing.assert_close(m['loss'],m['loss_recon']+.05*m['loss_understanding_base']+.0005*m['loss_instance_grounding'])
        self.assertNotIn('loss_gaussian_direct',m);self.assertTrue(torch.isfinite(m['instance_attribute_norm_mean']))
        m['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
        groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})

    def test_strict_checkpoint_roundtrip_and_recipe_guards(self):
        from scripts.train_unified_query_v1 import trainable_state,restore_state
        from scripts.unified_query_initial_state import validate_zero_initial
        model=self.model();state=trainable_state(model);other=self.model();restore_state(other,state)
        for k,v in trainable_state(other).items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
        meta=dict(architecture=model.architecture_name,config=model.config.to_dict(),manifest_sha256='train',heldout_sha256='heldout',world_size=1,
            omega_identity=model.frozen_omega.source_identity,sam_identity=None,**model.extra_checkpoint_metadata)
        blob=dict(meta=meta,progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer=dict(state={}),rank_rng=[{}])
        validate_zero_initial(blob,model,manifest_sha256='train',heldout_sha256='heldout',world=1)
        bad=copy.deepcopy(blob);bad['meta']['direct_instance_attribute_recipe']['embedding_dim']=16
        with self.assertRaises(ValueError):validate_zero_initial(bad,model,manifest_sha256='train',heldout_sha256='heldout',world=1)
        base=LocusGSInstanceGroundedUnified(opt(),config=config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())
        with self.assertRaises(ValueError):restore_state(model,trainable_state(base))
        self.assertEqual(meta['direct_instance_attribute_recipe'],asdict(RECIPE))


if __name__=='__main__':unittest.main()
