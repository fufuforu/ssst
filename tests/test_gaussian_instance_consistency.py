from dataclasses import replace
import copy,unittest
from unittest.mock import patch
import torch
from torch import nn

from tests.test_unified_query_v1 import config,opt,FakeOmega
from tests.test_unified_instance_grounding import SpatialFeatureRenderer
from tokengs.models.input_types import ModelInputDecoder
from tokengs.models.unified_instance_grounding import LocusGSInstanceGroundedUnified,rendered_instance_grounding_loss
from tokengs.models.gaussian_instance_consistency import (
    RECIPE,LocusGSGaussianInstanceConsistency,contribution_instance_targets,
    local_different_instance_neighbors,direct_gaussian_consistency_loss)
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets
from tokengs.models.superpoint_structural_variants import build_structural_model


def model():
    return build_structural_model('unified_gaussian_consistency',opt().evolve(img_size=(8,8)),
        config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())


def assigned(labels):
    labels=torch.tensor([labels],dtype=torch.long)
    return dict(labels=labels,purity=torch.ones_like(labels,dtype=torch.float),mass=torch.ones_like(labels,dtype=torch.float))


def loss_for(features,parent,labels,xyz=None,recipe=RECIPE,assignment=None):
    n=len(parent)
    if xyz is None:xyz=torch.arange(n).float()[None,:,None].expand(1,n,3)*.001
    return direct_gaussian_consistency_loss(features,xyz,torch.tensor(parent),assigned(labels) if assignment is None else assignment,torch.tensor(1.),recipe)


class DirectLossTests(unittest.TestCase):
    def test_cancelling_other_parent_prototypes_are_skipped(self):
        f=torch.zeros(1,3,8);f[0,0,0]=1;f[0,1,1]=1;f[0,2,1]=-1;f.requires_grad_()
        loss,m=loss_for(f,[0,1,2],[0]*3)
        self.assertEqual(m['direct_positive_parent_count'],2)
        self.assertEqual(m['direct_positive_degenerate_parent_count'],1)
        loss.backward();self.assertTrue(torch.isfinite(f.grad).all())

    def test_ideal_cross_parent_identities_have_zero_loss_and_local_collapse_is_penalized(self):
        labels=[0,0,0,0,1,1,1,1];parent=[0,0,1,1,2,2,3,3]
        f=torch.zeros(1,8,8);f[0,:4,0]=1;f[0,4:,1]=1
        loss,m=loss_for(f,parent,labels)
        self.assertEqual(float(loss),0.);self.assertEqual(m['direct_positive_gt_count'],2)
        self.assertEqual(m['direct_positive_parent_count'],4);self.assertGreater(m['direct_negative_edges'],0)
        f[0,:,1]=0;f[0,:,0]=1
        loss,m=loss_for(f,parent,labels)
        self.assertAlmostEqual(float(loss),.5);self.assertAlmostEqual(float(m['loss_gaussian_direct_push']),.5)

    def test_positive_prototype_excludes_own_parent_and_has_expected_hinge(self):
        f=torch.zeros(1,4,8);f[0,:2,0]=1;f[0,2:,1]=1
        loss,m=loss_for(f,[0,0,1,1],[0]*4)
        torch.testing.assert_close(loss,torch.tensor(.9));self.assertEqual(m['direct_negative_edges'],0)
        # One parent cannot supply its own positive, regardless of child count.
        loss,m=loss_for(f,[0]*4,[0]*4)
        self.assertEqual(float(loss),0.);self.assertEqual(m['direct_positive_gt_count'],0)

    def test_balance_is_invariant_to_duplicated_children_and_instance_id_renaming(self):
        torch.manual_seed(21);f=torch.randn(1,8,8)
        parent=[0,0,1,1,2,2,3,3];labels=[0]*4+[1]*4
        no_neighbors=replace(RECIPE,negative_radius_scene_fraction=0.)
        a,_=loss_for(f,parent,labels,recipe=no_neighbors)
        index=[0,1,0,1,0,1,2,3,4,5,6,7]
        b,_=loss_for(f[:,index],[parent[i] for i in index],[labels[i] for i in index],recipe=no_neighbors)
        torch.testing.assert_close(a,b)
        c,_=loss_for(f,parent,[50 if x==0 else 900 for x in labels],recipe=no_neighbors)
        torch.testing.assert_close(a,c)

    def test_gt_balance_is_equal_even_when_parent_counts_differ(self):
        f=torch.zeros(1,4,8);f[0,:2,0]=1;f[0,2:,1]=1
        a,_=loss_for(f,[0,0,1,1],[0]*4)
        # GT1 has three identical parents and zero pull. It receives equal GT weight.
        extra=torch.zeros(1,9,8);extra[:,:,2]=1
        b,_=loss_for(torch.cat((f,extra),1),[0,0,1,1]+[2]*3+[3]*3+[4]*3,[0]*4+[1]*9)
        torch.testing.assert_close(b,a/2)

    def test_empty_mixed_low_mass_and_single_parent_are_finite_graph_connected(self):
        for case in ('empty','purity','mass','single_parent'):
            f=torch.randn(1,4,8,requires_grad=True);a=assigned([0]*4);parents=[0,0,1,1]
            if case=='empty':a['labels'].fill_(-1)
            if case=='purity':a['purity'].fill_(.69)
            if case=='mass':a['mass'].fill_(.0009)
            if case=='single_parent':parents=[0]*4
            loss,m=loss_for(f,parents,[0]*4,assignment=a);loss.backward()
            self.assertEqual(float(loss),0.);self.assertTrue(torch.isfinite(f.grad).all())
            self.assertEqual(float(f.grad.norm()),0.)

    def test_nearest_other_gt_ignores_dense_same_gt_neighbors_and_respects_scale_radius(self):
        xyz=torch.tensor([[0.,0,0],[.001,0,0],[.002,0,0],[.02,0,0],[2.,0,0]],requires_grad=True)
        labels=torch.tensor([0,0,0,1,2])
        recipe=replace(RECIPE,negative_neighbors=1)
        indices,within=local_different_instance_neighbors(xyz,labels,torch.tensor(1.),recipe)
        self.assertEqual(int(indices[0,0]),3);self.assertTrue(within[0,0]);self.assertFalse(within[4,0])
        a,b=local_different_instance_neighbors(xyz*3,labels,torch.tensor(3.),recipe)
        self.assertTrue(torch.equal(a,indices));self.assertTrue(torch.equal(b,within))
        self.assertFalse(indices.requires_grad);self.assertFalse(within.requires_grad)

    def test_loss_does_not_differentiate_geometry_purity_or_mass(self):
        torch.manual_seed(42);f=torch.randn(1,8,8,requires_grad=True)
        xyz=(torch.randn(1,8,3)*.001).requires_grad_();a=assigned([0]*4+[1]*4)
        a['purity'].requires_grad_();a['mass'].requires_grad_();scale=torch.tensor(1.,requires_grad=True)
        loss,_=direct_gaussian_consistency_loss(f,xyz,torch.tensor([0,0,1,1,2,2,3,3]),a,scale)
        loss.backward();self.assertGreater(float(f.grad.norm()),0.)
        for value in (xyz,a['purity'],a['mass'],scale):self.assertIsNone(value.grad)

    def test_ownership_uses_valid_stuff_denominator_rejects_ambiguous_and_low_mass(self):
        class Renderer(nn.Module):
            def render_feature_channels(self,g,f,camera,intrinsics):
                matrix=torch.tensor([[.9,0.,.5,.0001],[.1,.9,.5,0.]])
                image=torch.einsum('pn,bnc->bcp',matrix,f).reshape(1,1,f.shape[-1],1,2).expand(1,2,-1,-1,-1)
                return dict(images_pred=image)
        batch=dict(semantic_label_all=torch.tensor([[[[2,0]],[[2,0]]]]),instance_label_all=torch.tensor([[[[1,0]],[[1,0]]]]))
        decoder=ModelInputDecoder(cam_view=torch.eye(4).expand(1,2,4,4),intrinsics=torch.ones(1,2,4))
        g=torch.zeros(1,4,14,requires_grad=True)
        result=contribution_instance_targets(g,decoder,Renderer(),build_context_instance_targets(batch))
        self.assertEqual(result['labels'].tolist(),[[0,-1,-1,-1]])
        torch.testing.assert_close(result['purity'],torch.tensor([[.9,0.,.5,1.]]))
        self.assertFalse(any(value.requires_grad for value in result.values()))
        batch['semantic_label_all'].fill_(0)
        with patch.object(Renderer,'render_feature_channels',side_effect=AssertionError('No GT: do not rasterize')):
            empty=contribution_instance_targets(g,decoder,Renderer(),build_context_instance_targets(batch))
            self.assertTrue((empty['labels']==-1).all())


class DirectModelTests(unittest.TestCase):
    def test_same_parameters_rng_and_native_outputs_without_gt(self):
        torch.manual_seed(42);baseline=LocusGSInstanceGroundedUnified(opt().evolve(img_size=(8,8)),config=config(),omega=FakeOmega(),renderer=SpatialFeatureRenderer())
        rng=torch.get_rng_state();torch.manual_seed(42);new=model()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(baseline.state_dict().keys(),new.state_dict().keys())
        for name,value in baseline.state_dict().items():torch.testing.assert_close(new.state_dict()[name],value,rtol=0,atol=0)
        rgb=torch.rand(1,2,3,256,256);a=baseline.generate(rgb);b=new.generate(rgb)
        self.assertEqual(a.keys(),b.keys())
        for key in ('gaussians','gaussian_mask_logits','thing_logits19','region_mass','semantic_scores','instance_eligible','gaussian_instance_features'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        torch.testing.assert_close(new.core.direct_instance_features(b),b['gaussian_instance_features'],rtol=0,atol=0)

    def test_new_loss_reaches_understanding_shared_but_not_any_explicit_geometry_head(self):
        torch.manual_seed(42);new=model();rgb=torch.rand(1,2,3,256,256)
        _,cloud,_=new.encode_context(rgb);p=new.generate(rgb,superpoint_ids=torch.arange(len(cloud['points']))%4)
        features=new.core.direct_instance_features(p);parent=p['gaussian_parent_query']
        a=assigned((parent%2).tolist())
        p['gaussians'].retain_grad();p['states'][-1]['mu'].retain_grad();p['states'][-1]['radii'].retain_grad()
        direct,_=direct_gaussian_consistency_loss(features,p['gaussians'][...,:3],parent,a,p['cloud']['scale']*100)
        direct.backward()
        for module in (new.core.instance_head,new.core.child_feature,new.core.semantic_adapter,new.core.blocks,new.gfm_projection):
            self.assertGreater(sum(float(x.grad.abs().sum()) for x in module.parameters() if x.grad is not None),0.)
        for module in (new.core.geometry,new.core.activation_head,new.core.reconstruction_adapter,new.frozen_omega,new.core.classifier,new.core.query_mask):
            self.assertTrue(all(x.grad is None for x in module.parameters()))
        self.assertIsNone(p['gaussians'].grad);self.assertIsNone(p['states'][-1]['mu'].grad);self.assertIsNone(p['states'][-1]['radii'].grad)

    def test_old_rendered_loss_still_updates_gaussian_geometry(self):
        new=model();p=new.generate(torch.rand(1,2,3,256,256))
        batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long));batch['instance_label_all'][...,-4:]=2
        loss,_=rendered_instance_grounding_loss(p['rendered_instance_features'],build_context_instance_targets(batch))
        p['gaussians'].retain_grad();loss.backward()
        self.assertGreater(float(p['gaussians'].grad[...,:3].norm()),0.)
        self.assertTrue(any(x.grad is not None for x in new.core.activation_head.parameters()))

    def test_step_loss_retains_original_terms_context_only_and_effective_weight(self):
        new=model();rgb=torch.rand(1,2,3,256,256)
        batch=dict(images_input=rgb,images_all=rgb,semantic_label_all=torch.full((1,3,8,8),2),instance_label_all=torch.ones(1,3,8,8,dtype=torch.long),
                   has_mask=False,masks_all=torch.ones(1,2,1,256,256))
        batch['instance_label_all'][...,-4:]=2
        def calibration(images,p):return dict(c2w=torch.eye(4).expand(1,2,4,4),intrinsics=torch.ones(1,2,4),intrinsics_matrix=torch.eye(3).expand(1,2,3,3))
        def objective(states,decoder,supervision):
            loss=new.activation_head(states[-1]['tokens'],states[-1]['mu'],states[-1]['radii']).square().mean()
            return loss,{f'psnr_layer{new.supervised_layers[-1]}':loss.detach()*0},None,None,None
        with patch.object(new,'calibrate_targets',side_effect=calibration),patch.object(new,'_layer_objective',side_effect=objective):
            _,m=new.step_loss(batch,understanding_weight=.05)
            torch.testing.assert_close(m['loss'],m['loss_recon']+.05*m['loss_understanding_base']+.0005*m['loss_instance_grounding']+.0005*m['loss_gaussian_direct'])
            self.assertEqual(m['gaussian_direct_effective_weight'],.0005)
            self.assertEqual(m['instance_feature_supervised_pixels'],128)
            m['loss'].backward()

    def test_checkpoint_recipe_identity_and_parameter_groups(self):
        from scripts.train_unified_query_v1 import trainable_state,restore_state
        from scripts.unified_query_initial_state import validate_zero_initial
        new=model();state=trainable_state(new);restore_state(new,state)
        groups=new.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in new.parameters() if p.requires_grad})
        meta=dict(architecture=new.architecture_name,config=new.config.to_dict(),manifest_sha256='train',heldout_sha256='heldout',world_size=1,
            omega_identity=new.frozen_omega.source_identity,sam_identity=None,**new.extra_checkpoint_metadata)
        saved=dict(meta=meta,progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer=dict(state={}),rank_rng=[{}])
        def check(value):validate_zero_initial(value,new,manifest_sha256='train',heldout_sha256='heldout',world=1)
        check(saved)
        for key,value in (('architecture',LocusGSInstanceGroundedUnified.architecture_name),('gaussian_consistency_recipe',{})):
            bad=copy.deepcopy(saved);bad['meta'][key]=value
            with self.assertRaises(ValueError):check(bad)


if __name__=='__main__':unittest.main()
