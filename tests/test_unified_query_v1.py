import copy
import unittest
from unittest.mock import patch
import torch
from torch import nn
from tokengs.options import config_defaults
from tokengs.models.unified_superpoints import UnifiedQueryConfig,voxel_partition,scatter_mean,SuperpointQueryInitializer,PointNeighborhoodEncoder
from tokengs.models.unified_query_core import UnifiedQueryCore
from tokengs.models.unified_query_loss import unified_instance_loss
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
from scripts.export_object_locus_v3_set_official import assemble_panoptic


def config():return UnifiedQueryConfig(dim=16,heads=4,depth=2,point_grid=4,max_points=16,max_superpoints=4,neighbors=3,point_layers=1,checkpoint_blocks=False)
def opt():return config_defaults['train_siu3r_unified_query_v1'].evolve(dec_patch_size=2)
def cloud():
    p=torch.tensor([[0.,0,0],[.01,0,0],[1.,0,0],[1.01,0,0]])
    return dict(points=p,features=torch.randn(4,16,requires_grad=True),origin=p.mean(0),scale=torch.tensor(.5),source_to_point=torch.arange(4),source_indices=torch.arange(4))


class FakeOmega(nn.Module):
    source_identity={'synthetic':True}
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(1.))
    @torch.no_grad()
    def forward(self,rgb):
        b,v=rgb.shape[:2];device=rgb.device
        y,x=torch.meshgrid(torch.arange(8,device=device),torch.arange(8,device=device),indexing='ij')
        p=torch.stack((x*.02,y*.02,.25+x*y*.001),-1).float()[None,None].expand(b,v,-1,-1,-1).clone()
        feature=torch.linspace(-1,1,2048,device=device)[None,None,None,:].expand(b,v,1024,-1)
        return dict(points=p,confidence=torch.ones(b,v,1,8,8,device=device),depth=p[...,2].unsqueeze(2),
            patch_layers=[feature]*4,c2w=torch.eye(4,device=device).expand(b,v,4,4).clone(),k=torch.eye(3,device=device).expand(b,v,3,3).clone())


class FakeRenderer(nn.Module):
    def render(self,g,cam_view,bg_color,intrinsics):
        b,v=cam_view.shape[:2];rgb=g[...,11:14].mean(1)[:,None,:,None,None].expand(b,v,3,8,8)
        depth=g[...,2].mean(1)[:,None,None,None,None].expand(b,v,1,8,8)
        alpha=g[...,3].mean(1)[:,None,None,None,None].expand(b,v,1,8,8)
        return dict(images_pred=rgb,depths_pred=depth,alphas_pred=alpha)
    def render_feature_channels(self,g,features,cam_view,intrinsics):
        b,v=cam_view.shape[:2];alpha=g[...,3].mean(1)[:,None,None,None,None].expand(b,v,1,8,8)
        image=features.mean(1)[:,None,:,None,None].expand(b,v,features.shape[-1],8,8)*alpha
        return dict(images_pred=image,alphas_pred=alpha)


class UnifiedTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(3)

    def test_spatial_partition_full_coverage_and_fusion_gradients(self):
        p=cloud()['points'];ids,size=voxel_partition(p,.5,.02,2)
        self.assertEqual(len(ids),4);self.assertLessEqual(int(ids.max())+1,2);self.assertGreater(size,0)
        f=torch.randn(4,3,requires_grad=True);pooled=scatter_mean(f,ids,int(ids.max())+1)
        pooled.sum().backward();self.assertTrue((f.grad>0).all())
        with self.assertRaises(ValueError):voxel_partition(p*float('nan'),.5,.02,2)

    def test_3d_neighbors_change_point_features(self):
        module=PointNeighborhoodEncoder(16,1,2);c=cloud();a=module(c['features'],c['points'],c['scale'])
        p=c['points'].clone();p[1]=torch.tensor([5.,5,5]);b=module(c['features'],p,c['scale'])
        self.assertFalse(torch.allclose(a[0],b[0]))

    def test_superpoint_pooling_external_ids_no_fallback_and_gradient(self):
        module=SuperpointQueryInitializer(config());c=cloud();ids=torch.tensor([9,9,30,30])
        out=module(c,ids);self.assertEqual(out['query'].shape,(1,2,16));self.assertEqual(out['superpoint_counts'].tolist(),[2,2])
        self.assertFalse(any('fallback' in n for n,_ in module.named_parameters()))
        out['query'][0,0,0].backward();self.assertGreater(float(c['features'].grad.norm()),0.)
        self.assertGreater(float(module.point_encoder.layers[0][0].weight.grad.norm()),0.)
        self.assertGreater(float(module.point_weight[2].weight.grad.norm()),0.)
        with self.assertRaises(ValueError):module(c,torch.tensor([-1,1,2,3]))
        with self.assertRaises(ValueError):module(c,torch.ones(4))

    def test_one_sequence_two_heads_and_shared_task_gradients(self):
        core=UnifiedQueryCore(opt(),config());c=cloud();out=core(c,torch.randn(1,5,16),torch.tensor([0,0,1,1]))
        self.assertEqual(out['gaussians'].shape,(1,8,14));self.assertEqual(out['p_class'].shape,(1,2,19));self.assertEqual(out['gaussian_membership'].shape,(1,8,4))
        self.assertEqual(out['gaussian_parent_query'].tolist(),[0]*4+[1]*4)
        self.assertTrue((out['gaussian_membership']>0).all()) # no parent-instance hard assignment
        rec=out['gaussians'].square().mean();under=out['thing_logits19'].square().mean()+out['gaussian_membership'].mean()
        p=core.blocks[0].self_attn.in_proj_weight
        for loss in (rec,under):
            gradient=torch.autograd.grad(loss,p,retain_graph=True)[0];self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0.)
        groups=core.parameter_groups();ids=[id(p) for group in groups.values() for p in group]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in core.parameters()})

    def test_source_constrained_matching_and_empty_scenes(self):
        n=2;regions=torch.rand(1,2,n+2,2,2,requires_grad=True)
        pred=dict(thing_logits19=torch.randn(1,n,19,requires_grad=True),region_mass=regions,num_thing_queries=n,
            cloud=dict(source_indices=torch.arange(8)),superpoints=dict(source_superpoint_ids=torch.tensor([0,0,1,1,0,0,1,1])))
        batch=dict(semantic_label_all=torch.tensor([[[[2,2],[0,0]],[[2,2],[0,0]]]]),instance_label_all=torch.tensor([[[[1,1],[0,0]],[[1,1],[0,0]]]]))
        loss,metrics=unified_instance_loss(pred,batch,config());self.assertEqual(metrics['matched_gt_count'],1);self.assertEqual(pred['final_pairs'][0][0].tolist(),[0]);loss.backward()
        self.assertGreater(float(pred['thing_logits19'].grad.norm()),0.)
        # GT still exists, but no sampled support belongs to it: no fabricated match.
        pred['cloud']['source_indices']=torch.tensor([2,3,6,7]);pred['superpoints']['source_superpoint_ids']=torch.tensor([0,0,1,1])
        _,metrics=unified_instance_loss(pred,batch,config());self.assertEqual(metrics['matched_gt_count'],0);self.assertEqual(metrics['unsupported_GT_count'],1)
        batch['semantic_label_all'].zero_();batch['instance_label_all'].zero_()
        loss,metrics=unified_instance_loss(pred,batch,config());self.assertEqual(metrics['gt_count'],0);self.assertTrue(torch.isfinite(loss))

    def test_variable_query_panoptic_and_100_query_compatibility(self):
        for n in (3,100):
            p=torch.zeros(1,n,19);p[:,:,18]=1;p[0,0]=0;p[0,0,2]=1
            mass=torch.zeros(1,2,n+2,2,2);mass[:,:,0,0]=1;mass[:,:,n+1,1]=1
            out=dict(p_class=p,region_mass=mass,alpha=torch.ones(1,2,1,2,2),semantic_scores=torch.zeros(1,2,20,2,2))
            sem,ins,_=assemble_panoptic(out)
            self.assertTrue((sem[:,0]==4).all());self.assertTrue((ins[:,0]==1).all());self.assertTrue((sem[:,1]==1).all());self.assertTrue((ins[:,1]==0).all())

    def test_model_context_only_forward_checkpoint_and_render(self):
        model=LocusGSUnifiedQueryV1(opt(),config=config(),omega=FakeOmega(),renderer=FakeRenderer());model.train()
        self.assertFalse(model.frozen_omega.training);self.assertFalse(model.frozen_omega.weight.requires_grad)
        rgb=torch.rand(1,2,3,256,256);out=model.generate(rgb)
        n=out['num_thing_queries'];self.assertLessEqual(n,4);self.assertEqual(out['region_mass'].shape,(1,2,n+2,8,8))
        generated=model.render_generated_at(out,torch.eye(4).expand(1,3,4,4),torch.ones(1,3,4));self.assertEqual(generated['region_mass'].shape[1],3)
        replacement=LocusGSUnifiedQueryV1(opt(),config=config(),omega=FakeOmega(),renderer=FakeRenderer());replacement.load_state_dict(model.state_dict())
        other=replacement.generate(rgb);torch.testing.assert_close(out['gaussians'],other['gaussians'],rtol=0,atol=0)
        with self.assertRaises(ValueError):model.generate(rgb.expand(2,-1,-1,-1,-1))
        with self.assertRaises(ValueError):model.generate(rgb*float('nan'))

    def test_checkpoint_whitelist_roundtrip_and_rejects_old_architecture(self):
        from scripts.train_unified_query_v1 import trainable_state,restore_state
        model=LocusGSUnifiedQueryV1(opt(),config=config(),omega=FakeOmega(),renderer=FakeRenderer())
        state=trainable_state(model);self.assertFalse(any(k.startswith('frozen_omega.') for k in state))
        before=model.core.classifier.weight.detach().clone()
        with torch.no_grad():model.core.classifier.weight.add_(2.)
        restore_state(model,state);torch.testing.assert_close(model.core.classifier.weight,before,rtol=0,atol=0)
        with self.assertRaises(ValueError):restore_state(model,{'query_initializer.fallback':torch.randn(100,256)})

    def test_training_loss_orchestration_keeps_both_shared_paths(self):
        model=LocusGSUnifiedQueryV1(opt(),config=config(),omega=FakeOmega(),renderer=FakeRenderer())
        rgb=torch.rand(1,2,3,256,256)
        batch=dict(images_input=rgb,images_all=torch.rand(1,3,3,256,256),semantic_label_all=torch.full((1,3,8,8),2),instance_label_all=torch.ones(1,3,8,8,dtype=torch.long),has_mask=False,masks_all=torch.ones(1,3,1,256,256))
        def calibration(images,generated):
            self.assertTrue(torch.equal(generated['generation_context_rgb'],rgb))
            return dict(c2w=torch.eye(4).expand(1,3,4,4),intrinsics=torch.ones(1,3,4),intrinsics_matrix=torch.eye(3).expand(1,3,3,3))
        def objective(states,decoder,supervision):
            loss=sum(model.activation_head(s['tokens'],s['mu'],s['radii']).square().mean() for s in states)
            return loss,{f'psnr_layer{model.supervised_layers[-1]}':loss.detach()*0},None,None,None
        with patch.object(model,'calibrate_targets',side_effect=calibration),patch.object(model,'_layer_objective',side_effect=objective):
            output,metrics=model.step_loss(batch,understanding_weight=.1)
            for key in ('loss_recon','loss_understanding'):
                gradient=torch.autograd.grad(metrics[key],model.gfm_projection[1].weight,retain_graph=True)[0]
                self.assertGreater(float(gradient.norm()),0.);self.assertTrue(torch.isfinite(gradient).all())
            metrics['loss'].backward();self.assertIsNone(model.frozen_omega.weight.grad)
            self.assertIn('final_pairs',output['prediction'])


if __name__=='__main__':unittest.main()
