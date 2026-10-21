"""CPU contracts for the registered interleaved coupling."""
import inspect
import unittest
import torch
from torch import nn
from tokengs.models.object_locus_joint_v1 import (
    joint_beta, joint_route, joint_residual, LocusGSObjectLocusJointV1Recon,
    ObjectLocusJointAnchorDecoder)
from tokengs.models.locusgs_recon import LocusGSAnchorDecoder
from scripts.object_locus_joint_v1_runtime import (
    build_model, build_optimizer, backward_gradient_controlled,
    lr_multiplier, understanding_weight, build_manifest, build_plan)


class JointContracts(unittest.TestCase):
    def test_route_shapes_normalization_and_finite_guard(self):
        torch.manual_seed(42)
        a=torch.randn(1,1024,256);q=torch.randn(1,102,256)
        mu=torch.randn(1,1024,3);c=torch.randn(1,100,3);s=torch.ones(1,100,3)
        t,u,v=joint_route(a,q,mu,c,s)
        self.assertEqual(t.shape,(1,1024,102));self.assertEqual(u.shape,(1,1024,256))
        torch.testing.assert_close(t.sum(-1),torch.ones(1,1024))
        self.assertEqual(v.shape,(1,102,256))
        c[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):joint_route(a,q,mu,c,s)

    def test_residual_bound_beta_and_schedule(self):
        h=torch.randn(1,1024,1024);raw=torch.randn_like(h)*100
        for step in (-1,0,1,40,200,3584):
            beta=joint_beta(step);d=joint_residual(h,raw,beta)
            self.assertTrue((d.norm(dim=-1)<=.1*beta*h.norm(dim=-1)+1e-5).all())
        self.assertEqual(joint_beta(0),0);self.assertEqual(joint_beta(200),1)
        self.assertEqual(lr_multiplier(0),0);self.assertEqual(lr_multiplier(200),1)
        self.assertAlmostEqual(lr_multiplier(3584),.1)
        self.assertEqual(understanding_weight(40),.2)
        self.assertTrue(torch.equal(joint_residual(torch.zeros_like(h),raw,1),torch.zeros_like(h)))

    def test_transfer_init_optimizer_and_trainability(self):
        from scripts.object_locus_v3_set_runtime import build_model as baseline
        model,opt,transfer=build_model('cpu',arm='control')
        self.assertEqual((transfer['matched_reconstruction_tensor_count'],
            transfer['new_object_locus_tensor_count'],transfer['new_injection_tensor_count']),(450,78,4))
        reference,_,_=baseline('cpu')
        for k,v in reference.state_dict().items():self.assertTrue(torch.equal(v,model.state_dict()[k]),k)
        joint,_,_=build_model('cpu',arm='joint')
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,joint.state_dict()[k]),k)
        self.assertEqual(sum(p.numel() for p in model.object_locus_joint_injection.parameters()),1048576)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(torch.count_nonzero(p)==0 for p in model.object_locus_joint_injection.parameters()))
        optimizer,audit=build_optimizer(model);self.assertTrue(audit['all_trainable_once'])
        for p in model.object_locus_joint_injection.parameters():
            group=next(g for g in optimizer.param_groups if any(x is p for x in g['params']))
            self.assertEqual(group['lr'],1e-4);self.assertEqual(group['weight_decay'],.05)
        self.assertIsNone(model.anchor_decoder.token_update_hook)
        self.assertEqual(len(model.state_dict()),532)

    def test_gc_excludes_both_object_branches(self):
        model=nn.Module();model.reconstruction=nn.Linear(1,1,bias=False)
        model.object_locus_v3_set=nn.Linear(1,1,bias=False)
        model.object_locus_joint_injection=nn.Linear(1,1,bias=False)
        loss=sum(p.sum() for p in model.parameters())
        audit=backward_gradient_controlled(model,loss,loss,1)
        self.assertEqual(audit['registered_hook_count'],1);self.assertEqual(audit['removed_hook_count'],1)
        self.assertAlmostEqual(float(model.reconstruction.weight.grad),1.01,places=6)
        self.assertEqual(float(model.object_locus_v3_set.weight.grad),2)
        self.assertEqual(float(model.object_locus_joint_injection.weight.grad),2)

    def test_forward_order_loss_and_plan(self):
        source=inspect.getsource(ObjectLocusJointAnchorDecoder.forward_stateful)
        original=inspect.getsource(LocusGSAnchorDecoder.forward)
        # Exact attention and MLP operations, including their scale wrappers.
        start='            tokens = tokens + block.gs_cross_attn_scale('
        stop='            previous_mu = mu'
        self.assertEqual(source[source.index(start):source.index(stop)],original[original.index(start):original.index(stop)])
        self.assertLess(source.index('mu = mu + head_mu'),source.index('scene_normalization(mu)'))
        self.assertLess(source.index('initialize_states('),source.index('forward_registered_layer('))
        self.assertLess(source.index('forward_registered_layer('),source.index('tokens = tokens + delta'))
        self.assertLess(source.index('tokens = tokens + delta'),source.index('states.append('))
        step=inspect.getsource(LocusGSObjectLocusJointV1Recon.step_loss)
        self.assertIn('object_locus_v3_set_losses(',step);self.assertIn('self._layer_objective(',step)
        forward=inspect.getsource(LocusGSObjectLocusJointV1Recon.forward_object_locus)
        self.assertLess(forward.index('decode_object_locus('),forward.index('activation_head('))
        self.assertLess(forward.index('activation_head('),forward.index('self._readout('))
        plan=build_plan(build_manifest());self.assertEqual(len(plan['entries']),3584)
        counts={i:0 for i in range(56)}
        for x in plan['entries']:counts[x['window_index']]+=1
        self.assertEqual(set(counts.values()),{64})


if __name__=='__main__':unittest.main()

class AnalyticProjectionContracts(unittest.TestCase):
    def setup_camera(self):
        return torch.eye(4).reshape(1,1,4,4),torch.tensor([[[100.,100.,128.,128.]]])

    def test_projection_visibility_and_near_plane(self):
        from tokengs.models.canonical_recon import project_points_means2d,visibility_loss_from_points
        cam,intr=self.setup_camera()
        for xyz,uv,expected in [((0,0,1),(128,128),0),((2,0,1),(328,128),.5625),
                                ((3,0,1),(428,128),1),((0,0,.01),(128,128),1)]:
            p=torch.tensor([[xyz]],dtype=torch.float32)
            torch.testing.assert_close(project_points_means2d(p,cam,intr),torch.tensor([[[uv]]],dtype=torch.float32))
            loss=visibility_loss_from_points(p,cam,intr,(256,256),clamp_max=1,znear=.025)
            self.assertAlmostEqual(float(loss),expected)
        cam=cam.repeat(1,2,1,1);cam[0,1,3,0]=3
        p=torch.tensor([[[0.,0.,1.]]])
        self.assertEqual(float(visibility_loss_from_points(p,cam,intr.repeat(1,2,1),(256,256),clamp_max=1,znear=.025)),0)
        p=torch.tensor([[[1.5,0.,1.]]],requires_grad=True)
        cam,intr=self.setup_camera()
        loss=visibility_loss_from_points(p,cam,intr,(256,256),clamp_max=1,znear=.025)
        self.assertAlmostEqual(float(loss),.171875);loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)

    def test_translated_camera_renderer_wrapper(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from tokengs.models.canonical_recon import project_points_means2d
        from tokengs.rendering.gs import GaussianRenderer
        cam,intr=self.setup_camera();cam[0,0,3,:3]=torch.tensor([.3,-.2,.4])
        xyz=torch.tensor([[[.2,.1,1.],[1.5,.4,2.]]],requires_grad=True)
        w2c=cam.transpose(-1,-2)
        camera=torch.matmul(xyz,w2c[0,0,:3,:3].T)+w2c[0,0,:3,3]
        expected=torch.stack((100*camera[...,0]/camera[...,2]+128,100*camera[...,1]/camera[...,2]+128),-1).unsqueeze(1)
        torch.testing.assert_close(project_points_means2d(xyz,cam,intr),expected)
        ks=torch.tensor([[[[100.,0,128],[0,100.,128],[0,0,1.]]]])
        def fake(**kwargs):
            return torch.zeros(1,256,256,4),torch.ones(1,256,256,1),{}
        with patch('tokengs.rendering.gs.rasterization',side_effect=fake):
            out=GaussianRenderer(SimpleNamespace()).render_standard(xyz,torch.ones(1,2),torch.ones(1,2,3),torch.ones(1,2,4),torch.ones(1,2,3),w2c,ks,torch.ones(1,1,3),256,256,.025,100)
        torch.testing.assert_close(out['means2d_pred'],expected)
        self.assertTrue(out['means2d_pred'].requires_grad)
