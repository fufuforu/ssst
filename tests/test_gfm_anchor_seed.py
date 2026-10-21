import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from tokengs.models.anchor_seed_initialization import UpdatedAnchorSAMQueryInitializer
from tokengs.models.dual_stream_decoder import PointSeededAnchorDecoder
from tokengs.models.object_locus_panoptic_v1_controller import ObjectLocusPanopticV1Controller
from scripts.gfm_anchor_seed_policy import configure,validate_initial
from tests.test_gfm_v4_isolated_policy import IsolatedPolicyTests


class Scale(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(.01))
    def forward(self,x,*args,**kwargs):return x*self.weight


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.calls=0
        self.gs_cross_attn_scale=nn.Identity();self.gs_self_attn_scale=nn.Identity();self.mlp_scale=nn.Identity()
        self.gs_self_attn=Scale();self.mlp=Scale()
    def gs_cross_attn(self,x,k,v,attn_bias=None):self.calls+=1;return x*.001


def fixture():
    decoder=PointSeededAnchorDecoder.__new__(PointSeededAnchorDecoder);nn.Module.__init__(decoder)
    decoder.opt=SimpleNamespace(locusgs_ray_bias_scale=0.,locusgs_pe_num_freqs=0)
    decoder.epsilon=.001;decoder.pe_mode='persistent';decoder.pe_mlp=nn.Linear(3,1024)
    decoder.decoder_blocks=nn.ModuleList([Block() for _ in range(12)])
    decoder.refine_mu=nn.ModuleList([nn.Linear(1024,3) for _ in range(12)])
    decoder.refine_rho=nn.ModuleList([nn.Linear(1024,1) for _ in range(12)])
    for head in list(decoder.refine_mu)+list(decoder.refine_rho):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    decoder.eval();controller=ObjectLocusPanopticV1Controller();mask=nn.Linear(256,256)
    tokens=torch.randn(1,1024,1024,requires_grad=True);mu=torch.randn(1,1024,3,requires_grad=True);radius=torch.ones(1,1024)*.2
    latent=SimpleNamespace(keys=torch.randn(1,1,3,4),values=torch.randn(1,1,3,4));rays=(torch.randn(1,3,3),torch.randn(1,3,3))
    fm=torch.randn(1,2,256,2,2);q=torch.randn(1,100,256,requires_grad=True);c=torch.zeros(1,100,3);s=torch.ones(1,100,3)
    return decoder,controller,mask,tokens,mu,radius,latent,rays,fm,q,c,s


class AnchorSeedTests(unittest.TestCase):
    def test_two_stages_preserve_reconstruction_and_replay_without_second_reconstruction(self):
        torch.manual_seed(5);decoder,controller,mask,tokens,mu,radius,latent,rays,fm,q,c,s=fixture()
        kw=dict(seed_mu=mu,seed_radius=radius,seed_c=c,seed_s=s,write_strength=0.)
        native,_=decoder.forward_stateful(tokens,latent,rays,controller,fm,q,mask,0,**kw)
        reconstruction,_=decoder.forward_stateful(tokens,latent,rays,None,None,None,None,0,reconstruction_only=True,**kw)
        calls=[b.calls for b in decoder.decoder_blocks]
        replay,_=decoder.forward_stateful(tokens.detach(),latent,rays,controller,fm,q,mask,0,reconstruction_states=reconstruction,**kw)
        self.assertEqual(calls,[b.calls for b in decoder.decoder_blocks])
        for a,b,curr in zip(native,reconstruction,replay):
            for key in ('tokens','mu','rho','radii'):
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0);torch.testing.assert_close(b[key],curr[key],rtol=0,atol=0)
            if a['layer'] in (6,8,10,12):torch.testing.assert_close(a['q'],curr['q'],rtol=0,atol=0)
        grads=torch.autograd.grad(replay[-1]['anchor_membership'].sum(),(tokens,mu,controller.proj_h.weight),allow_unused=True)
        self.assertIsNone(grads[0]);self.assertIsNone(grads[1]);self.assertGreater(float(grads[2].norm()),0.)
        with self.assertRaises(ValueError):decoder.forward_stateful(tokens,latent,rays,None,None,None,None,0,**dict(kw,write_strength=.1),reconstruction_only=True)

    def test_initializer_uses_anchor_content_with_gradients_but_detached_positions(self):
        torch.manual_seed(6);init=UpdatedAnchorSAMQueryInitializer()
        points=torch.zeros(1,2,4,4,3);mu=torch.tensor([[[-.1,0,0],[.1,0,0],[0,.1,0],[0,0,.1]]],requires_grad=True)
        # Put support exactly at Anchor zero, and retain other Anchors for scene scale.
        points[:]=mu.detach()[0,0]
        fm=torch.randn(1,2,256,2,2,requires_grad=True);features=torch.randn(1,4,256,requires_grad=True)
        props=[[[dict(score=1.,mask=torch.ones(4,4,dtype=torch.bool))],[]]]
        context=dict(features=features,mu=mu,records=[[]])
        q,c,s,logs=init(fm,props,points,mu.detach(),seed_context=context)
        self.assertEqual(logs[0]['proposal_queries'],1);self.assertEqual(logs[0]['unsupported_anchor_seed_regions'],0)
        loss=q[0,0,0];grads=torch.autograd.grad(loss,(features,mu,fm,init.content_projection[1].weight,init.point_weight[2].weight,init.fallback,init.residual_gate_logit),allow_unused=True)
        self.assertIsNone(grads[1]);self.assertIsNone(grads[2])
        for i in (0,3,4,5,6):self.assertGreater(float(grads[i].norm()),0.)
        shifted=dict(features=features.detach()+torch.randn_like(features),mu=mu,records=[[]])
        q2,c2,s2,logs2=init(fm,props,points,mu.detach(),seed_context=shifted)
        self.assertFalse(torch.equal(q[:,:1],q2[:,:1]));torch.testing.assert_close(q[:,1:],q2[:,1:],rtol=0,atol=0)
        for a,b in ((c,c2),(s,s2)):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(logs2[0]['proposal_queries'],logs[0]['proposal_queries'])
        unsupported=dict(features=features,mu=mu.detach()+100,records=[[]])
        q3,_,_,log3=init(fm,props,points,mu.detach(),seed_context=unsupported)
        self.assertEqual(log3[0]['unsupported_anchor_seed_regions'],1)
        torch.testing.assert_close(q3,init.norm(init.fallback)[None],rtol=0,atol=0)
        with self.assertRaises(ValueError):init(fm,props,points,mu.detach())

    def test_model_orchestration_runs_reconstruction_first_and_keeps_reconstruction_gradient(self):
        from tokengs.models.object_locus_gfm_dual_stream_v4 import LocusGSGFMDualStreamV4
        torch.manual_seed(7);decoder,controller,mask,tokens,mu,radius,latent,rays,fm,q,c,s=fixture()
        initializer=UpdatedAnchorSAMQueryInitializer()
        model=SimpleNamespace(anchor_query_seed=True,anchor_decoder=decoder,panoptic=controller,understanding_step=0,
            understanding=SimpleNamespace(mask_embedder=mask),query_initializer=initializer,query_seed_memory=lambda raw,f: f)
        points=mu.detach()[:,0,None,None,None,:].expand(1,2,4,4,3).clone()
        props=[[[dict(score=1.,mask=torch.ones(4,4,dtype=torch.bool))],[]]]
        observations=[]
        hook=initializer.register_forward_pre_hook(lambda *args: observations.append(sum(b.calls for b in decoder.decoder_blocks)))
        result=LocusGSGFMDualStreamV4.decode_scene(model,tokens,latent,rays,fm,props,dict(points=points),mu,radius,None)
        hook.remove();states,_,qpre,_,_,logs,recfinal=result
        self.assertEqual(observations,[12]);self.assertEqual(sum(b.calls for b in decoder.decoder_blocks),12)
        self.assertEqual(logs[0]['unsupported_anchor_seed_regions'],0)
        gradients=torch.autograd.grad(qpre[0,0,0],(tokens,mu,controller.proj_h.weight),allow_unused=True,retain_graph=True)
        self.assertIsNone(gradients[0]);self.assertIsNone(gradients[1]);self.assertGreater(float(gradients[2].norm()),0.)
        # Production reconstruction loss uses the returned per-layer states.
        reconstruction_loss=sum(state['tokens'].square().mean()+state['mu'].square().mean()+state['rho'].square().mean() for state in states)
        gradients=torch.autograd.grad(reconstruction_loss,(tokens,mu,controller.proj_h.weight),allow_unused=True)
        self.assertGreater(float(gradients[0].norm()),0.);self.assertGreater(float(gradients[1].norm()),0.);self.assertIsNone(gradients[2])
        torch.testing.assert_close(states[-1]['tokens'],recfinal['tokens'],rtol=0,atol=0)

    def test_recipe_locks_only_allowed_structure_settings(self):
        saved,_,blob=IsolatedPolicyTests().fixture();saved['config']['required_gradient_families']=['geometric_query_pooling','semantic_query_projection'];target=copy.deepcopy(saved)
        configure(target['config']);target['recipe']=target['config']['recipe'];validate_initial(saved,target,blob)
        for key,value in [('anchor_seed_neighbors',8),('understanding_lr',.01),('decoder_execution_order','interleaved')]:
            bad=copy.deepcopy(target);bad['config'][key]=value
            with self.assertRaises(ValueError):validate_initial(saved,bad,blob)


if __name__=='__main__':unittest.main()
