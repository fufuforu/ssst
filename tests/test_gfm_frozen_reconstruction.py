import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from scripts.dual_stream_training_policy import freeze_reconstruction,enforce_frozen_reconstruction_eval,reconstruction_state_digest,configure_frozen_reconstruction,validate_joint_warmup_source


class FrozenReconstructionTests(unittest.TestCase):
    def test_frozen_weights_moments_buffers_modes_and_output_are_preserved(self):
        model=nn.Module();model.write_strength=0.
        model.reconstruction_adapter=nn.Sequential(nn.Linear(3,3),nn.Dropout(.8))
        model.reconstruction_adapter.register_buffer('offset',torch.zeros(1))
        model.understanding=nn.Linear(3,1);model.query_initializer=nn.Linear(3,1);model.panoptic=nn.Linear(3,1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        sum(p.sum() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
        moments={p:copy.deepcopy(optimizer.state[p]) for p in model.reconstruction_adapter.parameters()}
        freeze_reconstruction(model);initial=reconstruction_state_digest(model)
        model.train();enforce_frozen_reconstruction_eval(model)
        self.assertFalse(model.reconstruction_adapter.training);self.assertTrue(model.understanding.training)
        x=torch.ones(2,3);before=model.reconstruction_adapter(x).detach().clone()
        loss=model.understanding(model.reconstruction_adapter(x)).square().mean();loss.backward();optimizer.step()
        self.assertEqual(reconstruction_state_digest(model),initial)
        self.assertTrue(torch.equal(before,model.reconstruction_adapter(x)))
        for p,state in moments.items():
            self.assertIsNone(p.grad)
            for k,v in state.items():self.assertTrue(torch.equal(optimizer.state[p][k],v))
        with torch.no_grad():model.reconstruction_adapter.offset.add_(1)
        self.assertNotEqual(reconstruction_state_digest(model),initial)

    def test_freeze_rejects_forward_write_and_accepts_same_adaptation_checkpoint(self):
        model=nn.Module();model.write_strength=.1
        with self.assertRaises(ValueError):freeze_reconstruction(model)
        from tests.test_gfm_joint_warmup import JointWarmupTests
        saved,target,progress=JointWarmupTests().fixture()
        configure_frozen_reconstruction(target['config']);target['recipe']=target['config']['recipe']
        validate_joint_warmup_source(saved,target,progress)
        self.assertEqual(target['config']['reconstruction_lr'],0.)
        self.assertEqual(target['config']['understanding_loss_warmup_updates'],1043)

    def test_actual_decoder_keeps_reconstruction_values_and_detaches_zero_write_graph(self):
        from tokengs.models.dual_stream_decoder import PointSeededAnchorDecoder
        class Cross(nn.Linear):
            def forward(self,h,*args,**kwargs):return super().forward(h)
        class Block(nn.Module):
            def __init__(self):
                super().__init__();self.gs_cross_attn=Cross(4,4);self.gs_self_attn=Cross(4,4);self.mlp=nn.Linear(4,4)
                self.gs_cross_attn_scale=self.gs_self_attn_scale=self.mlp_scale=nn.Identity()
        class PE(nn.Module):
            def forward(self,x):return x.new_zeros((*x.shape[:-1],4))
        class Decoder(PointSeededAnchorDecoder):
            def __init__(self):
                nn.Module.__init__(self);self.opt=SimpleNamespace(locusgs_ray_bias_scale=0.,locusgs_pe_num_freqs=1)
                self.epsilon=1e-6;self.pe_mode='persistent';self.pe_mlp=PE()
                self.decoder_blocks=nn.ModuleList([Block() for _ in range(12)])
                self.refine_mu=nn.ModuleList([nn.Linear(4,3) for _ in range(12)])
                self.refine_rho=nn.ModuleList([nn.Linear(4,1) for _ in range(12)])
                self.frozen_reconstruction_control=True
                self.requires_grad_(False);self.eval()
            def activated_radius(self,r):return torch.nn.functional.softplus(r)+self.epsilon
        class Layer(nn.Module):
            def __init__(self):super().__init__();self.shift=nn.Parameter(torch.tensor(.1))
            def forward(self,h,a,mu,q,c,s,ell,image,mask_embedder,f_anchor,exposure,**kwargs):
                q=q+self.shift
                return dict(q=q,c=c,s=s,joint_delta=torch.ones_like(h)*q.mean(),anchor_membership=q.mean().sigmoid().expand(h.shape[0],h.shape[1],q.shape[1]))
        class Controller(nn.Module):
            def __init__(self):
                super().__init__();self.stuff_seed=nn.Parameter(torch.zeros(2,256));self.stuff_ln=self.ln_mask_a=self.W_mask_a=nn.Identity()
                self.layers=nn.ModuleDict({f'L{i}':Layer() for i in (6,8,10,12)})
            def encode_token(self,tokens,*args):return tokens
        decoder=Decoder();controller=Controller();tokens=torch.ones(1,1024,4)
        latent=SimpleNamespace(keys=torch.zeros(1,1,2,4),values=torch.zeros(1,1,2,4))
        rays=(torch.zeros(1,2,3),torch.ones(1,2,3));fm=torch.zeros(1,2,256,2,2);q=torch.ones(1,1,256,requires_grad=True)
        kwargs=dict(seed_mu=torch.linspace(.1,1.,1024).view(1,1024,1).expand(-1,-1,3).contiguous(),seed_radius=torch.full((1,1024),.1),seed_c=torch.zeros(1,1,3),seed_s=torch.ones(1,1,3),write_strength=0.)
        def run():return decoder.forward_stateful(tokens,latent,rays,controller,fm,q,nn.Identity(),0,**kwargs)[0][-1]
        before=run();self.assertFalse(before['tokens'].requires_grad);self.assertTrue(before['q'].requires_grad)
        before['q'].sum().backward();self.assertIsNotNone(controller.layers['L12'].shift.grad)
        with torch.no_grad():
            for layer in controller.layers.values():layer.shift.add_(2.)
        after=run();self.assertTrue(torch.equal(before['tokens'],after['tokens']));self.assertTrue(torch.equal(before['mu'],after['mu']))
        self.assertFalse(torch.equal(before['q'],after['q']))

if __name__=='__main__':unittest.main()
