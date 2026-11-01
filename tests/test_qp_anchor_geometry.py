"""Check geometry-key alignment, unit conversion, local decoding and checkpoint gradients."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(ROOT)
import unittest,copy,dataclasses
from types import SimpleNamespace
import torch
from native_scannet.core import options,build_model
from native_scannet.anchor_geometry import install,memory_rays,geometry_bias
from tokengs.options import config_defaults
from scripts.membership_geometry_sdpa import install as install_sdpa
install_sdpa();torch.set_num_threads(2)

class GeometryTests(unittest.TestCase):
    def seed(self):
        mu=torch.tensor([[[-.1,0.,1.],[.1,0.,1.],[0.,-.1,1.],[0.,.1,1.]]])
        k=torch.tensor([[32.,0.,128.],[0.,32.,128.],[0.,0.,1.]]).expand(1,2,3,3).clone()
        return dict(mu=mu,radius=torch.full((1,4),.3),color=torch.rand(1,4,3),visual=torch.rand(1,4,2048),
            omega=dict(k=k,c2w=torch.eye(4).expand(1,2,4,4).clone(),native_patch_start=1))
    def model(self,dynamic):
        torch.manual_seed(42)
        opt=dataclasses.replace(options(),enc_embed_dim=32,token_dim=32,enc_num_heads=4,num_gs_tokens=4,mlp_ratio=2.)
        return install(build_model(opt),config_defaults['train_siu3r_unified_query_v1'],dynamic)
    def test_layout_units_and_gradients(self):
        seed=self.seed();moment,direction,special=memory_rays(seed,10)
        mu=seed['mu'].clone().requires_grad_();radius=seed['radius'].clone().requires_grad_();gamma=torch.tensor(0.,requires_grad=True)
        kwargs=dict(sigma0=1.,bandwidth_floor=1e-6,clamp_min=-20)
        bias=geometry_bias(mu,radius,moment,direction,special,10,gamma,**kwargs)
        self.assertEqual(bias.shape,(1,1,5,10));self.assertEqual(int(bias[:,:,0].count_nonzero()),0)
        self.assertEqual(int(bias[:,:,:,0].count_nonzero()),0);self.assertEqual(int(bias[:,:,:,5].count_nonzero()),0)
        scaled=geometry_bias(mu*4,radius*4,moment*4,direction,special,10,gamma,**dict(kwargs,bandwidth_floor=4e-6))
        torch.testing.assert_close(bias,scaled)
        bias.sum().backward();self.assertGreater(float(mu.grad.norm()),0);self.assertGreater(float(radius.grad.norm()),0);self.assertGreater(float(gamma.grad.abs()),0)
        with self.assertRaises(ValueError):memory_rays(seed,12)
    def test_pair_common_weights_local_decode_and_backward(self):
        a,b=self.model(False),self.model(True)
        common=dict(a.named_parameters());other=dict(b.named_parameters())
        self.assertTrue(all(torch.equal(p,other[n]) for n,p in common.items()))
        latent=SimpleNamespace(keys=torch.randn(1,4,10,8),values=torch.randn(1,4,10,8),eye_token=torch.randn(1,32),
            post_rgb_keys=torch.randn(1,4,8,8),post_rgb_values=torch.randn(1,4,8,8))
        seed=self.seed()
        for model in [a,b]:
            model.anchor_seed=seed;model.train();out=model.forward_decoder(latent)
            self.assertEqual(out.shape,(1,256,23));self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()
            self.assertGreater(float(model.gs_tokens.grad.norm()),0)
            self.assertGreater(float(model.anchor_content[0].weight.grad.norm()),0)
            self.assertGreater(float(model.anchor_visual[1].weight.grad.norm()),0)
            if model.anchor_dynamic:
                for head in list(model.anchor_refine_mu)+list(model.anchor_refine_rho):self.assertGreater(float(head.weight.grad.norm()),0)
                self.assertGreater(float(model.anchor_gamma.grad.norm()),0)
            # Zero raw position channels must decode exactly at the provided anchors, with no native z offset.
            with torch.no_grad():
                model.non_rgb_activation_head.deconv.weight[:3].zero_() # overwritten below by all64 child XYZ channels
                weight=model.non_rgb_activation_head.deconv.weight.reshape(64,10,32);weight[:,:3].zero_()
                bias=model.non_rgb_activation_head.deconv.bias.reshape(64,10);bias[:,:3].zero_()
                tokens=torch.randn(1,4,32);partial=model._decode_partial_gaussians(tokens)
                expected=model.anchor_final[0][:,:,None].expand(1,4,64,3).reshape(1,256,3)
                torch.testing.assert_close(partial[...,:3],expected)
    def test_checkpoint_eager_equivalence(self):
        model=self.model(True);seed=self.seed();model.anchor_seed=seed
        latent=SimpleNamespace(keys=torch.randn(1,4,10,8),values=torch.randn(1,4,10,8),eye_token=torch.randn(1,32),
            post_rgb_keys=torch.randn(1,4,8,8),post_rgb_values=torch.randn(1,4,8,8))
        model.eval();eager=model.forward_decoder(latent);eager.square().mean().backward();grad=model.gs_tokens.grad.clone();model.zero_grad(set_to_none=True)
        model.train();checked=model.forward_decoder(latent);checked.square().mean().backward()
        torch.testing.assert_close(eager,checked);torch.testing.assert_close(grad,model.gs_tokens.grad)
if __name__=='__main__':unittest.main()
