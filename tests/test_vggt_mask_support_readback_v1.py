"""Classification intervention must preserve masks, core gradients and controls."""
import unittest
import torch
from tokengs.models.object_locus_mask_support_readback_v1 import MaskSupportReadback


class Contracts(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(42)
        return (torch.randn(1,102,256,requires_grad=True),torch.randn(1,8,256,requires_grad=True),
                torch.rand(1,8,102,requires_grad=True),torch.ones(1,8,1,requires_grad=True))

    def test_zero_initialization_and_no_core_gradient(self):
        q,f,m,v=self.inputs();module=MaskSupportReadback('object_support')
        before=[x.detach().clone() for x in (q,f,m,v)]
        out=module(q,f,m,v)
        self.assertTrue(torch.equal(out,q))
        out.square().sum().backward()
        self.assertTrue(all(x.grad is None for x in (q,f,m,v)))
        self.assertGreater(float(module.projection[-1].weight.grad.norm()),0)
        self.assertTrue(all(torch.equal(x.detach(),y) for x,y in zip((q,f,m,v),before)))

    def test_scene_control_has_same_capacity_but_no_query_mask_dependence(self):
        q,f,m,v=self.inputs();scene=MaskSupportReadback('scene_support');obj=MaskSupportReadback('object_support')
        self.assertEqual(sum(p.numel() for p in scene.parameters()),sum(p.numel() for p in obj.parameters()))
        changed=m.detach().clone();changed[:,:,0]=0
        self.assertTrue(torch.equal(scene.pool(f,m,v),scene.pool(f,changed,v)))
        self.assertFalse(torch.equal(obj.pool(f,m,v)[:,0],obj.pool(f,changed,v)[:,0]))
        self.assertTrue(torch.equal(scene.pool(f,m,v)[:,0],scene.pool(f,m,v)[:,1]))

    def test_query_mask_permutation_tracks_the_support_and_empty_visibility_is_finite(self):
        q,f,m,v=self.inputs();obj=MaskSupportReadback('object_support')
        permutation=torch.randperm(100)
        shuffled=torch.cat((m[:,:,:100][:,:,permutation],m[:,:,100:]),2)
        torch.testing.assert_close(obj.pool(f,shuffled,v),obj.pool(f,m,v)[:,permutation],rtol=1e-6,atol=1e-6)
        pooled=obj.pool(f,m,torch.zeros_like(v))
        self.assertTrue(torch.isfinite(pooled).all());self.assertEqual(float(pooled.abs().sum()),0)

    def test_nonfinite_features_are_not_hidden_by_empty_visibility(self):
        q,f,m,v=self.inputs();f.data[0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):MaskSupportReadback('object_support').pool(f,m,torch.zeros_like(v))


if __name__=='__main__':unittest.main()
