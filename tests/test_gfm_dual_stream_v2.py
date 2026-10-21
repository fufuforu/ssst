import unittest
from unittest.mock import patch
import torch
from torch import nn
from tokengs.models.gfm_dual_stream_components import GFMReconstructionAdapter,GFMSAMQueryInitializer,sample_anchor_gfm
from tokengs.models.dual_stream_initialization import seed_anchors

class GFMComponentsTests(unittest.TestCase):
    def test_kv_and_new_adapter_gradients(self):
        adapter=GFMReconstructionAdapter()
        layers=[torch.randn(1,2,1024,2048) for _ in range(4)]
        out=adapter(layers)
        self.assertEqual(out.keys.shape,(1,16,2048,64));self.assertEqual(out.values.shape,out.keys.shape)
        out.values.square().mean().backward()
        self.assertGreater(float(adapter.projection[0].weight.grad.norm()),0.)
    def test_anchor_visual_uses_source_view(self):
        grid=torch.cat((torch.ones(1,1,3,4,4),torch.full((1,1,3,4,4),7.)),1)
        selected=sample_anchor_gfm(grid,torch.tensor([[0,255,256,511]]),image_hw=(16,16))
        self.assertTrue(torch.equal(selected[0,:,0],torch.tensor([1.,1.,7.,7.])))
    def test_small_mask_pools_raw_gfm_before_projection(self):
        init=GFMSAMQueryInitializer();gfm=torch.randn(1,2,2048,4,4)
        mask=torch.zeros(16,16,dtype=torch.bool);mask[3,3]=True
        points=torch.zeros(1,2,16,16,3);points[...,2]=1.
        mu=torch.randn(1,1024,3)*.1
        captured=[]
        hook=init.content_projection.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
        q,c,s,logs=init(gfm,[[[dict(mask=mask,score=.9)],[]]],points,mu)
        hook.remove()
        self.assertEqual(logs[0]['proposal_queries'],1)
        self.assertTrue(torch.allclose(captured[0],gfm[0,0,:,0,0]))
        (q[0,0]*torch.arange(256)).sum().backward()
        self.assertGreater(float(init.content_projection[1].weight.grad.norm()),0.)
    def test_point_seed_source_ids_correspond(self):
        points=torch.randn(1,2,16,16,3);confidence=torch.ones(1,2,1,16,16);rgb=torch.rand(1,2,3,16,16)
        mu,r,c,ids=seed_anchors(points,confidence,rgb,count=32,candidate_side=8,return_source_indices=True)
        self.assertTrue(torch.equal(mu[0],points.reshape(-1,3)[ids[0]]))
    def test_old_encoder_absent_and_two_adapters_present(self):
        from tokengs.options import config_defaults
        from tokengs.models.object_locus_gfm_dual_stream_v2 import LocusGSGFMDualStreamV2,REMOVED_ENCODER_PREFIXES
        with patch('tokengs.models.object_locus_dual_stream_v1.FrozenOmega',side_effect=lambda *a,**k:nn.Identity()),patch('tokengs.models.object_locus_dual_stream_v1.FrozenSAM',side_effect=lambda *a,**k:nn.Identity()),torch.device('meta'):
            model=LocusGSGFMDualStreamV2(config_defaults['train_siu3r_object_locus_gfm_dual_stream_v2'])
        names=list(dict(model.named_parameters()))
        self.assertFalse(any(n.startswith(REMOVED_ENCODER_PREFIXES) for n in names))
        for prefix in ('reconstruction_adapter.','understanding.','query_initializer.content_projection.','anchor_visual_projection.'):
            self.assertTrue(any(n.startswith(prefix) for n in names))
        self.assertEqual(model.write_strength,0.)

if __name__=='__main__':unittest.main()
