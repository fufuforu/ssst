"""Adapted semantic pooling and the exact startup-receipt failure regression."""
import unittest
from unittest.mock import patch
import torch
from torch import nn
from tokengs.models.gfm_dual_stream_components import SemanticSAMQueryInitializer
from tokengs.models.object_locus_gfm_dual_stream_v3 import LocusGSGFMDualStreamV3
from scripts.gfm_dual_stream_transition import transition_receipt

class SemanticPoolTests(unittest.TestCase):
    def test_semantic_pooling_and_gradient_reaches_adapter(self):
        adapter=nn.Conv2d(2048,256,1)
        raw=torch.randn(1,2,2048,4,4)
        fm=adapter(raw.flatten(0,1)).reshape(1,2,256,4,4);fm.retain_grad()
        memory=LocusGSGFMDualStreamV3.query_seed_memory(None,raw,fm)
        self.assertIs(memory,fm)
        mask=torch.zeros(16,16,dtype=torch.bool);mask[1,1]=True
        points=torch.zeros(1,2,16,16,3);points[...,2]=1.
        init=SemanticSAMQueryInitializer();captured=[]
        handle=init.content_projection.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
        q,_,_,logs=init(memory,[[[dict(mask=mask,score=.9)],[]]],points,torch.randn(1,1024,3))
        handle.remove()
        self.assertTrue(torch.allclose(captured[0],fm[0,0,:,0,0]))
        self.assertIn('semantic adapter output',logs[0]['pooling'])
        (q[0,0]*torch.arange(256)).sum().backward()
        self.assertGreater(float(adapter.weight.grad.norm()),0.)
        self.assertGreater(float(fm.grad.norm()),0.)
        self.assertGreater(float(init.content_projection[1].weight.grad.norm()),0.)
    def test_raw_gfm_rejected_by_semantic_initializer(self):
        with self.assertRaises(ValueError):
            SemanticSAMQueryInitializer()(torch.randn(1,2,2048,4,4),[[[],[]]],torch.zeros(1,2,16,16,3),torch.randn(1,1024,3))
    def test_receipt_repeated_count_merges_without_duplicate_keyword_error(self):
        receipt=transition_receipt({'retained_optimizer_states':610,'source_completed_short_updates':50},{'retained_optimizer_states':610,'new_parameters':100})
        self.assertEqual(receipt['retained_optimizer_states'],610)
        self.assertTrue(receipt['optimizer_moments_retained_by_parameter_name'])
        with self.assertRaises(ValueError):transition_receipt({'retained_optimizer_states':610},{'retained_optimizer_states':609})
    def test_real_model_semantic_projection_shape(self):
        from tokengs.options import config_defaults
        with patch('tokengs.models.object_locus_dual_stream_v1.FrozenOmega',side_effect=lambda *a,**k:nn.Identity()),patch('tokengs.models.object_locus_dual_stream_v1.FrozenSAM',side_effect=lambda *a,**k:nn.Identity()),torch.device('meta'):
            model=LocusGSGFMDualStreamV3(config_defaults['train_siu3r_object_locus_gfm_dual_stream_v3'])
        self.assertEqual(model.query_initializer.content_projection[1].weight.shape,(256,256))
        self.assertEqual(model.write_strength,0.)

if __name__=='__main__':unittest.main()
