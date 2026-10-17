import unittest
from dataclasses import replace
import torch
from tests.test_unified_query_v1 import config,cloud
from tokengs.models.dual_superpoint_initialization import DualSuperpointInitialization


class DualSeedTests(unittest.TestCase):
    def test_common_geometry_independent_widths_and_3d_gradients(self):
        torch.manual_seed(42)
        model=DualSuperpointInitialization(replace(config(),dim=32),config())
        c=cloud();c['point_superpoint_ids']=torch.tensor([0,0,1,1])
        reconstruction=torch.randn(4,32,requires_grad=True);understanding=torch.randn(4,16,requires_grad=True)
        out=model(c,reconstruction,understanding);a,b=out['reconstruction'],out['understanding']
        self.assertEqual(a['query'].shape,(1,2,32));self.assertEqual(b['query'].shape,(1,2,16))
        torch.testing.assert_close(a['centers'],b['centers']);torch.testing.assert_close(a['radii'],b['radii'])
        a['query'][0,0,0].backward();self.assertGreater(float(reconstruction.grad.norm()),0.);self.assertIsNone(understanding.grad)
        b['query'][0,0,0].backward();self.assertGreater(float(understanding.grad.norm()),0.)
        self.assertGreater(float(model.reconstruction.point_encoder.layers[0][0].weight.grad.norm()),0.)
        self.assertGreater(float(model.understanding.point_encoder.layers[0][0].weight.grad.norm()),0.)
        with self.assertRaises(ValueError):model(cloud(),reconstruction,understanding)
        with self.assertRaises(ValueError):model(c,reconstruction[:3],understanding)


if __name__=='__main__':unittest.main()
