"""CPU-only route, parameter identity, and fixed sampling contracts."""
import unittest
import copy

import numpy as np
import torch
from torch import nn

from tokengs.models.object_locus_panoptic_v1_controller import RegisteredObjectLayer
from tokengs.models.object_locus_mask_guided import MaskGuidedRegisteredObjectLayer
from scripts.object_locus_mask_guided_runtime import lr_multiplier, manifest_and_plan
from scripts.object_locus_mask_guided_runtime import build_optimizer


class ObjectLocusMaskGuidedContracts(unittest.TestCase):
    def test_class_conversion_preserves_parameter_state_and_rng(self):
        torch.manual_seed(42)
        control = nn.ModuleDict({f'L{x}': RegisteredObjectLayer() for x in (6,8,10,12)})
        torch.manual_seed(42)
        masked = nn.ModuleDict({f'L{x}': RegisteredObjectLayer() for x in (6,8,10,12)})
        expected = {k: v.clone() for k, v in control.state_dict().items()}
        keys_shapes = {k: (tuple(v.shape), v.dtype) for k, v in masked.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        for layer in masked.values():
            layer.__class__ = MaskGuidedRegisteredObjectLayer
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(keys_shapes, {k: (tuple(v.shape), v.dtype) for k, v in masked.state_dict().items()})
        for key, value in expected.items():
            self.assertTrue(torch.equal(value, masked.state_dict()[key]))

    def test_mask_route_and_evidence_normalization_axes(self):
        torch.manual_seed(7)
        layer = MaskGuidedRegisteredObjectLayer().eval()
        b = 1
        h = torch.randn(b, 1024, 1024)
        a = torch.randn(b, 1024, 256)
        mu = torch.randn(b, 1024, 3)
        q = torch.randn(b, 102, 256)
        c = torch.randn(b, 100, 3)
        s = torch.ones(b, 100, 3)
        ell = torch.ones(b)
        image = torch.randn(b, 2048, 256)
        f = torch.randn(b, 1024, 256)
        embed = nn.Linear(256, 256, bias=False)
        out = layer(h,a,mu,q,c,s,ell,image,embed,f,8)
        self.assertEqual(tuple(out['route'].shape), (1,1024,103))
        self.assertEqual(tuple(out['anchor_mask_logits'].shape), (1,1024,102))
        self.assertTrue(torch.allclose(out['route'].sum(-1), torch.ones(1,1024), atol=1e-6))
        self.assertEqual(out['evidence_attention'].shape[-1], 1024)
        self.assertTrue(torch.allclose(out['evidence_attention'].sum(-1), torch.ones_like(out['evidence_attention'].sum(-1)), atol=1e-6))
        z = f @ out['m_query'].transpose(1,2)
        expected = torch.cat((z, z.new_zeros(1,1024,1)), -1).softmax(-1)
        self.assertTrue(torch.allclose(out['route'], expected))
        self.assertTrue(torch.allclose(out['anchor_membership'], z.sigmoid()))

        original=RegisteredObjectLayer().eval();control=copy.deepcopy(original)
        control.load_state_dict(original.state_dict(),strict=True)
        baseline=original(h,a,mu,q,c,s,ell,image,embed,f,8)
        unchanged=control(h,a,mu,q,c,s,ell,image,embed,f,8)
        for key,value in baseline.items():
            if torch.is_tensor(value):self.assertTrue(torch.equal(value,unchanged[key]),key)

    def test_optimizer_parameter_coverage_and_no_decay(self):
        model=nn.Module()
        model.reconstruction=nn.Linear(4,4)
        model.understanding=nn.Module()
        model.understanding.weight=nn.Parameter(torch.randn(4,4))
        model.understanding.norm=nn.LayerNorm(4)
        model.understanding.level_embed=nn.Parameter(torch.randn(1,4))
        model.panoptic=nn.Linear(4,4)
        optimizer=build_optimizer(model)
        ids=[id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters()})
        group_for={id(p):g for g in optimizer.param_groups for p in g['params']}
        self.assertEqual(group_for[id(model.reconstruction.weight)]['weight_decay'],0.0)
        self.assertEqual(group_for[id(model.understanding.weight)]['weight_decay'],0.05)
        self.assertEqual(group_for[id(model.understanding.level_embed)]['weight_decay'],0.0)

    def test_fixed_schedule_and_global_batch_plan(self):
        self.assertAlmostEqual(lr_multiplier(0), 1/25)
        self.assertAlmostEqual(lr_multiplier(24), 1.0)
        manifest, plan = manifest_and_plan()
        self.assertEqual(len(plan['entries']), 448)
        counts = np.zeros(56, dtype=np.int64)
        for epoch in range(64):
            expected = np.random.default_rng(42+epoch).permutation(56)
            entries = plan['entries'][epoch*7:(epoch+1)*7]
            seen = [i for row in entries for i in row['rank_windows']]
            self.assertEqual(seen, expected.tolist())
            counts[seen] += 1
        self.assertTrue(np.all(counts == 64))
        self.assertEqual(len(manifest['train_all56']), 56)


if __name__ == '__main__':
    unittest.main()
