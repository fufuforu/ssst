"""CPU contracts for independent reconstruction/understanding token counts."""
from dataclasses import replace
import unittest

import torch

from tests.test_unified_query_v1 import config, opt, FakeOmega, FakeSAM, FakeRenderer
from tokengs.models.superpoint_structural_variants import (
    MODELS, LocusGSSuperpointDual, LocusGSSuperpointDualPointReconstruction)
from tokengs.models.gfm_dual_stream_components import gfm_grid, sample_anchor_gfm
from tokengs.models.dual_stream_initialization import seed_anchors


def cfg():
    return replace(config(), dim=256, heads=8, depth=12, superpoint_source='esam_sam',
                   matching_mode='source_many_to_one')


def build(cls=LocusGSSuperpointDualPointReconstruction):
    return cls(opt().evolve(dec_patch_size=8, img_size=(8, 8)), config=cfg(),
               omega=FakeOmega(), sam=FakeSAM(), renderer=FakeRenderer())


class PointReconstructionTests(unittest.TestCase):
    def test_shared_architecture_and_parameter_groups(self):
        with torch.device('meta'):
            old, new = build(LocusGSSuperpointDual), build()
        a, b = old.state_dict(), new.state_dict()
        removed = ('core.initializer.reconstruction.', 'core.reconstruction_seed_projection.')
        added = ('core.gs_tokens', 'core.anchor_content.', 'core.anchor_visual_projection.')
        self.assertEqual(set(a)-set(b), {k for k in a if k.startswith(removed)})
        self.assertEqual(set(b)-set(a), {k for k in b if k.startswith(added)})
        for k in set(a) & set(b):
            self.assertEqual(a[k].shape, b[k].shape, k)
        self.assertIs(MODELS['dual_point1024'], type(new))
        groups = new.parameter_groups()
        ids = [id(p) for group in groups.values() for p in group]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in new.parameters() if p.requires_grad})
        rec_ids = {id(p) for p in groups['reconstruction']}
        for name, p in new.core.named_parameters():
            if name.startswith(('gs_tokens', 'anchor_content.', 'anchor_visual_projection.')):
                self.assertIn(id(p), rec_ids)

    def test_original_point_seed_formula_sources_and_gradients(self):
        # Allocate only the new seed modules; the 12-layer decoder stays meta.
        with torch.device('meta'):
            core = build().core
        core.gs_tokens = torch.nn.Parameter(torch.randn(1024, 1024)*.02)
        for module in (core.anchor_content, core.anchor_visual_projection):
            module.to_empty(device='cpu')
            for child in module.modules():
                if hasattr(child, 'reset_parameters'):
                    child.reset_parameters()
        rgb = torch.rand(1, 2, 3, 256, 256)
        omega = FakeOmega()(rgb)
        # Native pixel coordinates, as in real Omega output.
        omega['points'] = torch.randn(1, 2, 256, 256, 3, requires_grad=True)
        omega['confidence'] = torch.ones(1, 2, 1, 256, 256)
        result = core.reconstruction_seed(omega, rgb)
        mu, radius, color, source = seed_anchors(omega['points'], omega['confidence'], rgb,
                                               1024, return_source_indices=True)
        self.assertTrue(torch.equal(result['source_indices'], source))
        self.assertTrue(torch.equal(mu[0], omega['points'].reshape(-1, 3)[source[0]]))
        origin = mu.mean(1, keepdim=True)
        ell = (mu-origin).square().sum(-1).mean(-1).sqrt().clamp_min(.05)
        content = torch.cat(((mu-origin)/ell[:, None, None],
                             (radius/ell[:, None])[..., None], color), -1)
        expected = (core.gs_tokens[None].expand(1, -1, -1).clone() + core.anchor_content(content)
                    + core.anchor_visual_projection(sample_anchor_gfm(gfm_grid(omega['patch_layers']), source)))
        torch.testing.assert_close(result['tokens'], expected, rtol=0, atol=0)
        result['tokens'].square().mean().backward()
        self.assertIsNone(omega['points'].grad)
        for p in (core.gs_tokens, core.anchor_content[0].weight, core.anchor_visual_projection[1].weight):
            self.assertGreater(float(p.grad.norm()), 0.)

    def test_real_decoder_readout_with_1024_anchors_and_two_queries(self):
        model = build().eval()
        with torch.no_grad():
            out = model.generate(torch.rand(1, 2, 3, 256, 256))
        self.assertEqual(out['num_thing_queries'], 2)
        self.assertEqual(out['num_reconstruction_tokens'], 1024)
        self.assertEqual(out['gaussians'].shape, (1, 65536, 14))
        self.assertEqual(out['gaussian_membership'].shape, (1, 65536, 4))
        self.assertEqual(out['states'][-1]['tokens'].shape, (1, 1024, 1024))
        self.assertEqual(out['thing_logits19'].shape, (1, 2, 19))
        self.assertNotIn('gaussian_parent_query', out)
        self.assertEqual(out['gaussian_parent_anchor'].shape, (65536,))
        self.assertEqual(int(out['gaussian_parent_anchor'].max()), 1023)
        self.assertTrue(torch.isfinite(out['gaussians']).all())
        self.assertTrue(torch.isfinite(out['gaussian_membership']).all())


if __name__ == '__main__':
    unittest.main()
