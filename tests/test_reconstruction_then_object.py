"""Final reconstruction -> geometric Object read -> semantic Object read."""
import unittest
import torch
from torch import nn

from tokengs.models.reconstruction_then_object import LocusGSReconstructionThenObject, ReconstructionThenObjectCore
from tokengs.models.dynamic_panoptic_controller import DynamicPanopticController
from tokengs.models.superpoint_structural_variants import LocusGSSuperpointDualPointReconstruction
from tokengs.models.unified_superpoints import UnifiedQueryConfig
from tests.test_superpoint_dual_point_reconstruction import build
from tests.test_unified_query_v1 import opt, FakeOmega, FakeSAM, FakeRenderer


def reader_fixture():
    core = ReconstructionThenObjectCore.__new__(ReconstructionThenObjectCore)
    nn.Module.__init__(core)
    core.panoptic = DynamicPanopticController()
    core.semantic_memory = nn.Module()
    core.semantic_memory.mask_embedder = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))
    # A initializes attention residual outputs at zero. Activate these outputs
    # to test live gradient routes after those trainable gates have moved.
    for layer in core.panoptic.layers.values():
        nn.init.eye_(layer.anchor_attention.O.weight)
        nn.init.eye_(layer.image_attention.O.weight)
    rec = torch.randn(1, 3, 1024, requires_grad=True)
    mu = torch.randn(1, 3, 3)
    states = [dict(layer=i+1, tokens=rec+(i+1)*.01, mu=mu+(i+1)*.001,
                   radii=torch.full((1, 3), .2)) for i in range(12)]
    under = dict(query=torch.randn(1, 2, 256, requires_grad=True), centers=torch.randn(1, 2, 3),
                 radii=torch.full((1, 2), .3))
    memory = dict(fm=torch.randn(1, 2, 256, 4, 4, requires_grad=True))
    return core, states, under, memory, rec


class ReconstructionThenObjectTests(unittest.TestCase):
    def test_exact_parameter_layout_and_group_coverage_matches_point1024(self):
        with torch.device('meta'):
            # Full config has two point-encoder layers and 664 state tensors;
            # the fast forward fixture intentionally has only one layer.
            def full(cls):
                return cls(opt().evolve(dec_patch_size=8, img_size=(8, 8)), config=UnifiedQueryConfig(),
                           omega=FakeOmega(), sam=FakeSAM(), renderer=FakeRenderer())
            old, new = full(LocusGSSuperpointDualPointReconstruction), full(LocusGSReconstructionThenObject)
        # Match checkpoint semantics: frozen backbone tensors are excluded.
        # The synthetic backbones each have one dummy weight.
        a = {k: v for k, v in old.state_dict().items() if not k.startswith(('frozen_omega.', 'sam_proposals.'))}
        b = {k: v for k, v in new.state_dict().items() if not k.startswith(('frozen_omega.', 'sam_proposals.'))}
        self.assertEqual(set(a), set(b))
        self.assertEqual(len(b), 664)
        for k in a:
            self.assertEqual(a[k].shape, b[k].shape, k)
        ids = [id(p) for ps in new.parameter_groups().values() for p in ps]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in new.parameters() if p.requires_grad})
        meta = new.extra_checkpoint_metadata['reconstruction_then_object']
        self.assertEqual(meta['reconstruction_read_layers'], [12]*4)
        self.assertFalse(meta['detached_reconstruction'])

    def test_live_final_geometry_and_semantic_gradient_routes(self):
        core, states, under, memory, rec = reader_fixture()
        out = core.object_updates(states, under, memory)
        loss = out[-1]['q'][..., 0].square().mean()
        grads = torch.autograd.grad(loss, (rec, memory['fm'], under['query']))
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.norm()), 0.)
        for layer in (6, 8, 10, 12):
            self.assertEqual(out[layer-1]['reconstruction_read_layer'], 12)

    def test_semantic_changes_never_write_reconstruction_or_replace_superpoint_seed(self):
        core, states, under, memory, _ = reader_fixture()
        before = [{k: s[k].detach().clone() for k in ('tokens', 'mu', 'radii')} for s in states]
        qpre = under['query'].detach().clone()
        with torch.no_grad():
            first = core.object_updates([dict(s) for s in states], under, memory)
            other = core.object_updates([dict(s) for s in states], under, dict(fm=memory['fm']+torch.randn_like(memory['fm'])))
        self.assertGreater(float((first[-1]['q']-other[-1]['q']).abs().max()), 0.)
        torch.testing.assert_close(under['query'], qpre, rtol=0, atol=0)
        for output in (first, other):
            for original, actual in zip(before, output):
                for key in original:
                    torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)

    def test_real_1024_forward_has_all_reconstruction_before_geometry_semantic_reads(self):
        model = build(LocusGSReconstructionThenObject).eval()
        events = []; evidence = []; hooks = []
        for i, block in enumerate(model.core.blocks):
            hooks.append(block.gs_cross_attn.register_forward_pre_hook(
                lambda m, args, index=i: events.append(('reconstruction', index+1))))
        for index in (6, 8, 10, 12):
            layer = model.core.panoptic.layers[f'L{index}']
            hooks.append(layer.anchor_attention.register_forward_pre_hook(
                lambda m, args, i=index: events.append(('geometry', i))))
            hooks.append(layer.image_attention.register_forward_pre_hook(
                lambda m, args, i=index: events.append(('semantic', i))))
            hooks.append(layer.register_forward_pre_hook(lambda m, args: evidence.append(args[0])))
        with torch.no_grad():
            out = model.generate(torch.rand(1, 2, 3, 256, 256))
        for hook in hooks:
            hook.remove()
        expected = [('reconstruction', i+1) for i in range(12)]
        expected += [event for i in (6, 8, 10, 12) for event in [('geometry', i), ('semantic', i)]]
        self.assertEqual(events, expected)
        self.assertEqual(len(evidence), 4)
        for tokens in evidence:
            torch.testing.assert_close(tokens, out['states'][-1]['tokens'], rtol=0, atol=0)
            self.assertEqual(tokens.shape, (1, 1024, 1024))
        self.assertEqual(out['num_thing_queries'], 2)
        self.assertEqual(out['gaussians'].shape, (1, 65536, 14))
        self.assertEqual(out['gaussian_membership'].shape, (1, 65536, 4))
        self.assertTrue(torch.isfinite(out['gaussians']).all())
        self.assertTrue(torch.isfinite(out['gaussian_membership']).all())


if __name__ == '__main__':
    unittest.main()
