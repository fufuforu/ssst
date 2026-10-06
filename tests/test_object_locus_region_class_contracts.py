"""CPU contracts for region-conditioned final classification."""
from __future__ import annotations

import gc
import inspect
import unittest

import numpy as np
import torch
from torch import nn

from scripts import object_locus_region_class_runtime as runtime
from tokengs.models.object_locus_region_class import (
    RegionClassObjectLocusPanopticV1Recon, add_region_class_projection,
    classify_with_region, pool_region_features,
)


class TinyPanoptic(nn.Module):
    def __init__(self):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1))
        self.head = nn.Linear(256, 19, bias=False)

    def classify(self, q):
        logits = self.head(q[:, :100])
        probs = logits.softmax(-1)
        return {'thing_logits19': logits, 'class_logits19': logits, 'p_class': probs}


class RegionClassContracts(unittest.TestCase):
    def test_two_rank_microbatch_order_preserves_locked_global_batch(self):
        entry = {'rank_windows': [13, 4, 29, 2, 41, 5, 17, 33]}
        rank0 = runtime.rank_micro_windows(entry, 0)
        rank1 = runtime.rank_micro_windows(entry, 1)
        self.assertEqual(rank0, [13, 29, 41, 17])
        self.assertEqual(rank1, [4, 2, 5, 33])
        self.assertEqual([item for pair in zip(rank0, rank1) for item in pair], entry['rank_windows'])

    def test_pool_shapes_two_view_constant_and_low_mass(self):
        weights = torch.zeros(2, 2, 100, 3, 4)
        features = torch.zeros(2, 2, 256, 3, 4)
        weights[:, :, 0] = 0.25
        features[:] = 3.5
        pooled, mass = pool_region_features(weights, features)
        self.assertEqual(tuple(pooled.shape), (2, 100, 256))
        self.assertEqual(tuple(mass.shape), (2, 100))
        self.assertTrue(torch.allclose(pooled[:, 0], torch.full((2, 256), 3.5)))
        self.assertTrue(torch.equal(pooled[:, 1:], torch.zeros_like(pooled[:, 1:])))
        self.assertTrue(torch.allclose(mass[:, 0], torch.full((2,), 6.0)))

    def test_projection_rng_zero_init_and_common_state(self):
        torch.manual_seed(42)
        c = TinyPanoptic()
        torch.manual_seed(42)
        r = TinyPanoptic()
        before = {k: v.detach().clone() for k, v in r.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        add_region_class_projection(r, seed=31417)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for k, value in before.items():
            self.assertTrue(torch.equal(value, r.state_dict()[k]), k)
            self.assertTrue(torch.equal(value, c.state_dict()[k]), k)
        self.assertEqual(r.region_class_proj.weight.numel(), 65_536)
        self.assertEqual(int(r.region_class_proj.weight.count_nonzero()), 0)

    def test_fusion_changes_only_classifier_input_and_disabled_path(self):
        torch.manual_seed(3)
        p = TinyPanoptic()
        add_region_class_projection(p)
        q = torch.randn(2, 102, 256)
        z = torch.randn(2, 100, 256)
        original = q.clone()
        baseline = p.classify(q)
        fused, q_class = classify_with_region(p, q, z)
        self.assertTrue(torch.equal(q, original))
        self.assertTrue(torch.equal(q_class[:, 100:], q[:, 100:]))
        self.assertTrue(torch.equal(q_class[:, :100], q[:, :100]))
        self.assertTrue(torch.equal(fused['thing_logits19'], baseline['thing_logits19']))
        disabled, q_disabled = classify_with_region(p, q, z, enabled=False)
        self.assertIs(q_disabled, q)
        self.assertTrue(torch.equal(disabled['thing_logits19'], baseline['thing_logits19']))

    def test_nonzero_test_projection_backpropagates_to_features_and_mass(self):
        torch.manual_seed(7)
        p = TinyPanoptic()
        add_region_class_projection(p)
        with torch.no_grad():
            nn.init.xavier_uniform_(p.region_class_proj.weight)
        weights = torch.rand(1, 2, 100, 4, 5, requires_grad=True)
        feature = torch.randn(1, 2, 256, 4, 5, requires_grad=True)
        pooled, _ = pool_region_features(weights, feature)
        q = torch.randn(1, 102, 256)
        cls, _ = classify_with_region(p, q, pooled)
        loss = nn.functional.cross_entropy(cls['thing_logits19'].reshape(-1, 19),
            torch.randint(0, 19, (100,)))
        loss.backward()
        for grad in (weights.grad, feature.grad, p.region_class_proj.weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.norm()), 0.0)

    def test_readout_has_context_only_inputs_and_no_gt(self):
        source = inspect.getsource(RegionClassObjectLocusPanopticV1Recon._readout)
        self.assertIn('read_context_decoder', source)
        self.assertIn('feature_grid', source)
        self.assertNotIn('semantic_label', source)
        self.assertNotIn('instance_label', source)
        self.assertNotIn('novel', source)

    def test_optimizer_projection_is_new_decay_and_coverage_is_exact(self):
        model = nn.Module()
        model.reconstruction = nn.Linear(4, 4)
        model.understanding = nn.Module()
        model.understanding.weight = nn.Parameter(torch.randn(4, 4))
        model.understanding.norm = nn.LayerNorm(4)
        model.panoptic = TinyPanoptic()
        add_region_class_projection(model.panoptic)
        optimizer = runtime.build_optimizer(model)
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        groups = [g for g in optimizer.param_groups if id(model.panoptic.region_class_proj.weight) in {id(p) for p in g['params']}]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['name'], 'new_decay')
        self.assertEqual(groups[0]['peak_lr'], 1e-4)
        self.assertEqual(groups[0]['weight_decay'], 0.05)

    def test_full_fresh_build_shared_state_parameter_counts_and_plan(self):
        manifest, plan = runtime.manifest_and_plan()
        saved_manifest = runtime.CM_CONTROL / 'data_manifest.json'
        saved_plan = runtime.CM_CONTROL / 'training_plan.json'
        self.assertEqual(manifest, runtime.json.loads(saved_manifest.read_text()))
        self.assertEqual(plan, runtime.json.loads(saved_plan.read_text()))
        c, _ = runtime.build_model('control', 'cpu', report=False)
        c_hash = runtime.state_sha(c.state_dict())
        self.assertEqual(sum(p.numel() for p in c.parameters()), 572_432_103)
        del c
        gc.collect()
        r, _ = runtime.build_model('region_class', 'cpu', report=False)
        self.assertEqual(sum(p.numel() for p in r.parameters()), 572_497_639)
        self.assertEqual(runtime.state_sha(r.state_dict(), exclude_region=True), c_hash)
        opt = runtime.build_optimizer(r)
        group = next(g for g in opt.param_groups if id(r.panoptic.region_class_proj.weight) in {id(p) for p in g['params']})
        self.assertEqual(group['name'], 'new_decay')
        del opt, r
        gc.collect()


if __name__ == '__main__':
    unittest.main()
