import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import torch

path = Path(__file__).resolve().parents[1] / 'tokengs/models/object_locus_readout_heads_v1.py'
spec = importlib.util.spec_from_file_location('readout_heads_test_module', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ReadoutContracts(unittest.TestCase):
    def test_export_quality_is_score_only_and_gate_applies_to_png_and_rows(self):
        export_path = path.parents[2] / 'scripts/export_object_locus_v3_set_official.py'
        export_spec = importlib.util.spec_from_file_location('readout_export_test', export_path)
        exporter = importlib.util.module_from_spec(export_spec)
        export_spec.loader.exec_module(exporter)
        pclass = torch.zeros(1, 100, 19)
        pclass[..., 18] = 1
        pclass[0, 0, 0], pclass[0, 0, 18] = .3, .7
        pclass[0, 1, 1], pclass[0, 1, 18] = .9, .1
        regions = torch.zeros(1, 2, 102, 4, 4)
        regions[:, :, 0, 2:] = .8
        regions[:, :, 1, :2] = .8
        output = dict(p_class=pclass, region_mass=regions, alpha=torch.ones(1, 2, 1, 4, 4),
                      semantic_scores=torch.ones(1, 2, 20, 4, 4))
        batch = dict(frame_ids=torch.tensor([[1, 2]]), semantic_label_all=torch.full((1, 2, 4, 4), 2),
                     instance_label_all=torch.ones(1, 2, 4, 4, dtype=torch.long))
        window = dict(scene='fixture', context=[1, 2], novel=[])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arm in ('reference', 'quality', 'objectness'):
                options = {}
                if arm == 'quality': options['score_multiplier'] = torch.full((100,), .5)
                if arm == 'objectness':
                    eligible = pclass[0].argmax(-1) != 18
                    eligible[0] = True
                    options['eligible_override'] = eligible
                exporter.write_official_pair(output, batch, window, root / arm, **options)
            parts = {arm: root / arm / 'fixture_context1_2/context_seg_pred' for arm in ('reference', 'quality', 'objectness')}
            a, q, o = [json.loads((parts[arm] / 'pred.json').read_text()) for arm in parts]
            self.assertEqual([row['id'] for row in a], [2])
            self.assertEqual([row['id'] for row in o], [1, 2])
            self.assertEqual(q[0]['score'], a[0]['score'] * .5)
            for filename in ('fixture_pred1.png', 'fixture_pred2.png'):
                self.assertEqual((parts['reference'] / filename).read_bytes(), (parts['quality'] / filename).read_bytes())
                self.assertNotEqual((parts['reference'] / filename).read_bytes(), (parts['objectness'] / filename).read_bytes())

    def test_zero_objectness_preserves_filter_and_conditional_classes(self):
        torch.manual_seed(42)
        logits = torch.randn(8, 100, 19)
        residual = torch.zeros(8, 100)
        original = logits.softmax(-1)
        self.assertTrue(torch.equal(original, m.candidate_probabilities(logits, residual)))
        shifted = m.candidate_probabilities(logits, residual + .7)
        self.assertTrue(torch.allclose(shifted[..., :18] / shifted[..., :18].sum(-1, keepdim=True), logits[..., :18].softmax(-1), atol=1e-6))
        old = (original.argmax(-1) != 18) & (original[..., :18].max(-1).values >= .05)
        self.assertTrue(torch.equal(old, m.candidate_eligibility(logits, residual)))

    def test_targets_reductions_and_gradient_ownership(self):
        torch.manual_seed(7)
        head = m.ScalarReadout()
        features = torch.randn(8, 100, 256, requires_grad=True)
        logits = torch.randn(8, 100, 19, requires_grad=True)
        batch = dict(features=features, class_logits=logits, quality_target=torch.rand(8, 100),
                     quality_valid=torch.rand(8, 100) > .8, object_target=torch.rand(8, 100) > .9)
        self.assertTrue(torch.equal(head(features).sigmoid(), torch.full((8, 100), .5)))
        for arm in ('quality', 'objectness'):
            head.zero_grad()
            whole = m.diagnostic_loss(head, batch, arm)
            split = sum(m.diagnostic_loss(head, {k: v[i:i+1] for k, v in batch.items()}, arm) for i in range(8)) / 8
            self.assertTrue(torch.allclose(whole, split, atol=1e-7))
            whole.backward()
            self.assertIsNone(features.grad)
            self.assertIsNone(logits.grad)
            self.assertGreater(float(head.net[-1].weight.grad.norm()), 0)
        empty = dict(batch, quality_valid=torch.zeros(8, 100, dtype=torch.bool))
        self.assertEqual(float(m.diagnostic_loss(head, empty, 'quality')), 0.)


if __name__ == '__main__':
    unittest.main()
