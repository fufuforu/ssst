import unittest
import torch

from scripts.vggt_feature_probe_math_v1 import new_stats, accumulate, solve, design, pool_image


class ProbeContracts(unittest.TestCase):
    def test_stream_and_full_equal_solutions_and_gradient_detachment(self):
        torch.manual_seed(27)
        x = torch.randn(8, 100, 256, requires_grad=True)
        batch = dict(class_target=torch.randint(0, 19, (8, 100)),
            quality_target=torch.rand(8, 100), quality_valid=torch.rand(8, 100) > .4)
        a, b = new_stats('cpu'), new_stats('cpu')
        accumulate(a, x, batch)
        for i in range(8):
            accumulate(b, x[i:i + 1], {k: v[i:i + 1] for k, v in batch.items()})
        ca, cb = solve(a), solve(b)
        self.assertTrue(torch.allclose(design(x) @ ca['class'], design(x) @ cb['class'], atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(design(x) @ ca['quality'], design(x) @ cb['quality'], atol=2e-5, rtol=2e-5))
        self.assertIsNone(x.grad)
        self.assertFalse(design(x).requires_grad)

    def test_predicted_support_keeps_views_and_empty_support_zero(self):
        features = torch.zeros(1, 2, 3, 2, 2)
        features[:, 0] = 2
        features[:, 1] = 9
        masks = torch.zeros(1, 2, 100, 4, 4)
        masks[:, 0, 0] = 1
        masks[:, 1, 1] = 1
        pooled = pool_image(features, masks)
        self.assertTrue(torch.equal(pooled[0, 0], torch.full((3,), 2.)))
        self.assertTrue(torch.equal(pooled[0, 1], torch.full((3,), 9.)))
        self.assertTrue(torch.equal(pooled[0, 2:], torch.zeros(98, 3)))

    def test_null_probe_is_weighted_prior_and_empty_quality_is_zero(self):
        x = torch.zeros(2, 100, 256)
        labels = torch.full((2, 100), 18)
        labels[:, :10] = 3
        batch = dict(class_target=labels, quality_target=torch.zeros(2, 100), quality_valid=torch.zeros(2, 100, dtype=torch.bool))
        stat = new_stats('cpu')
        accumulate(stat, x, batch)
        coef = solve(stat)
        score = design(x) @ coef['class']
        self.assertAlmostEqual(float(score[0, 0, 3]), 10 / 19, places=5)
        self.assertAlmostEqual(float(score[0, 0, 18]), 9 / 19, places=5)
        self.assertTrue(torch.equal(coef['quality'], torch.zeros(257, 1)))


if __name__ == '__main__':
    unittest.main()
