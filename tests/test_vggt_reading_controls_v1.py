import unittest
import torch
from tokengs.models.object_locus_reading_controls_v1 import ReadingControl, support_bias, local_visual, depth_bias, ARMS
from tokengs.models.object_locus_panoptic_v1_controller import EvidenceAttention


class ReadingContracts(unittest.TestCase):
    def test_previous_support_locality_and_global_fallback(self):
        masks = torch.zeros(1, 4, 102, requires_grad=True)
        with torch.no_grad():
            masks[0, 0, 0] = 1
            masks[0, :, 2] = .3
        bias = support_bias(masks)
        self.assertEqual(tuple(bias.shape), (1, 102, 4))
        self.assertEqual(float(bias[0, 0, 0]), 0)
        self.assertLess(float(bias[0, 0, 1]), -2)
        self.assertTrue(torch.equal(bias[0, 1:3], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(bias[0, 100:], torch.zeros(2, 4)))
        self.assertFalse(bias.requires_grad)

    def test_two_view_projection_bounds_and_gradient(self):
        fm = torch.zeros(1, 2, 256, 4, 4, requires_grad=True)
        with torch.no_grad():
            fm[:, 0] = 2
            fm[:, 1] = 4
        mu = torch.tensor([[[0., 0., 1.], [0., 0., -1.], [10., 0., 1.]]], requires_grad=True)
        c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
        k = torch.tensor([[128., 0., 128.], [0., 128., 128.], [0., 0., 1.]]).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
        v, valid = local_visual(mu, fm, c2w, k)
        self.assertTrue(torch.allclose(v[0, 0], torch.full((256,), 3.)))
        self.assertEqual(valid.flatten().tolist(), [True, False, False])
        self.assertEqual(float(v[0, 1:].abs().sum()), 0.)
        v.sum().backward()
        self.assertGreater(float(fm.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(mu.grad).all())

    def test_along_ray_reliability_and_shared_translation(self):
        mu = torch.tensor([[[0., 0., 1.]]], requires_grad=True)
        points = torch.tensor([[[0., 0., 1.], [0., 0., 2.], [0., 0., 2.]]])
        d = torch.tensor([[[0., 0., 1.]]]).repeat(1, 3, 1)
        r = torch.ones(1, 1)
        conf = torch.tensor([[1., 0., 1.]])
        bias = depth_bias(mu, r, points, d, conf, .01)
        self.assertTrue(torch.equal(bias, torch.tensor([[[0., 0., -.5]]])))
        self.assertTrue(torch.equal(bias, depth_bias(mu + 5, r, points + 5, d, conf, .01)))
        bias.sum().backward()
        self.assertGreater(float(mu.grad.abs().sum()), 0)

    def test_zero_start_actual_attention_and_equal_capacity(self):
        torch.manual_seed(54)
        attention = EvidenceAttention()
        q, a = torch.randn(1, 102, 256), torch.randn(1, 4, 256)
        bias = torch.randn(1, 102, 4)
        support = torch.rand(1, 4, 102)
        original = attention(q, a, bias)
        for arm in ARMS:
            control = ReadingControl(arm)
            self.assertEqual(sum(p.numel() for p in control.parameters()), 0)
            actual = attention(q, a, control.object_bias(bias, support))
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(original, actual)))
        control = ReadingControl('object_local_read')
        control.completed_updates = 20
        actual = attention(q, a, control.object_bias(bias, support))
        self.assertFalse(torch.equal(original[0], actual[0]))
        self.assertTrue(torch.equal(control.object_bias(bias, None), bias))


if __name__ == '__main__':
    unittest.main()
