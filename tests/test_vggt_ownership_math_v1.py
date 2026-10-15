import unittest
import torch
import torch.nn.functional as F
from scripts.vggt_ownership_math_v1 import feedback_attribution, parent_pool, js_divergence


class OwnershipMath(unittest.TestCase):
    def test_signed_feedback_contributions_reconstruct_capped_nonlinear_delta(self):
        torch.manual_seed(731)
        h, q = torch.randn(1, 7, 12), torch.randn(1, 5, 4)
        route = torch.randn(1, 7, 6).softmax(-1)
        projection = torch.nn.Linear(4, 12, bias=False)
        message = route[..., :5] @ F.layer_norm(q, (4,), eps=1e-5)
        sigma = (h.square().mean(-1, keepdim=True) + 1e-6).sqrt()
        raw = sigma * torch.tanh(projection(message) / sigma)
        cap = (.25 * h.norm(dim=-1, keepdim=True) / (raw.norm(dim=-1, keepdim=True) + 1e-6)).clamp(max=1)
        actual = .05 * raw * cap
        norm, signed, rebuilt, error = feedback_attribution(h, q, route, projection, 35776, .05, actual)
        self.assertTrue(torch.allclose(rebuilt, actual, atol=1e-7))
        self.assertTrue(torch.allclose(signed.sum(-1), torch.ones(1, 7), atol=2e-5))
        self.assertTrue((norm >= 0).all())
        self.assertLess(error, 1e-6)

    def test_parent_mapping_preserves_anchor_major_64_children(self):
        values = torch.cat((torch.ones(64, 2), 3 * torch.ones(64, 2)))
        mass = torch.cat((torch.ones(64), torch.zeros(64)))
        pooled, total = parent_pool(values, mass, 2)
        self.assertTrue(torch.equal(pooled, torch.tensor([[1., 1.], [0., 0.]])))
        self.assertTrue(torch.equal(total, torch.tensor([64., 0.])))

    def test_spatial_js_ignores_mass_scale_but_detects_correspondence_shuffle(self):
        x = torch.eye(4)
        self.assertTrue(torch.equal(js_divergence(x, x * 7), torch.zeros(4)))
        self.assertTrue(torch.allclose(js_divergence(x, x.roll(1, 0)), torch.ones(4)))


if __name__ == '__main__':
    unittest.main()
