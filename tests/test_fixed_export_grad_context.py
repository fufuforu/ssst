"""Exercise the actual renderer adjoint under the exporter's actual context."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('actual_renderer_lift', ROOT / 'tokengs/models/object_locus_panoptic_v1_lift.py')
lift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lift)


class LinearRenderer:
    def __init__(self):
        self.weights = torch.tensor([[[[[2., 1.], [1., 3.]]], [[[1., 2.], [4., 1.]]]]])

    def render_feature_channels(self, gaussians, features, cam, intr):
        return {'images_pred': torch.einsum('bvhwn,bnc->bvchw', self.weights, features)}


class ExportGradContextTests(unittest.TestCase):
    def setUp(self):
        self.renderer = LinearRenderer()
        self.features = torch.arange(12., requires_grad=True).reshape(1, 2, 3, 1, 2)
        self.gaussians = torch.nn.Parameter(torch.zeros(1, 2, 14))
        self.decoder = SimpleNamespace(cam_view=torch.eye(4)[None, None].repeat(1, 2, 1, 1), intrinsics=torch.ones(1, 2, 4))

    def test_reproduce_old_inference_mode_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'does not require grad'):
            with torch.inference_mode():
                lift.lift_features(self.features, self.gaussians, self.decoder, self.renderer)

    def test_actual_export_context_allows_local_adjoint_without_parameter_updates(self):
        tree = ast.parse((ROOT / 'scripts/export_fixed_query_diagnostics.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export')
        block = next(n for n in function.body if isinstance(n, ast.With))
        context = compile(ast.Expression(body=block.items[0].context_expr), '<actual-export-context>', 'eval')
        before = self.gaussians.detach().clone()
        version = self.gaussians._version
        with eval(context, {'torch': torch}):
            evidence, gate, mass = lift.lift_features(self.features, self.gaussians, self.decoder, self.renderer)
            self.assertFalse(torch.is_grad_enabled())
        expected_mass = self.renderer.weights.sum((1, 2, 3))[..., None]
        expected_lift = torch.einsum('bvhwn,bvchw->bnc', self.renderer.weights, self.features.detach())
        torch.testing.assert_close(mass, expected_mass)
        torch.testing.assert_close(evidence, expected_lift / expected_mass)
        torch.testing.assert_close(gate, expected_mass / (expected_mass + 1))
        self.assertFalse(evidence.requires_grad)
        self.assertIsNone(self.gaussians.grad)
        self.assertEqual(self.gaussians._version, version)
        self.assertTrue(torch.equal(before, self.gaussians))


if __name__ == '__main__':
    unittest.main()
