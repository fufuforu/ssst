import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('diag', Path(__file__).parents[1] / 'scripts/diagnose_vggt_reconstruction_v1.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class DiagnosticGauges(unittest.TestCase):
    def test_global_gt_gauge_is_removed_and_scales_are_distinct(self):
        cams = torch.eye(4).repeat(1, 3, 1, 1)
        cams[0, 1, 0, 3] = .15
        cams[0, 2, 0, 3] = .3
        pred = cams[:, :2].clone()
        pred[0, 1, 0, 3] = .05
        depth = torch.full((1, 2, 1, 3, 3), .75)
        base, canonical, legacy, _ = diag.gt_cameras(cams, pred, depth)
        gauge = torch.eye(4)
        gauge[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        gauge[:3, 3] = torch.tensor([3., -7., 4.])
        moved, _, _, _ = diag.gt_cameras(gauge @ cams, pred, depth)
        torch.testing.assert_close(base, moved)
        torch.testing.assert_close(base[:, 1, :3, 3], pred[:, 1, :3, 3])
        torch.testing.assert_close(canonical[:, 1, :3, 3], legacy[:, 1, :3, 3] / 3)
        torch.testing.assert_close(cams[:, 1, 0, 3], torch.tensor([.15]))

    def test_degenerate_gauges_fail_explicitly(self):
        cams = torch.eye(4).repeat(1, 2, 1, 1)
        depth = torch.ones(1, 2, 1, 3, 3)
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            diag.gt_cameras(cams, cams, depth)
        cams[0, 1, 0, 3] = 1
        with self.assertRaisesRegex(ValueError, 'depth'):
            diag.gt_cameras(cams, cams, depth * 0)

    def test_intrinsics_resize_preserves_pixel_rays(self):
        v = torch.tensor([[[210., 200., 127., 120.]]])
        k = diag.k_matrix(v)
        resized = k.clone()
        resized[..., :2, :] *= 518 / 256
        pixel256 = torch.tensor([70., 40., 1.])
        pixel518 = pixel256.clone()
        pixel518[:2] *= 518 / 256
        torch.testing.assert_close(torch.linalg.inv(k) @ pixel256,
                                   torch.linalg.inv(resized) @ pixel518)

    def test_psnr_preserves_pinned_zero_endpoint_range(self):
        gt = torch.linspace(.2, .9, 24).reshape(1, 3, 2, 4)
        _, value = diag.image_psnr(gt + .1, gt)
        torch.testing.assert_close(value, torch.tensor([19.08485031]))


if __name__ == '__main__':
    unittest.main()
