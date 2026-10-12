import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location('overfit', Path(__file__).parents[1] / 'scripts/train_vggt_overfit_diagnostic_v1.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


class OverfitContracts(unittest.TestCase):
    def test_exact_equal_window_exposures_global16(self):
        counts = [0] * 4
        for rank in range(8):
            for slot in diag.rank_window_slots(rank):
                counts[slot] += 1
        self.assertEqual(counts, [4, 4, 4, 4])
        self.assertEqual(sum(counts) * diag.TOTAL_UPDATES, 4800)

    def test_short_warmup_then_common_peak(self):
        self.assertEqual(diag.pilot_lr_update(0), 9)
        self.assertEqual(diag.pilot_lr_update(19), 199)
        self.assertEqual(diag.pilot_lr_update(299), 199)
        with self.assertRaises(ValueError):
            diag.pilot_lr_update(300)

    def test_positive_gt_gauge_and_global_pose_invariance(self):
        cams = torch.eye(4).repeat(1, 4, 1, 1)
        cams[0, :, 0, 3] = torch.tensor([0., .3, .4, .5])
        pred = cams[:, :2].clone()
        pred[0, 1, 0, 3] = .1
        gt, scale = diag.gt_baseline_gauge(cams, pred)
        torch.testing.assert_close(scale, torch.tensor([1/3]))
        torch.testing.assert_close(gt[:, 1, :3, 3], pred[:, 1, :3, 3])
        transform = torch.eye(4)
        transform[:3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transform[:3, 3] = torch.tensor([3., -2., 5.])
        moved, _ = diag.gt_baseline_gauge(transform @ cams, pred)
        torch.testing.assert_close(gt, moved)
        torch.testing.assert_close(cams[0, 1, 0, 3], torch.tensor(.3))

    def test_invalid_gt_gauge_is_not_hidden(self):
        cams = torch.eye(4).repeat(1, 2, 1, 1)
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            diag.gt_baseline_gauge(cams, cams)
        cams[0, 1, 0, 3] = .1
        bad = cams.clone()
        bad[0, 1, 0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'improper'):
            diag.gt_baseline_gauge(bad, cams)

    def test_cache_preserves_dtype_and_has_no_gradient(self):
        x = torch.ones(3, requires_grad=True)
        value = diag.tree_to({'x': (x, torch.ones(2, dtype=torch.bfloat16))}, 'cpu')
        self.assertFalse(value['x'][0].requires_grad)
        self.assertEqual(value['x'][1].dtype, torch.bfloat16)

    def test_trainable_whitelist_excludes_encoders_and_feedback(self):
        for name in ('frozen_vggt.model.aggregator.frame_blocks.0.weight',
                     'understanding.encoder.enc_blocks.0.weight', 'panoptic.layers.L6.weight', 'lpips.weight'):
            self.assertFalse(name.startswith(diag.TRAIN_PREFIXES))
        for name in ('gs_tokens', 'anchor_decoder.mu', 'enc_dec_backbone.decoder_blocks.0.weight',
                     'activation_head.proj.weight', 'vggt_memory_adapter.proj.weight'):
            self.assertTrue(name.startswith(diag.TRAIN_PREFIXES))


if __name__ == '__main__':
    unittest.main()
