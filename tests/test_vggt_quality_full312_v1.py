"""Expanded evaluation must preserve query IDs and canonical C evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.eval_vggt_quality_full312_v1 import quality_rows, link_pair


class FixedQualityExportTests(unittest.TestCase):
    def test_scores_follow_query_ids_without_mutating_canonical_rows(self):
        original = [{'id': 100, 'label_id': 5, 'score': .8}, {'id': 2, 'label_id': 7, 'score': .4}]
        multipliers = np.ones(100, dtype=np.float32)
        multipliers[99], multipliers[1] = .25, .5
        result = quality_rows(original, multipliers)
        self.assertEqual([r['score'] for r in result], [.2, .2])
        self.assertEqual(original[0]['score'], .8)
        with self.assertRaises(AssertionError):
            quality_rows(original, np.full(100, np.nan))
        with self.assertRaises(AssertionError):
            quality_rows([dict(original[0], id=101)], multipliers)

    def test_images_remain_canonical_and_score_json_is_owned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical, destination = root / 'C', root / 'Q'
            for sub in ('rgb', 'depth', 'rgb_gt', 'depth_gt', 'context_seg_gt', 'target_seg_gt'):
                (canonical / sub).mkdir(parents=True)
                (canonical / sub / 'evidence.png').write_bytes(b'fixed-evidence')
            rows = [{'id': 2, 'label_id': 7, 'score': .8}]
            for view in ('context', 'target'):
                p = canonical / (view + '_seg_pred')
                p.mkdir()
                (p / 'mask.png').write_bytes(b'fixed-mask')
                (p / 'pred.json').write_text(json.dumps(rows))
            for filename in ('render_scores.json', 'depth_scores.json'):
                (canonical / filename).write_text('[]')
            link_pair(destination, canonical, np.full(100, .5))
            for view in ('context', 'target'):
                p = destination / (view + '_seg_pred')
                self.assertTrue((p / 'mask.png').is_symlink())
                self.assertEqual((p / 'mask.png').read_bytes(), b'fixed-mask')
                self.assertFalse((p / 'pred.json').is_symlink())
                self.assertEqual(json.loads((p / 'pred.json').read_text())[0]['score'], .4)
                self.assertEqual(json.loads((canonical / (view + '_seg_pred') / 'pred.json').read_text()), rows)


if __name__ == '__main__':
    unittest.main()
