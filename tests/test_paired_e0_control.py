import csv
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from test_fixed_query_oracles import example
from scripts.fixed_query_oracles import ARMS, oracle_variants
from scripts.export_object_locus_v3_set_official import write_official_pair
from scripts.paired_e0_checks import capture_native, verify_unmodified, audit_pair, verify_gt_files, verify_export_control
from scripts import score_fixed_query_diagnostics as scorer


class PairedE0Tests(unittest.TestCase):
    def setup_pair(self, directory):
        out, sem, ins = example()
        native = capture_native(out)
        variants, _ = oracle_variants(out, sem, ins)
        verify_unmodified(out, native, *variants['E0'])
        batch = dict(frame_ids=torch.tensor([[1,2,3,4,5,6]]), semantic_label_all=sem[:1].expand(6,-1,-1)[None],
                     instance_label_all=ins[:1].expand(6,-1,-1)[None])
        window = dict(scene='toy', context=[1,2], novel=[3,4,5,6])
        write_official_pair(*[variants['E0'][0], batch, window, Path(directory)/'current'], eligible_override=variants['E0'][1])
        pair = Path(directory)/'current/toy_context1_2'
        history = Path(directory)/'history/toy_context1_2'
        shutil.copytree(pair, history)
        return pair, history, native, out, variants

    def change_pixel(self, path):
        a = np.array(Image.open(path));a[0,0,0] ^= 1
        Image.fromarray(a).save(path)

    def test_historical_one_pixel_is_audited_not_silently_erased(self):
        with TemporaryDirectory() as d:
            pair, history, native, _, _ = self.setup_pair(d)
            self.change_pixel(history/'target_seg_pred/toy_pred3.png')
            verify_gt_files(pair, history)
            audit = audit_pair(pair, history, native['packed'], range(1,7), 'toy')
            self.assertTrue(audit['native_control_verified'])
            self.assertEqual(audit['checked_pngs'], 8)
            self.assertEqual(audit['historical_differing_pixels'], 1)
            self.assertEqual(audit['historical_differing_pngs'], 1)

    def test_changed_native_E0_fails_even_if_history_matches_it(self):
        with TemporaryDirectory() as d:
            pair, history, native, _, _ = self.setup_pair(d)
            self.change_pixel(pair/'target_seg_pred/toy_pred3.png')
            shutil.copyfile(pair/'target_seg_pred/toy_pred3.png', history/'target_seg_pred/toy_pred3.png')
            with self.assertRaisesRegex(ValueError, 'same-forward'):
                audit_pair(pair, history, native['packed'], range(1,7), 'toy')

    def test_GT_mismatch_and_missing_file_remain_hard_errors(self):
        with TemporaryDirectory() as d:
            pair, history, _, _, _ = self.setup_pair(d)
            self.change_pixel(pair/'target_seg_gt/toy_gt3.png')
            with self.assertRaisesRegex(ValueError, 'GT differs'):
                verify_gt_files(pair, history)
            (pair/'target_seg_gt/toy_gt3.png').unlink()
            with self.assertRaisesRegex(ValueError, 'GT file set'):
                verify_gt_files(pair, history)

    def test_modified_prediction_tensor_or_keep_is_rejected(self):
        out, sem, ins = example(); native = capture_native(out)
        variants, _ = oracle_variants(out, sem, ins)
        bad = variants['E0'][1].clone();bad[0] = ~bad[0]
        with self.assertRaisesRegex(ValueError, 'eligibility'):
            verify_unmodified(out, native, out, bad)
        out['region_mass'].add_(.001)
        with self.assertRaisesRegex(ValueError, 'tensor'):
            verify_unmodified(out, native, *variants['E0'])

    def test_export_receipt_requires_every_unique_window_verified(self):
        exported = dict(windows=1860, unique_scenes=312, parameter_updates=0,
                        records=[dict(name=str(i), native_control_verified=True) for i in range(1860)])
        receipt = dict(status='NATIVE_SAME_FORWARD_E0_VERIFIED', windows=1860)
        verify_export_control(exported, receipt)
        exported['records'][10]['native_control_verified'] = False
        with self.assertRaises(ValueError):verify_export_control(exported, receipt)
        exported['records'][10]['native_control_verified'] = True
        exported['records'][10]['name'] = '0'
        with self.assertRaises(ValueError):verify_export_control(exported, receipt)

    def test_full_scoring_uses_current_E0_and_preserves_historical_drift(self):
        # Actual scorer orchestration with official evaluate mocked, no GPU/job.
        raw = dict(psnr=23., ssim=.76, lpips=.38, absrel=.05, rmse=.17)
        for scope in ('context', 'target'):
            raw[scope+'_miou'] = .58;raw[scope+'_pq'] = .68
            raw[scope+'_map'] = {k:.2 for k in ('map','map_50','map_75','mar_1','mar_10','mar_100')}
        with TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();reference=Path(d)/'reference';reference.mkdir()
            reference_file=reference/'metrics.json'
            reference_file.write_text(json.dumps(dict(official_native_result=raw)))
            original=reference_file.read_bytes()
            records=[dict(name=str(i),native_control_verified=True,historical_differing_pixels=int(i==0)) for i in range(1860)]
            (root/'export_complete.json').write_text(json.dumps(dict(windows=1860,unique_scenes=312,parameter_updates=0,
                        records=records,baseline_differing_pngs=1)))
            (root/'PAIRED_BASELINE_VERIFIED.json').write_text(json.dumps(dict(status='NATIVE_SAME_FORWARD_E0_VERIFIED',windows=1860)))
            (root/'query_audit').mkdir()
            keys=('queries','baseline_gate_count','baseline_keep_count','supported_queries','supported_rejected_no_object',
                  'supported_rejected_confidence','supported_rejected_nms','wrong_class_supported')
            for i in range(1860):
                history={k:0 for k in ('checked_pngs','compared_pixels','historical_byte_differing_pngs','historical_differing_pngs',
                    'historical_differing_pixels','historical_semantic_differing_pixels','historical_instance_differing_pixels')}
                history.update(historical_max_score_difference=0.,historical_id_changes=[],historical_label_changes=[])
                (root/'query_audit'/f'{i}.json').write_text(json.dumps(dict(name=str(i),instances=[],historical_baseline_audit=history,**{k:0 for k in keys})))
            for arm in ARMS:(root/arm/'evaluation/all').mkdir(parents=True)
            values=[]
            for i in range(len(ARMS)):
                value=json.loads(json.dumps(raw))
                for scope in ('context','target'):value[scope+'_map']['map']=.25+(.1 if i==1 else 0)
                values.append(value)
            def checked(cmd,**kwargs):return scorer.SIU3R_COMMIT+'\n' if 'rev-parse' in cmd else ''
            with patch.object(scorer,'REFERENCE',reference),patch.object(scorer,'evaluate',side_effect=values),\
                 patch.object(scorer.subprocess,'check_output',side_effect=checked),\
                 patch.object(sys,'argv',['score','--run-root',str(root)]):scorer.main()
            self.assertEqual(reference_file.read_bytes(),original)
            historical=json.loads((root/'HISTORICAL_BASELINE_METRIC_AUDIT.json').read_text())
            self.assertAlmostEqual(historical['metric_delta']['context_map'],5.)
            self.assertFalse(historical['tolerance_pass_claimed'])
            self.assertFalse(historical['decoded_pixels_equal_to_history'])
            with (root/'metrics_and_deltas.csv').open() as f:rows={r['arm']:r for r in csv.DictReader(f)}
            self.assertAlmostEqual(float(rows['E1_minus_E0']['context_map']),10.)
            self.assertTrue((root/'COMPLETE.json').exists())
            self.assertFalse((root/'BASELINE_REPRODUCTION_VERIFIED.json').exists())


if __name__ == '__main__':unittest.main()
