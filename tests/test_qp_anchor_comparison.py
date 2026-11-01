"""Exercise fixed1860 comparison outputs and reject mismatched input proofs."""
from pathlib import Path
import tempfile,unittest,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import qp_anchor_pipeline as p

class PairComparisonTests(unittest.TestCase):
    def test_same_results_and_input_rejection(self):
        reference=Path('/space/mawb/ssst/group_plus/semantic_resolution_v1/v4init_isolated_noca_v4recipe_20261022_v1')
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);control=base/'control'
            raw=p.read(reference/'eval/evaluation/metrics.json')['official_native_result']
            for run in [base/'v1',base/'v2',control]:
                (run/'eval/evaluation').mkdir(parents=True)
                for name in ['metrics.json','export_complete.json','gt_reference_manifest.jsonl']:
                    (run/'eval/evaluation'/name).symlink_to(reference/'eval/evaluation'/name)
                p.write(run/'FORMAL_COMPLETE.json',dict(status='PASSED'))
                p.write(run/'eval/novel_metrics.json',dict(result={k:raw[k] for k in ['psnr','ssim','lpips','absrel','rmse']}))
            p.write(base/'v1/configuration.json',dict(native_control_root=str(control),baseline4_root=str(reference)))
            p.pair(base)
            result=p.read(base/'PAIR_RESULTS.json');self.assertEqual(len(result['scores']),4)
            self.assertTrue(all(v==0 for row in result['deltas'] for k,v in row.items() if k!='model'))
            self.assertEqual(len((base/'PAIR_11METRICS.csv').read_text().splitlines()),8)
            from scripts import olda4_ablation_pipeline as audit
            original=audit.signatures
            with patch.object(audit,'signatures',side_effect=lambda root:{} if root==control/'eval' else original(root)):
                with self.assertRaises(AssertionError):p.pair(base)
if __name__=='__main__':unittest.main()
