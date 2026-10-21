"""End-to-end coverage/paired report smoke using synthetic fixed-weight receipts."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from diagnostics.anchor_child_boundary.metrics import feature_metrics

class Report(unittest.TestCase):
    def test_full_report_pairs_by_scene_and_rejects_different_gt(self):
        root_repo=Path(__file__).resolve().parents[2]
        rng=np.random.default_rng(8);shared=rng.normal(size=(1,8));res=rng.normal(size=(1,64,8));c=np.zeros((64,2));c[:32,0]=4;c[32:,1]=4
        stats,anchors=feature_metrics(shared,res,(shared[:,None]+res).reshape(64,8),rng.uniform(.1,.9,(64,3)),c,np.full(64,5.))
        boundary=dict(f1_2px=.8,continuous_ap_2px=.7,interior_raw_response_fraction=.05,interior_threshold_fraction=.03)
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);config=base/'configuration.json';config.write_text(json.dumps(dict(sources=dict(A4={},A4plus4={}))))
            output=base/'full';output.mkdir()
            for label in ('A4','A4plus4'):
                records=[]
                for i in range(128):
                    folder=output/label/f'window_{i:03d}_scene{i}';folder.mkdir(parents=True)
                    for vi in range(2):
                        (folder/f'context{vi}_sixpanel.png').write_bytes(b'synthetic')
                        (folder/f'context{vi}_boundary.npz').write_bytes(b'synthetic')
                    row=dict(index=i,scene=f'scene{i}',context=[1,2],input_rgb_sha256='rgb',gt_sha256='gt',source_superpoint_sha256='sp',feature=stats,boundary=[boundary,boundary],equivalence={},anchors=anchors)
                    path=folder/'metrics.json';path.write_text(json.dumps(row));records.append(dict(index=i,path=str(path)))
                for rank in range(8):
                    (output/f'{label}_rank{rank:02d}_complete.json').write_text(json.dumps(dict(records=records[rank::8],backward_calls=0,parameter_updates=0,weight_sha256_before='fixed',weight_sha256_after='fixed')))
            cmd=[sys.executable,'scripts/summarize_anchor_child_boundary.py','--config',str(config),'--output',str(output),'--mode','full']
            good=subprocess.run(cmd,cwd=root_repo,capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stderr)
            self.assertTrue((output/'publication/REPORT.md').is_file())
            self.assertEqual(len(list((output/'publication').glob('*.png'))),16)
            badrow=output/'A4plus4/window_000_scene0/metrics.json';row=json.loads(badrow.read_text());row['gt_sha256']='different';badrow.write_text(json.dumps(row))
            bad=subprocess.run(cmd,cwd=root_repo,capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0);self.assertIn('Paired protocol mismatch: gt_sha256',bad.stderr)

if __name__=='__main__':unittest.main()
