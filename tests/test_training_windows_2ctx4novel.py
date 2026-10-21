import copy,json,tempfile,unittest
from pathlib import Path
from scripts.training_windows_2ctx4novel import PROTOCOL,extend_window,validate_manifest,load_training_windows

class WindowExtensionTests(unittest.TestCase):
    def parent(self):return dict(scene='scene',context=[0,10],novel=[2,8],target=[0,10,2,8],sampling_seed=42,thing_instance_count=3)
    def test_deterministic_context_and_parent_novel_preserved(self):
        p=self.parent();r=extend_window(p,list(range(11)),0,'sha')
        self.assertEqual(r,extend_window(p,list(range(11)),0,'sha'));self.assertEqual(r['context'],p['context'])
        self.assertEqual(len(r['novel']),4);self.assertEqual(len(set(r['target'])),6);self.assertTrue(set(p['novel']).issubset(r['novel']))
        self.assertTrue(all(0<x<10 for x in r['novel']));self.assertEqual(p,self.parent())
    def test_validation_and_identity_refuse_omission_or_context_changes(self):
        p=self.parent();r=extend_window(p,list(range(11)),0,'sha');d=dict(protocol=PROTOCOL,parent_manifest_sha256='sha',records=[r])
        self.assertEqual(validate_manifest(d,[p],'sha'),[r])
        for case in ('context','missing_old','protocol','parent_sha','metadata','order'):
            b=copy.deepcopy(d)
            if case=='context':b['records'][0]['context']=[0,11]
            if case=='missing_old':b['records'][0]['novel']=[1,3,5,7]
            if case=='protocol':b['protocol']='old'
            if case=='parent_sha':b['parent_manifest_sha256']='bad'
            if case=='metadata':b['records'][0]['thing_instance_count']=4
            if case=='order':b['records'][0]['target'].reverse()
            with self.subTest(case=case):
                with self.assertRaises(ValueError):validate_manifest(b,[p],'sha')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json';path.write_text(json.dumps(d))
            rows,identity=load_training_windows(path,[p],'sha');self.assertEqual(identity['total_supervised_views'],6)
            self.assertEqual(len(identity['training_window_manifest_sha256']),64)
    def test_short_or_missing_temporal_pool_fails_without_context_resampling(self):
        with self.assertRaises(ValueError):extend_window(self.parent(),[0,2,8,10],0,'sha')
        with self.assertRaises(ValueError):extend_window(self.parent(),[0,1,3,5,7,10],0,'sha')
