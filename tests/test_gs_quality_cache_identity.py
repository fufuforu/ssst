import copy,unittest
from scripts.gs_quality_cache_identity import validate_records

class CacheIdentityTests(unittest.TestCase):
    def setUp(self):
        self.windows=[dict(scene='train',context=[32,69],novel=[41,58]),dict(scene='train',context=[32,69],novel=[36,54]),dict(scene='eval',context=[1,2],novel=[3,4])]
        self.rows=[dict(window_index=i,part='training' if i<2 else 'official',scene=w['scene'],name=w['scene']+'_context'+'_'.join(map(str,w['context']))) for i,w in enumerate(self.windows)]
    def test_same_context_distinct_novel_training_windows_preserved(self):
        result=validate_records(self.rows,self.windows,2,1)
        self.assertEqual(result['training_windows'],2);self.assertEqual(result['training_unique_context_names'],1)
    def test_missing_or_duplicate_index_rejected(self):
        for rows in (self.rows[:-1],[self.rows[0],self.rows[0],self.rows[2]]):
            with self.assertRaisesRegex(ValueError,'index'):validate_records(rows,self.windows,2,1)
    def test_wrong_cohort_pairing_or_part_rejected(self):
        for key,value in [('scene','wrong'),('name','wrong'),('part','official')]:
            rows=copy.deepcopy(self.rows);rows[0][key]=value
            with self.assertRaisesRegex(ValueError,'identity'):validate_records(rows,self.windows,2,1)
    def test_duplicate_official_name_still_rejected(self):
        windows=[self.windows[0],self.windows[2],self.windows[2]]
        rows=[dict(self.rows[0]),dict(self.rows[2],window_index=1),dict(self.rows[2])]
        with self.assertRaisesRegex(ValueError,'official'):validate_records(rows,windows,1,2)
if __name__=='__main__':unittest.main()
