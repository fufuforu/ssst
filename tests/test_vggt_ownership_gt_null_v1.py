import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.analyze_vggt_ownership_v1 import gt_write_null


class OwnershipGTNull(unittest.TestCase):
    def test_correct_writes_are_distinguished_from_wrong_GT_correspondence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scene.json'
            norm = np.zeros((2,100),np.float16); norm[0,0]=1;norm[1,1]=1
            arrays = {f'L{layer}_contribution':norm for layer in (6,8,10,12)}
            arrays['parent_GT_mass']=np.eye(2,dtype=np.float32)
            arrays['parent_mass']=np.ones(2,dtype=np.float32)
            np.savez(path.with_suffix('.npz'),**arrays)
            record=dict(gt_ids=[1,2],source_index=7,diagnostic_query_owner=[0,1]+[-1]*98,
                ownership={str(layer):dict(write_other_GT_fraction=[0.]*100) for layer in (6,8,10,12)})
            result=gt_write_null(record,path)
            self.assertEqual(result['12']['write_other_GT_label_shuffle'][:2],[1.,1.])
            self.assertEqual(result['12']['write_own_GT_preference_over_null'][:2],[1.,1.])
            self.assertEqual(result['12']['write_object_fraction_of_visible_write'][:2],[1.,1.])
            self.assertEqual(result['12']['write_other_GT_fraction_of_visible_write'][:2],[0.,0.])

    def test_no_GT_does_not_invent_a_selectivity_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'empty.json'
            arrays={f'L{layer}_contribution':np.ones((2,100),np.float16) for layer in (6,8,10,12)}
            arrays['parent_GT_mass']=np.empty((2,0),np.float32)
            arrays['parent_mass']=np.ones(2,dtype=np.float32)
            np.savez(path.with_suffix('.npz'),**arrays)
            record=dict(gt_ids=[],source_index=8,diagnostic_query_owner=[-1]*100,ownership={str(layer):{} for layer in (6,8,10,12)})
            result=gt_write_null(record,path)
            self.assertTrue(all(v is None for v in result['12']['write_own_GT_preference_over_null']))


if __name__=='__main__':unittest.main()
