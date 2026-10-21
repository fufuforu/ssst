"""Targeted checks for saved-mask replay, including the official void/FN convention."""
import unittest
import numpy as np
from scripts.audit_point1024_pq_counts_20261022 import counts

class PQReplay(unittest.TestCase):
    def test_perfect(self):
        p=np.full(10,3001);r=counts(p,p)
        np.testing.assert_array_equal(r[2],[1,1,0,0,1,1])
    def test_wrong_class(self):
        r=counts(np.full(10,4001),np.full(10,3001))
        self.assertEqual(r[2,3],1);self.assertEqual(r[3,2],1);self.assertEqual(r[:,1].sum(),0)
    def test_gt_mostly_predicted_void_is_ignored_by_official_pq(self):
        r=counts(np.full(10,21000),np.full(10,3001))
        self.assertEqual(r[2,4],1);self.assertEqual(r[2,3],0);self.assertEqual(r[2,1],0)
    def test_iou_uses_official_float32(self):
        r=counts(np.array([3001,3001,3001]),np.array([3001,3001,4001]))
        self.assertEqual(r[2,0],float(np.float32(2)/np.float32(3)))
        self.assertEqual(r[2,1],1);self.assertEqual(r[3,3],1)

if __name__=='__main__':unittest.main()
