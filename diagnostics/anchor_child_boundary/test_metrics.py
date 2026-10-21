import unittest
import numpy as np
from diagnostics.anchor_child_boundary.metrics import attribute,feature_metrics,boundary_response,boundary_metrics,average_precision

class Definitions(unittest.TestCase):
    def test_attribution_uses_contributions_and_abstains(self):
        c=np.array([[4.,1.],[1.,4.],[2.5,2.5],[.1,0.]])
        label,_,_,_=attribute(c,np.array([6.,6.,6.,.1]))
        np.testing.assert_array_equal(label,[0,1,-1,-1])
    def test_features(self):
        rng=np.random.default_rng(1);s=rng.normal(size=(2,8));r=rng.normal(size=(2,64,8));f=(s[:,None]+r).reshape(128,8)
        c=np.zeros((128,2));c[:32,0]=4;c[32:64,1]=4;c[64:,0]=4
        m,rows=feature_metrics(s,r,f,rng.uniform(.1,.9,(128,4)),c,np.full(128,5.))
        self.assertEqual(m['multi_instance_anchors'],1);self.assertEqual(m['assigned_gs'],128)
        self.assertEqual(m['cross.purity0.7.child.cosine_different']['n'],1024)
        self.assertEqual(len(rows),2)
    def test_no_gt_instances(self):
        m,_=feature_metrics(np.ones((1,8)),np.zeros((1,64,8)),np.ones((64,8)),np.full((64,3),.5),np.zeros((64,0)),np.ones(64))
        self.assertEqual(m['assigned_gs'],0)
        self.assertEqual(m['multi_instance_anchors'],0)
    def test_boundaries_and_interior(self):
        h=32;sem=np.full((h,h),2);ins=np.ones((h,h),int);ins[:,16:]=2
        region=np.stack([ins==1,ins==2]).astype(float);raw=boundary_response(region)
        metrics,soft,gt=boundary_metrics(raw,sem,ins)
        self.assertGreater(metrics['f1_2px'],.99);self.assertLess(metrics['interior_raw_response_fraction'],1e-6)
        region[:,8,:]=1-region[:,8,:]
        bad=boundary_metrics(boundary_response(region),sem,ins)[0]
        self.assertGreater(bad['interior_raw_response_fraction'],.1)
    def test_tied_ap(self):
        self.assertAlmostEqual(average_precision(np.ones(4),np.array([True,False,True,False])),.5)
    def test_forward_transpose(self):
        w=np.array([[.6,.2,0.],[0.,.2,.3]])
        f=np.array([[1.,2.],[3.,4.]]);u=np.array([[2.,3.],[5.,6.],[7.,8.]])
        self.assertAlmostEqual(float(((w.T@f)*u).sum()),float((f*(w@u)).sum()))

if __name__=='__main__':unittest.main()
