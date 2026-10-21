"""Geometry and proposal contracts; synthetic masks are not SAM quality evidence."""
import unittest
import torch
from tokengs.models.dual_stream_initialization import unproject,seed_anchors,SAMQueryInitializer
from tokengs.models.dual_stream_calibration import align_omega_targets

class InitializationTests(unittest.TestCase):
    def geometry(self):
        depth=torch.ones(1,2,1,16,16)
        k=torch.tensor([[16.,0,8],[0,16.,8],[0,0,1]]).expand(1,2,3,3).clone()
        cam=torch.eye(4).expand(1,2,4,4).clone()
        return unproject(depth,k,cam),k,cam
    def test_reprojection(self):
        p,k,cam=self.geometry()
        xy=torch.einsum('bvij,bvhwj->bvhwi',k,p); xy=xy[...,:2]/xy[...,2:]
        self.assertTrue(torch.allclose(xy[0,0,3,5],torch.tensor([5.5,3.5])))
    def test_anchor_selection_and_invalid(self):
        p,_,_=self.geometry(); p[:,:,0]=float('nan')
        mu,rgb_radius,rgb=seed_anchors(p,torch.ones(1,2,1,16,16),torch.rand(1,2,3,16,16),count=32,candidate_side=8)
        self.assertEqual(mu.shape,(1,32,3)); self.assertTrue(torch.isfinite(mu).all()); self.assertTrue((rgb_radius>0).all())
        with self.assertRaises(ValueError):seed_anchors(p*float('nan'),torch.ones(1,2,1,16,16),torch.rand(1,2,3,16,16),count=32)
    def test_empty_fallback(self):
        p,_,_=self.geometry(); init=SAMQueryInitializer(dim=8)
        q,c,s,log=init(torch.rand(1,2,8,16,16),[[[],[]]],p,p.flatten(1,3))
        self.assertEqual(q.shape,(1,100,8));self.assertEqual(log[0]['learned_queries'],100)
        q.square().sum().backward();self.assertIsNotNone(init.fallback.grad)
    def test_merge_dedup_and_gradient(self):
        p,_,_=self.geometry(); init=SAMQueryInitializer(dim=8)
        fm=torch.ones(1,2,8,16,16,requires_grad=True)
        mask=torch.zeros(16,16,dtype=torch.bool);mask[3:9,3:9]=True
        prop=dict(mask=mask,score=.95)
        q,c,s,log=init(fm,[[[prop,prop],[prop]]],p,p.flatten(1,3))
        self.assertEqual(log[0]['proposal_queries'],1);self.assertEqual(log[0]['cross_view_groups'],1)
        q.sum().backward();self.assertIsNotNone(fm.grad)
    def test_distant_views_do_not_merge(self):
        p,_,_=self.geometry();p[:,1]+=10
        mask=torch.ones(16,16,dtype=torch.bool);prop=dict(mask=mask,score=.9)
        *_,log=SAMQueryInitializer(dim=8)(torch.ones(1,2,8,16,16),[[[prop],[prop]]],p,p.flatten(1,3))
        self.assertEqual(log[0]['proposal_queries'],2)
    def test_calibration_similarity_and_frozen_context(self):
        p,k,cam=self.geometry();p=p.repeat_interleave(16,2).repeat_interleave(16,3)
        # A non-planar support cloud for a well-constrained fit.
        p=p.clone();p[...,2]+=p[...,0].square()+p[...,1].square()
        cam=cam.repeat(1,2,1,1)[:,:3].clone();cam[:,2,0,3]=.4
        source=dict(points=p/2+torch.tensor([1.,2.,3.]),confidence=torch.ones(1,3,1,256,256),c2w=cam,k=k.repeat(1,2,1,1)[:,:3])
        generated=dict(predicted_points=p,confidence=torch.ones(1,2,1,256,256),predicted_context_c2w=cam[:,:2],predicted_context_intrinsics_matrix=k)
        result=align_omega_targets(source,generated)
        self.assertAlmostEqual(result['diagnostics'][0]['scale'],2.,places=4)
        self.assertTrue(torch.equal(result['c2w'][:,:2],cam[:,:2]))

if __name__=='__main__':unittest.main()
