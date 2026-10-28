import unittest
import torch
from scripts.membership_geometry_core import same_slots
from scripts.gs_quality_core import GSQualityHead,cross_scene_shuffle
from scripts.reconstruction_gap_core import align_gt_cameras,appearance_gaussians

class RepairTests(unittest.TestCase):
    def test_slot_gate_preserves_identity(self):
        self.assertTrue(same_slots(torch.tensor([[1,2]],dtype=torch.int32),torch.tensor([[1,2]],dtype=torch.int64)))
        self.assertFalse(same_slots(torch.tensor([1,2]),torch.tensor([2,1])))
        self.assertFalse(same_slots(torch.tensor([1,2]),torch.tensor([[1,2]])))
        with self.assertRaises(ValueError):same_slots(torch.tensor([1.]),torch.tensor([1]))
    def test_four_heads_and_shuffle(self):
        token=torch.randn(8,256,requires_grad=True);geo=torch.randn(8,32,requires_grad=True);initial={}
        for arm in ('A','B','C','D'):
            torch.manual_seed(31415);head=GSQualityHead(arm);initial[arm]=head.state_dict()
            y=head(token,geo);self.assertEqual(y.shape,(8,));(y-torch.linspace(0,1,8)).square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()))
        self.assertIsNone(token.grad);self.assertIsNone(geo.grad)
        for k in initial['C']:self.assertTrue(torch.equal(initial['C'][k],initial['D'][k]))
        scenes=['a']*4+['b']*3+['c']*2;d=cross_scene_shuffle(scenes)
        self.assertEqual(sorted(d.tolist()),list(range(9)))
        self.assertTrue(all(scenes[i]!=scenes[j] for i,j in enumerate(d)))
    def test_sparse_same_forward_masks_preserve_panoptic(self):
        from scripts.export_object_locus_v3_set_official import assemble_panoptic as original
        from scripts.reciprocity_panoptic import assemble_panoptic as adapted
        n=4;regions=torch.rand(1,6,n+2,8,8);p=torch.zeros(1,n,19);p[0,:,0]=.9;p[0,:,18]=.1
        p[0,2,0]=.1;p[0,2,18]=.9
        eligible=torch.tensor([True,False,True,True]);alpha=torch.ones(1,6,1,8,8);semantic=torch.rand(1,6,20,8,8)
        out=dict(region_mass=regions,p_class=p,instance_eligible=eligible,alpha=alpha,semantic_scores=semantic)
        keep=eligible&(p[0].argmax(-1)!=18)&(p[0,:,:18].max(-1).values>=.05)
        selected=torch.cat((keep.nonzero().flatten(),torch.tensor([n,n+1])))
        restored=torch.zeros_like(regions);restored[:,:,selected]=regions[:,:,selected]
        raw=semantic.argmax(2);onehot=torch.nn.functional.one_hot(raw,20).permute(0,1,4,2,3).float()
        a=original(out);b=adapted(dict(out,region_mass=restored,semantic_scores=onehot))
        for left,right in zip(a,b):self.assertTrue(torch.equal(left,right))

    def test_camera_sim3_recovery_and_degenerate_rejection(self):
        gt=torch.eye(4).repeat(6,1,1);gt[:,0,3]=torch.arange(6.)*.2
        theta=.4;c=torch.cos(torch.tensor(theta));s=torch.sin(torch.tensor(theta));r=torch.tensor([[c,0,s],[0,1,0],[-s,0,c]])
        p=gt[:2].clone();p[:,:3,:3]=r;p[:,:3,3]=2*(gt[:2,:3,3]@r.T)+torch.tensor([1.,2.,3.])
        out,a=align_gt_cameras(gt,p);self.assertTrue(a['reliable']);torch.testing.assert_close(out[:2],p)
        self.assertAlmostEqual(a['scale'],2.,places=5)
        gt[1]=gt[0];out,a=align_gt_cameras(gt,p);self.assertIsNone(out);self.assertFalse(a['reliable'])
    def test_appearance_gradient_only_colour(self):
        fixed=torch.randn(1,9,14,requires_grad=True);col=torch.randn(1,9,3,requires_grad=True)
        out=appearance_gaussians(fixed,col);out.square().sum().backward()
        self.assertIsNone(fixed.grad);self.assertGreater(float(col.grad.abs().sum()),0)
        self.assertTrue(torch.equal(out[...,:11],fixed[...,:11]))
if __name__=='__main__':unittest.main()
