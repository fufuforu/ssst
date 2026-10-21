"""Training gate, sampler and cache correctness; no trained-quality claims."""
import tempfile
import unittest
from pathlib import Path
import torch
import numpy as np
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches,acceptance,validate_resume_metadata
from scripts.dual_stream_frozen_cache import install_frozen_caches

class TrainingPolicyTests(unittest.TestCase):
    def test_full_epoch_and_global_batch_invariance(self):
        order=epoch_order(8337,0)
        self.assertEqual(len(order),8344);self.assertEqual(len(set(order)),8337)
        self.assertEqual(order,epoch_order(8337,0));self.assertNotEqual(order,epoch_order(8337,1))
        for world in (1,2,4,8):
            for step in (0,len(order)//8-1):
                rows=[rank_microbatches(order,step,r,world) for r in range(world)]
                reconstructed=[rows[r][m] for m in range(8//world) for r in range(world)]
                self.assertEqual(reconstructed,order[step*8:step*8+8])
    def fixture(self):
        before=dict(psnr=10.,loss_recon=1.,loss_understanding=5.,matched_soft_iou=.03,matched_binary_iou=.02,matched_classification_accuracy=.05,correct_class_nonempty_fraction=0.)
        after=dict(psnr=12.,loss_recon=.8,loss_understanding=4.,matched_soft_iou=.1,matched_binary_iou=.2,matched_classification_accuracy=.5,correct_class_nonempty_fraction=.3)
        proof=dict(finite=True,updated_families={f:True for f in ('reconstruction','understanding','queries','panoptic')})
        return before,after,proof
    def test_loss_down_alone_cannot_pass_gate(self):
        before,after,proof=self.fixture();after['matched_binary_iou']=0.
        self.assertFalse(acceptance(before,after,before,before,proof)['passed'])
    def test_valid_learning_passes_and_bad_validation_blocks(self):
        before,after,proof=self.fixture()
        self.assertTrue(acceptance(before,after,before,before,proof)['passed'])
        bad=dict(before,psnr=8.)
        self.assertFalse(acceptance(before,after,before,bad,proof)['passed'])
    def test_evidence_policy_allows_loss_increase_but_keeps_learning_and_finite_checks(self):
        before,after,proof=self.fixture();after.update(loss_recon=2.,loss_understanding=8.)
        self.assertFalse(acceptance(before,after,before,before,proof)['passed'])
        result=acceptance(before,after,before,before,proof,policy='learning_evidence_v2')
        self.assertTrue(result['passed']);self.assertFalse(result['observations']['train_understanding_loss_down_5pct'])
        proof['updated_families']['queries']=False
        self.assertFalse(acceptance(before,after,before,before,proof,policy='learning_evidence_v2')['passed'])
        proof['updated_families']['queries']=True;after['loss_recon']=float('nan')
        self.assertFalse(acceptance(before,after,before,before,proof,policy='learning_evidence_v2')['passed'])
    def test_explicit_resume_allows_only_code_and_gate_policy_change(self):
        saved=dict(execution_git_sha='a'*40,world_size=4,config={'lr':1e-4},manifest_sha256='data')
        current=dict(saved,execution_git_sha='b'*40,learning_gate_policy='learning_evidence_v2')
        validate_resume_metadata(saved,current,source_sha='a'*40)
        for changed in (dict(current,world_size=2),dict(current,manifest_sha256='wrong'),dict(current,config={'lr':2e-4})):
            with self.assertRaises(ValueError):validate_resume_metadata(saved,changed,source_sha='a'*40)
        with self.assertRaises(ValueError):validate_resume_metadata(saved,current)
        with self.assertRaises(ValueError):validate_resume_metadata(saved,current,source_sha='c'*40)
    def test_frozen_cache_exactness_and_rgb_invalidation(self):
        class Omega:
            source_identity={'id':'fake_for_unit_test_only'}
            def __init__(self):self.calls=0
            def forward(self,rgb):self.calls+=1;return {'points':rgb[:, :, :1].permute(0,1,3,4,2).repeat(1,1,1,1,3),'features':[rgb.clone()]}
        class SAM:
            source_identity={'id':'fake_for_unit_test_only'}
            def __init__(self):self.calls=0
            def forward(self,rgb):
                self.calls+=1
                return [[[dict(mask=(view[0].numpy()>.5),score=.95)] for view in scene] for scene in rgb]
        class Model:
            def __init__(self):self.frozen_omega=Omega();self.sam_proposals=SAM();self.calibcalls=0
            def calibrate_targets(self,rgb,generated):self.calibcalls+=1;return {'c2w':torch.eye(4)[None,None],'diagnostics':[{'scale':1.}]}
        model=Model();rgb=torch.rand(1,2,3,4,4)
        with tempfile.TemporaryDirectory() as temp:
            cache=install_frozen_caches(model,Path(temp))
            a=model.frozen_omega.forward(rgb);b=model.frozen_omega.forward(rgb)
            self.assertTrue(torch.equal(a['features'][0],b['features'][0]));self.assertEqual(model.frozen_omega.calls,1)
            a=model.sam_proposals.forward(rgb);b=model.sam_proposals.forward(rgb)
            self.assertTrue(np.array_equal(a[0][0][0]['mask'],b[0][0][0]['mask']));self.assertEqual(model.sam_proposals.calls,2)
            generation={'predicted_points':torch.ones(1,2,4,4,3)}
            a=model.calibrate_targets(rgb,generation);b=model.calibrate_targets(rgb,generation)
            self.assertTrue(torch.equal(a['c2w'],b['c2w']));self.assertEqual(model.calibcalls,1)
            model.calibrate_targets(rgb,{'predicted_points':generation['predicted_points']*2})
            self.assertEqual(model.calibcalls,2)
            changed=rgb.clone();changed[0,0,0,0,0]+=.1
            model.frozen_omega.forward(changed);model.sam_proposals.forward(changed)
            self.assertEqual(model.frozen_omega.calls,2);self.assertEqual(model.sam_proposals.calls,3)

if __name__=='__main__':unittest.main()
