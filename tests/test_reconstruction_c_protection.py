import copy,unittest
from pathlib import Path
import torch
from scripts.reconstruction_c_protection import capture,verify
from scripts.reconstruction_abcd_gate_reuse import update_signature,task_signature

class ProtectionTests(unittest.TestCase):
    def data(self):
        return dict(gaussians=torch.zeros(1,2,14),gaussian_membership=torch.full((1,2,2),.5),p_class=torch.full((1,2,2),.5),unified_query=torch.full((1,2,2),.5))
    def test_capture_uses_matching_no_grad(self):
        class Model:
            def generate(inner,rgb):
                self.assertFalse(torch.is_grad_enabled());return self.data()
        with torch.enable_grad():result=capture(Model(),torch.zeros(1))
        self.assertTrue(all(not t.requires_grad for t in result.values()))
    def test_small_roundoff_reported_and_real_semantic_change_rejected(self):
        before=self.data();repeat=copy.deepcopy(before);after=copy.deepcopy(before);after['gaussians'][...,11:]=.1
        repeat['gaussian_membership']+=1e-7;after['gaussian_membership']+=2e-7
        report=verify(before,repeat,after);self.assertEqual(report['status'],'PASSED')
        self.assertFalse(report['semantic_outputs']['gaussian_membership']['bitwise_equal'])
        after['gaussian_membership']+=1e-3
        self.assertEqual(verify(before,repeat,after)['status'],'FAILED')
    def test_noisy_control_never_raises_hard_cap(self):
        before=self.data();repeat=copy.deepcopy(before);after=copy.deepcopy(before);after['gaussians'][...,11:]=.1
        repeat['p_class']+=1e-3;after['p_class']+=1e-4
        r=verify(before,repeat,after);self.assertEqual(r['status'],'FAILED');self.assertLessEqual(r['semantic_outputs']['p_class']['allowed_abs'],1e-5)
    def test_geometry_protection_stays_exact(self):
        before=self.data();after=copy.deepcopy(before);after['gaussians'][...,11:]=.1;after['gaussians'][...,0]=1e-9
        with self.assertRaisesRegex(ValueError,'geometry/alpha changed'):verify(before,before,after)
    def test_reused_ABD_task_and_update_loop_are_identical(self):
        before=Path('/space/mawb/ssst_reconstruction_abcd_code_20261025_v3_devicefix/scripts/train_reconstruction_abcd.py')
        now=Path(__file__).resolve().parents[1]/'scripts/train_reconstruction_abcd.py'
        self.assertEqual(task_signature(before),task_signature(now))
        self.assertEqual(update_signature(before),update_signature(now))

if __name__=='__main__':unittest.main()
