import unittest
import torch
from scripts.gfm_instance_diagnostic_math import repeat_calibrated_witness


class WitnessTests(unittest.TestCase):
    def values(self):
        result={k:torch.ones(2,3) for k in ('gaussians','gaussian_membership','p_class','region_mass')}
        result['packed']=torch.zeros(100,100,dtype=torch.long)
        return result

    def test_accepts_exact_and_measured_native_jitter(self):
        a=self.values();b=self.values();c=self.values()
        self.assertTrue(repeat_calibrated_witness(a,b,c)['passed'])
        b['gaussian_membership'][0,0]+=1e-6;c['gaussian_membership'][1,0]+=2e-6
        proof=repeat_calibrated_witness(a,b,c)
        self.assertTrue(proof['passed']);self.assertGreater(proof['tensors']['gaussian_membership']['native_repeat_max_abs'],0)

    def test_rejects_real_changes_even_if_native_repeat_is_large(self):
        a=self.values();b=self.values();c=self.values()
        b['gaussian_membership']+=.001;c['gaussian_membership']+=.001
        self.assertFalse(repeat_calibrated_witness(a,b,c)['passed'])
        a=self.values();b=self.values();c=self.values();c['p_class']+=.001
        self.assertFalse(repeat_calibrated_witness(a,b,c)['passed'])

    def test_rejects_nonfinite_and_discrete_changes(self):
        a=self.values();b=self.values();c=self.values();c['region_mass'][0,0]=float('nan')
        self.assertFalse(repeat_calibrated_witness(a,b,c)['passed'])
        c=self.values();c['packed'][:2]=1
        self.assertFalse(repeat_calibrated_witness(a,b,c)['passed'])


if __name__=='__main__':unittest.main()
