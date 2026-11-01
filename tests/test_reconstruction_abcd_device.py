"""Exact migration audits normalize devices and reject value/dtype corruption.

CPU unit checks do not claim real CUDA execution; actual CUDA migration is
validated in each eight-rank smoke before any formal work.
"""
import unittest
import torch
from scripts.reconstruction_abcd_policy import exact_equal,expanded

class DeviceCopy:
    """Exercise the normalization boundary without a CUDA driver on login."""
    def __init__(self,t):self.value=t;self.shape=t.shape;self.dtype=t.dtype;self.copied=False
    def detach(self):return self
    def cpu(self):self.copied=True;return self.value

class DeviceAuditTests(unittest.TestCase):
    def test_both_operands_normalized_before_exact_comparison(self):
        a=DeviceCopy(torch.tensor([1.,2.]));b=DeviceCopy(torch.tensor([1.,2.]))
        self.assertTrue(exact_equal(a,b));self.assertTrue(a.copied and b.copied)
        b.value[1]+=1
        self.assertFalse(exact_equal(a,b))

    def test_rejects_silent_dtype_shape_or_value_changes(self):
        a=torch.tensor([1.,2.],requires_grad=True)
        self.assertTrue(exact_equal(a,a.detach().clone()))
        self.assertFalse(exact_equal(a,a.double()))
        self.assertFalse(exact_equal(a,a.reshape(1,2)))
        self.assertFalse(exact_equal(a,torch.tensor([1.,2.00001])))

    def test_expanded_weights_and_adam_prefix_are_exact_after_normalization(self):
        old=torch.randn(1024,1024)
        weights=expanded(old,'parameter');moment=expanded(old,'moment')
        self.assertTrue(exact_equal(DeviceCopy(weights[:1024]),old))
        self.assertTrue(exact_equal(DeviceCopy(moment[:1024]),old))
        self.assertEqual(int(moment[1024:].count_nonzero()),0)
        moment[0,0]+=1
        self.assertFalse(exact_equal(DeviceCopy(moment[:1024]),old))

if __name__=='__main__':unittest.main()
