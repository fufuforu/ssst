import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from scripts import membership_geometry_sdpa as guard

class GeometrySDPATests(unittest.TestCase):
    def setUp(self):
        self.original=F.scaled_dot_product_attention
        guard.COUNTERS.update(math_bias_calls=0,bias_only_calls=0)
        torch.manual_seed(31)
        self.q=torch.randn(1,3,7,8,dtype=torch.float64)
        self.k=torch.randn(1,3,13,8,dtype=torch.float64)
        self.v=torch.randn(1,3,13,8,dtype=torch.float64)
        self.bias=torch.randn(1,1,7,13,dtype=torch.float64,requires_grad=True)
        guard.install()
    def tearDown(self):F.scaled_dot_product_attention=self.original
    def manual(self,q,b):return ((q@self.k.transpose(-1,-2))/(q.shape[-1]**.5)+b).softmax(-1)@self.v

    def test_bias_only_values_and_gradients(self):
        actual=F.scaled_dot_product_attention(self.q,self.k,self.v,attn_mask=self.bias)
        expected=self.manual(self.q,self.bias)
        ga=torch.autograd.grad(actual.square().sum(),self.bias)[0]
        ge=torch.autograd.grad(expected.square().sum(),self.bias)[0]
        torch.testing.assert_close(actual,expected);torch.testing.assert_close(ga,ge)
        self.assertGreater(float(ga.abs().sum()),0);self.assertEqual(guard.evidence()['bias_only_calls'],1)

    def test_query_and_bias_gradients(self):
        q=self.q.requires_grad_()
        actual=F.scaled_dot_product_attention(q,self.k,self.v,self.bias)
        expected=self.manual(q,self.bias)
        ga=torch.autograd.grad(actual.square().sum(),(q,self.bias));ge=torch.autograd.grad(expected.square().sum(),(q,self.bias))
        for a,e in zip(ga,ge):torch.testing.assert_close(a,e)
        self.assertEqual(guard.evidence()['bias_only_calls'],0)

    def test_checkpoint_recomputation(self):
        def fn(b):return F.scaled_dot_product_attention(self.q,self.k,self.v,attn_mask=b).sin()
        actual=checkpoint(fn,self.bias,use_reentrant=False)
        ga=torch.autograd.grad(actual.square().sum(),self.bias)[0]
        ge=torch.autograd.grad(self.manual(self.q,self.bias).sin().square().sum(),self.bias)[0]
        torch.testing.assert_close(ga,ge)
        self.assertGreaterEqual(guard.evidence()['bias_only_calls'],2)

    def test_no_grad_and_fixed_bias_bypass_guard(self):
        with patch.object(guard,'sdpa_kernel',side_effect=AssertionError('Unexpected backend change')):
            with torch.no_grad():
                actual=F.scaled_dot_product_attention(self.q,self.k,self.v,attn_mask=self.bias)
                expected=self.original(self.q,self.k,self.v,attn_mask=self.bias)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            F.scaled_dot_product_attention(self.q.requires_grad_(),self.k,self.v,attn_mask=self.bias.detach()).sum().backward()
        self.assertEqual(guard.evidence()['math_bias_calls'],0)

    def test_install_is_idempotent_and_math_operator_used(self):
        first=F.scaled_dot_product_attention;guard.install();self.assertIs(first,F.scaled_dot_product_attention)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            F.scaled_dot_product_attention(self.q,self.k,self.v,attn_mask=self.bias).sum().backward()
        keys={e.key for e in prof.key_averages()}
        self.assertIn('aten::_scaled_dot_product_attention_math',keys)

if __name__=='__main__':unittest.main()
