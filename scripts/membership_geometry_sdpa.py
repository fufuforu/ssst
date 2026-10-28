"""Math SDPA for differentiable geometry biases, including bias-only gradients.

Avoid the installed fused CUDA backward's LSE strideH error. Guard inside SDPA
so activation-checkpoint recomputation uses the same backend. No-grad teacher,
export and unbiased attention retain their original paths.
"""
import functools
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
POLICY='math_sdpa_for_differentiable_additive_bias_v1'
COUNTERS=dict(math_bias_calls=0,bias_only_calls=0)

def install():
    original=F.scaled_dot_product_attention
    if getattr(original,'_membership_geometry_policy',None)==POLICY:return
    @functools.wraps(original)
    def guarded(query,key,value,*args,**kwargs):
        mask=kwargs.get('attn_mask',args[0] if args else None)
        if torch.is_grad_enabled() and torch.is_tensor(mask) and mask.requires_grad:
            COUNTERS['math_bias_calls']+=1
            if not any(x.requires_grad for x in (query,key,value)):COUNTERS['bias_only_calls']+=1
            with sdpa_kernel(SDPBackend.MATH):
                return original(query,key,value,*args,**kwargs)
        return original(query,key,value,*args,**kwargs)
    guarded._membership_geometry_policy=POLICY
    F.scaled_dot_product_attention=guarded

def evidence():return dict(policy=POLICY,**COUNTERS)
