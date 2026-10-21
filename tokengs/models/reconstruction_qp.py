"""Persistent learned Q + scene-dependent reconstruction P, Cartesian expansion.

Ordering is superpoint-major: (P0,Q0), ..., (P0,QK-1), (P1,Q0), ... .
Only reconstruction anchors expand. No truncation, pooling, extra normalization,
learned geometry offset, or expansion of the 256-D understanding query set.
"""
import torch
from torch import nn

class LearnedReconstructionAnchorExpansion(nn.Module):
    def __init__(self,num_slots=20,dim=1024,init_std=.02):
        super().__init__()
        if isinstance(num_slots,bool) or not isinstance(num_slots,int) or num_slots<1:
            raise ValueError('num_slots must be a positive integer')
        if dim<1 or init_std<=0:raise ValueError('Positive feature dimension and Q initialization std required')
        self.num_slots=num_slots;self.dim=dim
        self.learned_q=nn.Parameter(torch.empty(num_slots,dim))
        nn.init.normal_(self.learned_q,std=init_std)
    def forward(self,p,centers,radii):
        if p.ndim!=3 or p.shape[-1]!=self.dim or p.shape[1]<1:
            raise ValueError('Expected nonempty [batch,superpoints,dim] reconstruction P')
        b,n,d=p.shape
        if centers.shape!=(b,n,3) or radii.shape!=(b,n):raise ValueError('Seed geometry does not match P')
        # Q is an nn.Parameter shared across every scene and optimizer update.
        tokens=(p[:,:,None,:]+self.learned_q[None,None,:,:]).reshape(b,n*self.num_slots,d)
        parent=torch.arange(n,device=p.device).repeat_interleave(self.num_slots)
        slots=torch.arange(self.num_slots,device=p.device).repeat(n)
        return dict(tokens=tokens,centers=centers.repeat_interleave(self.num_slots,dim=1),
            radii=radii.repeat_interleave(self.num_slots,dim=1),
            anchor_to_superpoint=parent,anchor_slot_ids=slots)
