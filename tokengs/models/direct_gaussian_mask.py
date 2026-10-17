"""Query-generated nonlinear 3D Gaussian mask fields; no two-tower similarity."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DirectGaussianMaskField(nn.Module):
    def __init__(self,dim,hidden=16,chunk=256):
        super().__init__();self.hidden=hidden;self.chunk=chunk
        self.parameters_from_query=nn.Linear(dim,16*hidden+hidden+hidden+1)

    def _evaluate(self,parameters,gaussians,centers,scale):
        h=self.hidden;w,b,v,offset=torch.split(parameters,[16*h,h,h,1],-1);w=w.reshape(*w.shape[:2],16,h)
        delta=(gaussians[:,None,:,:3]-centers[:,:,None])/scale
        geom=gaussians[:,None,:,4:7].clamp_min(1e-8).log().expand(-1,len(centers[0]),-1,-1)
        extra=gaussians[:,None,:,[3,11,12,13]].expand(-1,len(centers[0]),-1,-1)
        inputs=torch.cat((delta,torch.sin(delta*torch.pi),torch.cos(delta*torch.pi),geom,extra),-1)
        hidden=F.gelu(torch.einsum('bqgf,bqfh->bqgh',inputs,w)+b[:,:,None])
        return ((hidden*v[:,:,None]).sum(-1)+offset).transpose(1,2)

    def forward(self,query,gaussians,centers,scale):
        parameters=self.parameters_from_query(query);pieces=[]
        for part in gaussians.split(self.chunk,dim=1):
            pieces.append(checkpoint(self._evaluate,parameters,part,centers,scale,use_reentrant=False) if self.training and torch.is_grad_enabled() else self._evaluate(parameters,part,centers,scale))
        return torch.cat(pieces,1)
