"""Verbatim actual sem256 execution class, captured 2026-10-20; no model edits."""
import torch
from torch import nn
import torch.nn.functional as F

class SemanticMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,256)) for _ in range(4)])
        self.layer_logits=nn.Parameter(torch.zeros(4))
        # High-resolution local RGB residual; randomly initialized, no MASt3R.
        self.local=nn.Sequential(nn.Conv2d(3,64,3,padding=1),nn.GELU(),nn.Conv2d(64,256,3,padding=1))
        self.refine=nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.GELU(),nn.Conv2d(256,256,1))
        self.mask_embedder=nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Linear(256,256))
    def forward(self,layers,rgb):
        b,v=rgb.shape[:2]
        patches=sum(w*proj(x.float()) for w,proj,x in zip(self.layer_logits.softmax(0),self.projections,layers))
        grid=patches.reshape(b*v,32,32,256).permute(0,3,1,2)
        grid=F.interpolate(grid,size=(256,256),mode='bilinear',align_corners=False)
        local=self.local(F.interpolate(rgb.flatten(0,1),size=(256,256),mode='bilinear',align_corners=False))
        return self.refine(grid+local).reshape(b,v,256,256,256)
