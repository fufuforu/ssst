"""CPU unit contracts only; real model/GPU preflights remain mandatory."""
import json
from pathlib import Path
import sys
import torch
from torch import nn

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from tokengs.models.object_locus_understanding_probes_v1 import AlignedGeometryFusion
from scripts.train_vggt_understanding_probes_v1 import COHORT,expected_counts,slots
from scripts import object_locus_frozen_vggt_posefree_runtime as native

torch.set_num_threads(4);torch.manual_seed(123)
fusion=AlignedGeometryFusion()
memory=torch.randn(1,2738,1024)
scales=tuple(torch.randn(1,2,256,n,n) for n in (8,16,32))
out=fusion(scales,memory)
assert all(torch.equal(x,y) for x,y in zip(out,scales))
# Once the residual learns, view correspondence affects outputs, and each
# view responds only to its own geometry (not the other view's geometry).
with torch.no_grad():fusion.out.weight.normal_(std=.01)
changed=memory.clone();changed[:,1369:]+=torch.randn_like(changed[:,1369:])
a,b=fusion(scales,memory),fusion(scales,changed)
assert all(torch.equal(x[:,0],y[:,0]) and not torch.equal(x[:,1],y[:,1]) for x,y in zip(a,b))
loss=sum(x.square().mean() for x in a);loss.backward()
assert fusion.geo_proj.weight.grad is not None and fusion.geo_proj.weight.grad.norm()>0
assert fusion.out.weight.grad is not None and fusion.out.weight.grad.norm()>0
counts=expected_counts(500)
assert counts.shape==(4,8337) and counts.sum()==4000 and counts.sum(0).max()==1
assert set(i for step in range(500) for rank in range(4) for i in slots(step,rank))==set(json.loads(COHORT.read_text())['train_source_indices'])

class ScalarProbe(nn.Module):
    def __init__(self):
        super().__init__();self.understanding=nn.Linear(2,1)
    def step_loss(self,batch,step,understanding_weight):
        return {},dict(loss_recon=torch.tensor(.5),loss_understanding=self.understanding(batch).square().mean())

model=ScalarProbe();optimizer=native.build_optimizer(model)
before=model.understanding.weight.detach().clone()
row=native.train_microbatch_window(model,optimizer,[torch.ones(1,2)]*2,1,base_exposure=8,
    world_size=1,accumulation=2,total_updates=4172,understanding_only=True,gradient_diagnostic=True)
assert not torch.equal(before,model.understanding.weight)
assert all(x['understanding.']['reconstruction_norm']==0 and x['understanding.']['weighted_understanding_norm']>0
           for x in row['microbatch_gradient_diagnostics'])
assert row['preclip_norm']>0 and row['completed_updates']==2
print(json.dumps(dict(status='CPU_UNIT_CONTRACTS_PASSED',zero_initial_fusion=True,view_correspondence=True,
    geometry_gradient=True,unique4000_stream=True,understanding_only_native_optimizer=True,
    not_a_real_model_or_gpu_proof=True),indent=2))
