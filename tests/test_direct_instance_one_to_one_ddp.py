"""Actual Gloo updates with N=K on one rank and N>K on another."""
import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tests.test_unified_query_v1 import config,opt,FakeOmega
from tests.test_unified_instance_grounding import SpatialFeatureRenderer
from tests.test_direct_instance_attributes import labels
from tokengs.models.direct_instance_attributes import LocusGSOneToOneDirectInstanceAttributes
from tokengs.models.unified_query_loss import unified_instance_loss
from tokengs.models.unified_instance_grounding import rendered_instance_grounding_loss


class Task(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=LocusGSOneToOneDirectInstanceAttributes(opt().evolve(img_size=(8,8)),
            config=replace(config(),matching_mode='global_hungarian',support_matching=False,checkpoint_blocks=True),
            omega=FakeOmega(),renderer=SpatialFeatureRenderer())

    def forward(self,rgb,rank):
        _,cloud,_=self.model.encode_context(rgb)
        p=self.model.generate(rgb,superpoint_ids=torch.arange(len(cloud['points']))%(rank+2))
        under,metrics=unified_instance_loss(p,labels(),self.model.config)
        assert metrics['matched_gt_count']==metrics['matched_query_count']==2 and metrics['unmatched_GT_count']==0
        aux,_=rendered_instance_grounding_loss(p['rendered_instance_features'],p['final_targets'])
        return p['gaussians'].square().mean()+.05*(under+.01*aux)


def worker(rank,path):
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+path)
    model=DDP(Task(),find_unused_parameters=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True);model(torch.rand(1,2,3,256,256),rank).backward()
        for p in model.parameters():
            if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all()
        for head in ('instance_attribute_head','instance_prototype_head','stuff_attribute_head'):
            assert sum(float(p.grad.abs().sum()) for p in getattr(model.module.model.core,head).parameters())>0
        optimizer.step()
        witness=hashlib.sha256()
        for name,p in model.named_parameters():
            if p.requires_grad:witness.update(name.encode());witness.update(p.detach().numpy().tobytes())
        values=[None]*2;dist.all_gather_object(values,witness.hexdigest());assert len(set(values))==1
    dist.destroy_process_group()


def test_two_rank_global_assignment_two_updates_all_parameters_used_and_synchronized():
    with tempfile.TemporaryDirectory() as tmp:
        mp.start_processes(worker,args=(str(Path(tmp)/'rendezvous'),),nprocs=2,join=True,start_method='fork')
