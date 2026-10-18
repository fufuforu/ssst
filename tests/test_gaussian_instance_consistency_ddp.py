"""Two real Gloo processes: ragged queries, skipped terms, checkpoint replay."""
import hashlib,tempfile,unittest
from pathlib import Path
from dataclasses import replace
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tests.test_unified_query_v1 import config,opt,FakeOmega
from tests.test_unified_instance_grounding import SpatialFeatureRenderer
from tokengs.models.gaussian_instance_consistency import LocusGSGaussianInstanceConsistency,direct_gaussian_consistency_loss
from tokengs.models.unified_instance_grounding import rendered_instance_grounding_loss
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets


class Task(nn.Module):
    def __init__(self):
        super().__init__();self.model=LocusGSGaussianInstanceConsistency(opt().evolve(img_size=(8,8)),
            config=replace(config(),checkpoint_blocks=True),omega=FakeOmega(),renderer=SpatialFeatureRenderer())

    def forward(self,rgb,rank,step):
        _,cloud,_=self.model.encode_context(rgb)
        p=self.model.generate(rgb,superpoint_ids=torch.arange(len(cloud['points']))%(rank+2))
        batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        batch['instance_label_all'][...,-4:]=2
        rendered,_=rendered_instance_grounding_loss(p['rendered_instance_features'],build_context_instance_targets(batch))
        parent=p['gaussian_parent_query'];labels=(parent%2)[None]
        if rank==0 and step==0:labels=torch.full_like(labels,-1)
        if step==1:labels=torch.zeros_like(labels) # Positive-only; no negative GT.
        assigned=dict(labels=labels,purity=torch.ones_like(labels,dtype=torch.float),mass=torch.ones_like(labels,dtype=torch.float))
        direct,_=direct_gaussian_consistency_loss(self.model.core.direct_instance_features(p),p['gaussians'][...,:3],parent,
                                                assigned,p['cloud']['scale']*100)
        return p['gaussians'].square().mean()+p['thing_logits19'].square().mean()+p['gaussian_membership'].mean()+.01*(rendered+direct)


def worker(rank,path):
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+path)
    model=DDP(Task(),find_unused_parameters=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    rgb=torch.rand(1,2,3,256,256)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);loss=model(rgb,rank,step);loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.module.model.core.instance_head[1].weight.grad.norm()>0
        optimizer.step();witness=hashlib.sha256()
        for name,p in model.named_parameters():
            if p.requires_grad:witness.update(name.encode());witness.update(p.detach().numpy().tobytes())
        values=[None]*2;dist.all_gather_object(values,witness.hexdigest());assert len(set(values))==1
    dist.destroy_process_group()


class DistributedDirectConsistencyTests(unittest.TestCase):
    def test_two_updates_sync_with_ragged_and_empty_direct_supervision(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.start_processes(worker,args=(str(Path(folder)/'rendezvous'),),nprocs=2,join=True,start_method='fork')


if __name__=='__main__':unittest.main()
