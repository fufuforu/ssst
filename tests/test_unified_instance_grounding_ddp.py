"""Two CPU Gloo ranks, actual new auxiliary loss, ragged queries, two steps."""
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
from tokengs.models.unified_instance_grounding import LocusGSInstanceGroundedUnified,rendered_instance_grounding_loss
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets


class Task(nn.Module):
    def __init__(self):
        super().__init__();self.model=LocusGSInstanceGroundedUnified(opt().evolve(img_size=(8,8)),config=replace(config(),checkpoint_blocks=True),omega=FakeOmega(),renderer=SpatialFeatureRenderer())
    def forward(self,rgb,rank):
        _,cloud,_=self.model.encode_context(rgb);ids=torch.arange(len(cloud['points']))%(rank+2)
        p=self.model.generate(rgb,superpoint_ids=ids)
        batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        batch['instance_label_all'][:,:,:,-4:]=2
        aux,_=rendered_instance_grounding_loss(p['rendered_instance_features'],build_context_instance_targets(batch))
        return p['gaussians'].square().mean()+p['thing_logits19'].square().mean()+p['gaussian_membership'].mean()+.01*aux


def worker(rank,path):
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+path)
    model=DDP(Task(),find_unused_parameters=True);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    rgb=torch.rand(1,2,3,256,256)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);loss=model(rgb,rank);loss.backward()
        assert torch.isfinite(loss)
        assert model.module.model.core.instance_head[1].weight.grad.norm()>0
        optimizer.step();witness=hashlib.sha256()
        for name,p in model.named_parameters():
            if p.requires_grad:witness.update(name.encode());witness.update(p.detach().numpy().tobytes())
        values=[None]*2;dist.all_gather_object(values,witness.hexdigest());assert len(set(values))==1
    dist.destroy_process_group()


class DistributedInstanceTests(unittest.TestCase):
    def test_auxiliary_ragged_checkpoint_and_parameter_sync(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.start_processes(worker,args=(str(Path(folder)/'rendezvous'),),nprocs=2,join=True,start_method='fork')


if __name__=='__main__':unittest.main()
