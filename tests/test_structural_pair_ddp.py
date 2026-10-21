"""Two real Gloo ranks: dynamic legacy readout and checkpointed direct fields."""
import hashlib,tempfile,unittest
from pathlib import Path
from dataclasses import replace
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
from tokengs.models.superpoint_structural_variants import build_structural_model


class LossTask(nn.Module):
    def __init__(self,kind):
        super().__init__();cfg=replace(config(),superpoint_source='esam_sam',matching_mode='source_many_to_one',checkpoint_blocks=True)
        options=opt()
        if kind=='dual_superpoint':cfg=replace(cfg,dim=256,heads=8,depth=12);options=options.evolve(dec_patch_size=8,img_size=(8,8))
        self.model=build_structural_model(kind,options,cfg,omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer())
    def forward(self,rgb,rank):
        _,cloud,_=self.model.encode_context(rgb)
        n=2+rank;ids=torch.arange(len(cloud['points']))%n
        out=self.model.generate(rgb,superpoint_ids=ids)
        return out['gaussians'].square().mean()+out['thing_logits19'].square().mean()+out['gaussian_membership'].mean()


def worker(rank,path,kind):
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+path)
    model=DDP(LossTask(kind),find_unused_parameters=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    rgb=torch.rand(1,2,3,256,256)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);loss=model(rgb,rank);loss.backward();assert torch.isfinite(loss)
        optimizer.step();h=hashlib.sha256()
        for p in model.parameters():
            if p.requires_grad:h.update(p.detach().numpy().tobytes())
        values=[None]*2;dist.all_gather_object(values,h.hexdigest());assert len(set(values))==1
    dist.destroy_process_group()


class StructuralDistributedTests(unittest.TestCase):
    def test_legacy_dual(self):self.run_variant('dual_superpoint')
    def test_unified_direct(self):self.run_variant('unified_direct')
    def run_variant(self,kind):
        with tempfile.TemporaryDirectory() as directory:
            mp.start_processes(worker,args=(str(Path(directory)/'rendezvous'),kind),nprocs=2,join=True,start_method='fork')


if __name__=='__main__':unittest.main()
