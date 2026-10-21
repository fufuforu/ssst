"""Real two-rank Gloo verification of ragged queries and checkpointed decoder."""
import hashlib,tempfile,unittest
from pathlib import Path
from dataclasses import replace
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tests.test_unified_query_v1 import config,opt,cloud
from tokengs.models.unified_query_core import UnifiedQueryCore


def worker(rank,path):
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+path)
    model=DDP(UnifiedQueryCore(opt(),replace(config(),checkpoint_blocks=True)),find_unused_parameters=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        ids=torch.tensor([0,0,1,1]) if rank==0 else torch.tensor([0,1,2,2])
        out=model(cloud(),torch.randn(1,5,16),ids)
        loss=out['gaussians'].square().mean()+out['thing_logits19'].square().mean()+out['gaussian_membership'].mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert model.module.initializer.point_encoder.layers[0][0].weight.grad.norm()>0
        optimizer.step()
        digest=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        values=[None]*2;dist.all_gather_object(values,digest);assert values[0]==values[1]
    dist.destroy_process_group()


class DistributedUnifiedTest(unittest.TestCase):
    def test_ragged_queries_checkpoint_and_parameter_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.start_processes(worker,args=(str(Path(directory)/'rendezvous'),),nprocs=2,join=True,start_method='fork')


if __name__=='__main__':unittest.main()
