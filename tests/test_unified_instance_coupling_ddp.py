"""Actual two-rank Gloo synchronization for all three new coupling controls."""
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
from tokengs.models.superpoint_structural_variants import build_structural_model
from tokengs.models.unified_instance_grounding import rendered_instance_grounding_loss
from tokengs.models.unified_instance_coupling import boundary_rgb_loss
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets


class Task(nn.Module):
    def __init__(self,architecture):
        super().__init__();self.model=build_structural_model(architecture,opt().evolve(img_size=(8,8)),
            replace(config(),checkpoint_blocks=True),omega=FakeOmega(),renderer=SpatialFeatureRenderer())
    def forward(self,rgb,rank):
        _,cloud,_=self.model.encode_context(rgb);ids=torch.arange(len(cloud['points']))%(rank+2)
        p=self.model.generate(rgb,superpoint_ids=ids)
        batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        batch['instance_label_all'][:,:,:,-4:]=2
        aux,_=rendered_instance_grounding_loss(p['rendered_instance_features'],build_context_instance_targets(batch))
        loss=p['gaussians'].square().mean()+p['thing_logits19'].square().mean()+p['gaussian_membership'].mean()+.01*aux
        if self.model.coupling_recipe.boundary:
            loss=loss+.02*boundary_rgb_loss(p['render']['images_pred'],rgb[:,:,:,:8,:8],p['rendered_instance_features'])[0]
        return loss


def worker(rank,path,architecture):
    torch.set_num_threads(1);torch.manual_seed(42)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+path)
    model=DDP(Task(architecture),find_unused_parameters=True)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    rgb=torch.rand(1,2,3,256,256)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);loss=model(rgb,rank);loss.backward()
        assert torch.isfinite(loss)
        assert model.module.model.core.instance_head[1].weight.grad.norm()>0
        if model.module.model.coupling_recipe.lift:assert model.module.model.core.lift_residual[1].weight.grad.norm()>0
        optimizer.step();witness=hashlib.sha256()
        for name,p in model.named_parameters():
            if p.requires_grad:witness.update(name.encode());witness.update(p.detach().numpy().tobytes())
        values=[None]*2;dist.all_gather_object(values,witness.hexdigest());assert len(set(values))==1
    dist.destroy_process_group()


class DistributedCouplingTests(unittest.TestCase):
    def test_three_variants_ragged_checkpointed_two_updates(self):
        for architecture in ('unified_instance_lift','unified_instance_boundary','unified_instance_bidirectional'):
            with self.subTest(architecture=architecture),tempfile.TemporaryDirectory() as folder:
                mp.start_processes(worker,args=(str(Path(folder)/'rendezvous'),architecture),nprocs=2,join=True,start_method='fork')


if __name__=='__main__':unittest.main()
