import copy
import os
import tempfile
import unittest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from scripts.isolated_gradient_routing import accumulate,synchronize,verify_parameters_identical


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.reconstruction=nn.Parameter(torch.tensor(1.))
        self.understanding=nn.Linear(1,1,bias=False)
        with torch.no_grad():self.understanding.weight.fill_(2.)
        self.rank_local=nn.Parameter(torch.tensor(3.));self.globally_unused=nn.Parameter(torch.tensor(4.))


def distributed_worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    model=Toy();r=model.reconstruction;u=model.understanding.weight.squeeze()
    # A reconstruction parameter is locally unused on rank0, used on rank1.
    rec=(r+u+rank).square()+(model.rank_local.square() if rank else 0.)
    under=.25*(10*r-3*u+rank).square()
    accumulate(model,rec,under);synchronize(model,bucket_bytes=4)
    torch.testing.assert_close(r.grad,torch.tensor(7.))
    torch.testing.assert_close(model.understanding.weight.grad.squeeze(),torch.tensor(-6.75))
    torch.testing.assert_close(model.rank_local.grad,torch.tensor(3.))
    assert model.globally_unused.grad is None
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    optimizer.step();verify_parameters_identical(model)
    assert model.globally_unused not in optimizer.state
    dist.destroy_process_group()


class IsolatedGradientTests(unittest.TestCase):
    def test_shared_nonreentrant_checkpoint_supports_both_loss_backward_paths(self):
        from torch.utils.checkpoint import checkpoint
        model=Toy();r=model.reconstruction;u=model.understanding.weight.squeeze()
        shared=checkpoint(lambda x,y:x+y,r,u,use_reentrant=False)
        accumulate(model,shared.square(),.25*(2*shared+r).square())
        self.assertAlmostEqual(float(r.grad),6.)
        self.assertAlmostEqual(float(model.understanding.weight.grad),7.)

    def test_only_assigned_loss_updates_each_parameter_group(self):
        model=Toy();r=model.reconstruction;u=model.understanding.weight.squeeze()
        accumulate(model,(r+u).square(),.25*(10*r-3*u).square())
        self.assertAlmostEqual(float(r.grad),6.)
        self.assertAlmostEqual(float(model.understanding.weight.grad),-6.)
        self.assertIsNone(model.globally_unused.grad)
        model.zero_grad(set_to_none=True)
        accumulate(model,(r+u).square(),100*(10*r-3*u).square())
        self.assertAlmostEqual(float(r.grad),6.) # forbidden under gradient stays excluded
        self.assertAlmostEqual(float(model.understanding.weight.grad),-2400.)

    def test_microbatch_accumulation_matches_average_and_no_unused_decay(self):
        model=Toy()
        for x in [0.,1.]:
            r=model.reconstruction;u=model.understanding.weight.squeeze()
            accumulate(model,(r+u+x).square(),.25*(10*r-3*u+x).square(),2)
        self.assertAlmostEqual(float(model.reconstruction.grad),7.)
        self.assertAlmostEqual(float(model.understanding.weight.grad),-6.75)
        original=model.globally_unused.detach().clone()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1);optimizer.step()
        self.assertTrue(torch.equal(original,model.globally_unused))
        self.assertNotIn(model.globally_unused,optimizer.state)

    def test_real_two_rank_reduction_handles_locally_and_globally_unused(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_worker,args=(os.path.join(folder,'rendezvous'),),nprocs=2,join=True)

    def test_task_returns_live_components_without_changing_forward_loss(self):
        from unittest.mock import patch
        from scripts.train_dual_stream_v1 import Task
        class Model(Toy):
            def step_loss(self,batch,*,step,understanding_weight):
                r=self.reconstruction;u=self.understanding.weight.squeeze()
                rec=(r+u).square();under=(10*r-3*u).square()
                return {'prediction':None},dict(loss_recon=rec,loss_understanding=under,loss=rec+understanding_weight*under)
        model=Model()
        with patch('scripts.train_dual_stream_v1.scalar_metrics',return_value={}):
            normal,_=Task(model)({},0,False,.25)
            losses,_=Task(model,isolated=True)({},0,False,.25)
        self.assertTrue(all(x.requires_grad for x in losses))
        torch.testing.assert_close(normal,losses[2]);torch.testing.assert_close(losses[0]+losses[1],normal)


if __name__=='__main__':unittest.main()
