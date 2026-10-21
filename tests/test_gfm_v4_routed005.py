import copy
import unittest
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.test_isolated_gradient_routing import Toy
from scripts.isolated_gradient_routing import accumulate,synchronize,verify_parameters_identical
from scripts.gfm_v4_isolated_policy import configure,validate_initial
from tests.test_gfm_v4_isolated_policy import IsolatedPolicyTests


def worker(rank,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    m=Toy();r=m.reconstruction;u=m.understanding.weight.squeeze()
    rec=(r+u+rank).square()
    under=.25*(10*r-3*u+rank).square()+(m.rank_local.square() if rank else 0.)
    accumulate(m,rec,under,reconstruction_understanding_scale=.05)
    synchronize(m,bucket_bytes=4)
    torch.testing.assert_close(r.grad,torch.tensor(8.125))
    torch.testing.assert_close(m.understanding.weight.grad.squeeze(),torch.tensor(-6.75))
    torch.testing.assert_close(m.rank_local.grad,torch.tensor(.15))
    assert m.globally_unused.grad is None
    optimizer=torch.optim.AdamW(m.parameters(),lr=.01);optimizer.step()
    verify_parameters_identical(m);assert m.globally_unused not in optimizer.state
    dist.destroy_process_group()


class RoutedTests(unittest.TestCase):
    def test_gradient_formula_and_understanding_unchanged_with_shared_checkpoint(self):
        from torch.utils.checkpoint import checkpoint
        m=Toy();r=m.reconstruction;u=m.understanding.weight.squeeze()
        shared=checkpoint(lambda x,y:x+y,r,u,use_reentrant=False)
        rec=shared.square();under=.25*(2*shared+r).square()
        expected_r=torch.autograd.grad(rec+.05*under,r,retain_graph=True)[0]
        expected_u=torch.autograd.grad(under,m.understanding.weight,retain_graph=True)[0]
        accumulate(m,rec,under,reconstruction_understanding_scale=.05)
        torch.testing.assert_close(r.grad,expected_r)
        torch.testing.assert_close(m.understanding.weight.grad,expected_u)
        self.assertIsNone(m.globally_unused.grad)

    def test_accumulation_and_reconstruction_only_under_loss_path(self):
        m=Toy()
        for x in (0.,1.):
            r=m.reconstruction;u=m.understanding.weight.squeeze()
            accumulate(m,(r+u+x).square(),.25*(10*r-3*u+x).square()+m.rank_local.square(),2,.05)
        self.assertAlmostEqual(float(m.reconstruction.grad),8.125)
        self.assertAlmostEqual(float(m.understanding.weight.grad),-6.75)
        self.assertAlmostEqual(float(m.rank_local.grad),.3,places=6)

    def test_untrained_policy_and_scale_identity(self):
        saved,target,blob=IsolatedPolicyTests().fixture()
        target=copy.deepcopy(saved);configure(target['config'],.05);target['recipe']=target['config']['recipe']
        validate_initial(saved,target,blob)
        target['config']['understanding_lr']*=2
        with self.assertRaises(ValueError):validate_initial(saved,target,blob)
        with self.assertRaises(ValueError):configure({},.1)

    def test_two_rank_scaled_gradients_and_unused_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(worker,args=(os.path.join(folder,'sync'),),nprocs=2,join=True)


if __name__=='__main__':unittest.main()
