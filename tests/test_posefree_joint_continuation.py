"""Contracts for the authorized two-arm continuation, including memory offload."""
import copy
import unittest
import torch
from torch import nn
from test_posefree_staged_adaptation import TinyStep, TinyAggregator, Camera, Depth, decode
from tokengs.models.frozen_vggt_posefree import FrozenVGGT
from scripts.object_locus_frozen_vggt_posefree_runtime import (
    continuation_configuration,configure_joint_continuation,gc_combine,
    build_optimizer,train_microbatch_window,rank_microbatch_indices,epoch_order,
    reshard_continuation_counts,continuation_rank_rng,capture_rank_rng,seed_everything)
from scripts.posefree_cpu_adamw import CPUOffloadAdamW


class JointContinuationContracts(unittest.TestCase):
    def test_eight_rank_configuration_changes_only_resources(self):
        for branch in ('frozen','joint_aa'):
            old=continuation_configuration(branch);new=continuation_configuration(branch,8)
            self.assertEqual({k for k in old if old[k]!=new[k]},{'node','world_size','accumulation'})
            self.assertEqual(new['world_size']*new['accumulation']*new['microbatch'],8)
            for update in (0,19,20,1042):
                self.assertEqual([i for r in range(4) for i in rank_microbatch_indices(6,update,r,world_size=4)],
                    [i for r in range(8) for i in rank_microbatch_indices(6,update,r,world_size=8)])

    def test_historical_exposures_reshard_without_recounting(self):
        import numpy as np
        for start,completed in ((0,6258),(6,20),(6,1100)):
            old=np.zeros((4,8337),dtype=np.int64)
            for update in range(completed):
                epoch,position=divmod(update,1043)
                for rank in range(4):
                    np.add.at(old[rank],list(rank_microbatch_indices(start+epoch,position,rank,world_size=4)),1)
            new=reshard_continuation_counts(old,start,completed,8)
            np.testing.assert_array_equal(new.sum(axis=0),old.sum(axis=0))
            self.assertEqual(int(new.sum()),completed*8)
            old[0,0]+=1
            with self.assertRaisesRegex(RuntimeError,'ownership'): reshard_continuation_counts(old,start,completed,8)

    def test_rng_reshard_preserves_parent_and_derives_repeatable_children(self):
        states=[]
        for seed in range(4):
            seed_everything(seed);states.append(capture_rank_rng())
        for rank in range(8):
            state=continuation_rank_rng(states,rank,8)
            repeated=continuation_rank_rng(states,rank,8)
            torch.testing.assert_close(state['torch_cpu'],repeated['torch_cpu'],rtol=0,atol=0)
            if rank%2==0:
                torch.testing.assert_close(state['torch_cpu'],states[rank//2]['torch_cpu'],rtol=0,atol=0)
                self.assertEqual(state['python'],states[rank//2]['python'])
            else: self.assertFalse(torch.equal(state['torch_cpu'],states[rank//2]['torch_cpu']))
        with self.assertRaisesRegex(RuntimeError,'unsupported'):continuation_rank_rng(states,0,2)

    def test_only_branch_and_aa_trainability_differ(self):
        a=continuation_configuration('frozen');b=continuation_configuration('joint_aa')
        self.assertEqual({k for k in a if a[k]!=b[k]},{'branch','aa_trainable'})
        self.assertEqual((a['total_updates'],a['total_exposures']),(2086,16688))
        for epoch in (6,7):
            rows=[i for u in range(1043) for rank in range(4)
                  for i in rank_microbatch_indices(epoch,u,rank,world_size=4)]
            self.assertEqual(rows,epoch_order(epoch).tolist())

    def test_joint_flags_and_common_precision(self):
        for branch in ('frozen','joint_aa'):
            v=nn.Module();v.aggregator=TinyAggregator();v.camera_head=Camera();v.depth_head=Depth()
            model=nn.Module();model.frozen_vggt=FrozenVGGT(model=v,pose_decoder=decode,test_only=True)
            for name in ('understanding','panoptic','vggt_memory_adapter','decoder','lpips_loss'):
                setattr(model,name,nn.Linear(2,2))
            model.lpips_loss.requires_grad_(False)
            optimizer=configure_joint_continuation(model,branch,torch.device('cpu'))
            self.assertFalse(model.reconstruction_adaptation)
            self.assertTrue(all(p.requires_grad for p in model.understanding.parameters()))
            self.assertTrue(all(p.requires_grad for p in model.panoptic.parameters()))
            self.assertFalse(any(p.requires_grad for p in model.lpips_loss.parameters()))
            self.assertTrue(all(p.dtype==torch.float32 for p in v.aggregator.parameters()))
            for name,p in model.frozen_vggt.named_parameters():
                self.assertEqual(p.requires_grad,branch=='joint_aa' and any(s in name for s in ('frame_blocks.','global_blocks.')))
            self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                             {id(p) for p in model.parameters() if p.requires_grad})

    def test_aa_receives_original_reconstruction_gc_rule(self):
        gr=torch.tensor([2.]);gu=torch.tensor([3.])
        names=['frozen_vggt.model.aggregator.frame_blocks.0.weight','decoder.weight','understanding.weight']
        values=gc_combine(names,[gr]*3,[gu]*3,vggt_reconstruction=True)
        torch.testing.assert_close(values[0],torch.tensor([2.03]))
        torch.testing.assert_close(values[1],values[0],rtol=0,atol=0)
        torch.testing.assert_close(values[2],torch.tensor([5.]))

    def test_saved_tensor_and_gradient_offload_preserves_native_step(self):
        torch.manual_seed(31);a=TinyStep();b=copy.deepcopy(a)
        native=build_optimizer(a);offload=CPUOffloadAdamW(build_optimizer(b).param_groups)
        batches=[(torch.tensor([[.3,.7]]),torch.tensor([[.2]])),
                 (torch.tensor([[-.1,.4]]),torch.tensor([[-.3]]))]
        for update in range(2):
            x=train_microbatch_window(a,native,batches,update,base_exposure=50064+8*update,
                accumulation=2,world_size=4,total_updates=2086)
            y=train_microbatch_window(b,offload,batches,update,base_exposure=50064+8*update,
                accumulation=2,world_size=4,total_updates=2086,cpu_saved_tensors=True,
                cpu_gradient_accumulation=True,vggt_reconstruction=True)
            self.assertEqual(x['understanding_weight'],1.)
            self.assertEqual(x['beta'],.1)
            self.assertEqual(x['preclip_norm'],y['preclip_norm'])
            for n,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[n],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
