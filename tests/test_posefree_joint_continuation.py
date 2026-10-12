"""Contracts for the authorized two-arm continuation, including memory offload."""
import copy
import unittest
import tempfile
import os
from pathlib import Path
from unittest import mock
import torch
from torch import nn
from test_posefree_staged_adaptation import TinyStep, TinyAggregator, Camera, Depth, decode
from tokengs.models.frozen_vggt_posefree import FrozenVGGT
from scripts.object_locus_frozen_vggt_posefree_runtime import (
    continuation_configuration,configure_joint_continuation,gc_combine,
    build_optimizer,train_microbatch_window,rank_microbatch_indices,epoch_order,
    reshard_continuation_counts,continuation_rank_rng,capture_rank_rng,seed_everything,
    continuation_clock,continuation_rank_indices,lr_multiplier,checkpoint_model_state,
    CONTINUATION_BATCH16_RECIPE,CONTINUATION_SOURCE_SHA,CONTINUATION_VGGT_SHA,EXPECTED_CHECKPOINT_SHA)
from scripts.posefree_cpu_adamw import CPUOffloadAdamW


def eight_rank_batch16_step(rank,directory):
    import torch.distributed as dist
    os.environ['GLOO_SOCKET_IFNAME']='lo';torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(directory)/'gloo16'),rank=rank,world_size=8)
    x=torch.arange(32,dtype=torch.float32).reshape(16,2)/8
    y=torch.linspace(-.4,.8,16).reshape(16,1)
    model=TinyStep();optimizer=CPUOffloadAdamW(build_optimizer(model).param_groups)
    for update in range(2):
        indices=(2*rank,2*rank+1) if update==0 else (rank,)
        row=train_microbatch_window(model,optimizer,[(x[i:i+1],y[i:i+1]) for i in indices],update,
            base_exposure=50064+16*update,world_size=8,accumulation=len(indices),total_updates=2086,
            lr_update=1 if update==0 else 2,cpu_saved_tensors=True,cpu_gradient_accumulation=True,vggt_reconstruction=True)
        torch.save(dict(model=model.state_dict(),row=row,optimizer=optimizer.state_dict()),
                   Path(directory)/f'rank{rank}_step{update}.pt')
    dist.destroy_process_group()


class JointContinuationContracts(unittest.TestCase):
    def test_batch16_changes_batch_schedule_but_preserves_source_and_other_math(self):
        a=continuation_configuration('frozen',8,16);b=continuation_configuration('joint_aa',8,16)
        self.assertEqual({k for k in a if a[k]!=b[k]},{'branch','aa_trainable'})
        self.assertEqual((a['recipe'],a['global_batch'],a['accumulation'],a['updates_per_epoch'],a['total_updates'],a['total_exposures']),
            (CONTINUATION_BATCH16_RECIPE,16,2,522,1044,16688))
        old=continuation_configuration('frozen',8)
        for key in ('lr_peak','clip_global_norm','gc_alpha','weight_decay','precision','source_checkpoint_sha256','source_adapted_vggt_sha256','source_completed_updates','source_completed_exposures'):
            self.assertEqual(a[key],old[key])
        with self.assertRaises(ValueError):continuation_configuration('frozen',4,16)

    def test_batch16_keeps_every_original_window_and_handles_tail_and_lr_clock(self):
        for epoch in (6,7):
            flat=[i for u in range(522) for rank in range(8)
                  for i in continuation_rank_indices(epoch,u,rank,world_size=8,global_batch=16)]
            self.assertEqual(flat,epoch_order(epoch).tolist())
            self.assertEqual(len(continuation_rank_indices(epoch,520,0,global_batch=16)),2)
            self.assertEqual(len(continuation_rank_indices(epoch,521,0,global_batch=16)),1)
        self.assertEqual(continuation_clock(521,16),(0,521,8336))
        self.assertEqual(continuation_clock(522,16),(1,0,8344))
        self.assertEqual(continuation_clock(523,16),(1,1,8360))
        self.assertEqual(continuation_clock(1044,16),(2,0,16688))
        self.assertEqual(lr_multiplier(continuation_clock(100,16)[2]//8-1,2086),1.)
        self.assertEqual(lr_multiplier(continuation_clock(1044,16)[2]//8-1,2086),.1)
        self.assertIn(6923,[i for u in range(2) for r in range(8) for i in continuation_rank_indices(0,u,r,global_batch=16)])

    def test_eight_rank_accumulation_and_tail_match_native_global_mean(self):
        x=torch.arange(32,dtype=torch.float32).reshape(16,2)/8
        y=torch.linspace(-.4,.8,16).reshape(16,1)
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(eight_rank_batch16_step,args=(directory,),nprocs=8,join=True,start_method='fork')
            model=TinyStep();optimizer=build_optimizer(model)
            for update,size in enumerate((16,8)):
                reference=train_microbatch_window(model,optimizer,[(x[:size],y[:size])],update,base_exposure=50064+16*update,
                    world_size=1,accumulation=1,total_updates=2086,lr_update=1 if update==0 else 2,vggt_reconstruction=True)
                for rank in range(8):
                    actual=torch.load(Path(directory)/f'rank{rank}_step{update}.pt',weights_only=True)
                    self.assertAlmostEqual(actual['row']['preclip_norm'],reference['preclip_norm'],places=5)
                    self.assertEqual(actual['row']['lr_multiplier'],reference['lr_multiplier'])
                    for name,value in model.state_dict().items():torch.testing.assert_close(actual['model'][name],value,rtol=1e-6,atol=1e-7)
                    for state in actual['optimizer']['optimizer']['state'].values():self.assertEqual(float(state['step']),update+1)

    def test_batch16_evaluation_accepts_actual_endpoint_and_rejects_old_update_clock(self):
        from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _restore_checkpoint
        v=nn.Module();v.aggregator=TinyAggregator();v.camera_head=Camera();v.depth_head=Depth()
        model=TinyStep();model.frozen_vggt=FrozenVGGT(model=v,pose_decoder=decode,test_only=True)
        keys=('repository','model_id','revision','files','loaded_subtrees','loaded_source_key_count','explicitly_excluded_source_key_count','loaded_key_sha256')
        revision='f'*40;manifest='d'*64
        payload=dict(recipe=CONTINUATION_BATCH16_RECIPE,branch='frozen',config=continuation_configuration('frozen',8,16),
            world_size=8,rank_rng=[None]*8,vggt_revision=revision,manifest_sha256=manifest,source_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA,
            vggt_artifact_identity={k:model.frozen_vggt.source_identity.get(k) for k in keys},continuation_source_sha256=CONTINUATION_SOURCE_SHA,
            phase='joint_continuation',epoch=8,completed_updates=7302,completed_exposures=66752,
            continuation_completed_updates=1044,continuation_completed_exposures=16688,model=checkpoint_model_state(model),
            adapted_vggt_identity=dict(path='/mock/adapted.pt',sha256=CONTINUATION_VGGT_SHA))
        with mock.patch('torch.load',return_value=payload),mock.patch('scripts.object_locus_frozen_vggt_posefree_runtime.restore_adapted_vggt') as restore:
            actual=_restore_checkpoint(model,Path('/mock/checkpoint.pt'),revision,manifest)
            self.assertEqual(actual['completed_updates'],7302);restore.assert_called_once()
            self.assertEqual(model.understanding_step,66752)
            wrong=dict(payload,completed_updates=8344)
            with mock.patch('torch.load',return_value=wrong),self.assertRaisesRegex(RuntimeError,'endpoint clock'):
                _restore_checkpoint(model,Path('/mock/checkpoint.pt'),revision,manifest)

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
