"""Numerical optimizer equivalence and true VGGT gradient/freeze boundaries."""
import unittest
import tempfile
from pathlib import Path
import torch
from torch import nn
from scripts.posefree_cpu_adamw import CPUOffloadAdamW
from tokengs.models.frozen_vggt_posefree import FrozenVGGT


class TinyStep(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ('decoder','understanding','vggt_memory_adapter'):
            module=nn.Linear(2,1,bias=False)
            with torch.no_grad(): module.weight.copy_(torch.tensor([[.2,-.3]]))
            setattr(self,name,module)
    def step_loss(self,batch,*,step,understanding_weight):
        x,y=batch
        value=self.decoder(x)+self.understanding(x)+self.vggt_memory_adapter(x)
        rec=(value-y).square().mean();under=(value+2*y).square().mean()
        return {},dict(loss_recon=rec,loss_understanding=under)


def four_rank_cpu_step(rank,directory):
    import os
    import torch.distributed as dist
    from scripts.object_locus_frozen_vggt_posefree_runtime import build_optimizer,train_microbatch_window
    os.environ['GLOO_SOCKET_IFNAME']='lo';torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(Path(directory)/'gloo'),rank=rank,world_size=4)
    x=torch.arange(16,dtype=torch.float32).reshape(8,2)/4
    y=torch.linspace(-.4,.8,8).reshape(8,1)
    for adapting in (False,True):
        model=TinyStep();optimizer=build_optimizer(model)
        batches=[(x[i:i+1],y[i:i+1]) for i in (rank*2,rank*2+1)]
        row=train_microbatch_window(model,optimizer,batches,1,base_exposure=100,
            world_size=4,accumulation=2,reconstruction_adaptation=adapting)
        torch.save(dict(model=model.state_dict(),row=row,optimizer=optimizer.state_dict()),
                   Path(directory)/f'rank{rank}_{adapting}.pt')
    dist.destroy_process_group()


class TinyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_blocks=nn.ModuleList([nn.Linear(1,1,bias=False)])
        self.global_blocks=nn.ModuleList([nn.Linear(1,1,bias=False)])
        self.patch_embed=nn.Linear(1,1,bias=False)
    def forward(self, images):
        value=self.global_blocks[0](self.frame_blocks[0](images.mean().reshape(1,1)))
        tokens=value.reshape(1,1,1,1).expand(1,2,1370,2048)
        layers=[None]*24
        for index in (4,11,17,23): layers[index]=tokens
        return layers,1


class Camera(nn.Module):
    def forward(self, tokens): return [tokens[23][:,:,0,:9]]


class Depth(nn.Module):
    def forward(self,tokens,images,patch_start_idx):
        value=torch.ones((1,2,1,518,518),device=images.device)
        return value,value


def decode(encoding,image_size_hw):
    ext=torch.eye(4)[:3].reshape(1,1,3,4).repeat(1,2,1,1)
    k=torch.eye(3).reshape(1,1,3,3).repeat(1,2,1,1)
    return ext,k


class Contracts(unittest.TestCase):
    def test_four_rank_accumulation_matches_global_eight_sample_update(self):
        from scripts.object_locus_frozen_vggt_posefree_runtime import build_optimizer,train_microbatch_window
        x=torch.arange(16,dtype=torch.float32).reshape(8,2)/4
        y=torch.linspace(-.4,.8,8).reshape(8,1)
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(four_rank_cpu_step,args=(directory,),
                nprocs=4,join=True,start_method='fork')
            for adapting in (False,True):
                model=TinyStep();optimizer=build_optimizer(model)
                reference=train_microbatch_window(model,optimizer,[(x,y)],1,base_exposure=100,
                    world_size=1,reconstruction_adaptation=adapting)
                for rank in range(4):
                    actual=torch.load(Path(directory)/f'rank{rank}_{adapting}.pt',weights_only=True)
                    self.assertEqual(actual['row']['completed_exposures'],16)
                    self.assertAlmostEqual(actual['row']['preclip_norm'],reference['preclip_norm'],places=5)
                    for name,value in model.state_dict().items():
                        torch.testing.assert_close(actual['model'][name],value,rtol=1e-6,atol=1e-7)
                    for state in actual['optimizer']['state'].values(): self.assertEqual(state['step'].item(),1)

    def test_four_rank_sampler_preserves_original_update_windows_and_budget(self):
        from scripts.object_locus_frozen_vggt_posefree_runtime import (
            rank_microbatch_indices,UPDATES_PER_EPOCH,epoch_order,staged_training_configuration)
        for epoch in range(6):
            for update in (0,1,UPDATES_PER_EPOCH//2,UPDATES_PER_EPOCH-1):
                old=[i for rank in range(8) for i in rank_microbatch_indices(epoch,update,rank)]
                new=[i for rank in range(4) for i in rank_microbatch_indices(epoch,update,rank,world_size=4)]
                self.assertEqual(old,new)
        full=[i for update in range(UPDATES_PER_EPOCH) for rank in range(4)
              for i in rank_microbatch_indices(0,update,rank,world_size=4)]
        self.assertEqual(full,epoch_order(0).tolist())
        old=staged_training_configuration(8);new=staged_training_configuration(4)
        changed={key for key in old if old[key]!=new[key]}
        self.assertEqual(changed,{'world_size','accumulation','node','gpu_model'})
        self.assertEqual(new['world_size']*new['microbatch']*new['accumulation'],8)
        self.assertEqual((new['total_updates'],new['total_exposures']),(6258,50064))

    def test_boundary_resume_preserves_rng_with_fresh_optimizer(self):
        from scripts.object_locus_frozen_vggt_posefree_runtime import (
            restore_staged_phase_state,capture_rank_rng,seed_everything)
        class FreshOptimizer:
            def load_state_dict(self, state): raise AssertionError('previous-phase optimizer restored')
        seed_everything(312);torch.rand(3)
        saved=capture_rank_rng();expected=torch.rand(5)
        seed_everything(999)
        restore_staged_phase_state(None,FreshOptimizer(),
            {'phase':'reconstruction_adaptation','rank_rng':[saved]},'frozen_joint',0)
        torch.testing.assert_close(torch.rand(5),expected,rtol=0,atol=0)

    def test_phase_transition_restores_original_trainable_set(self):
        from scripts.object_locus_frozen_vggt_posefree_runtime import configure_staged_phase
        vggt=nn.Module();vggt.aggregator=TinyAggregator();vggt.camera_head=Camera();vggt.depth_head=Depth()
        model=nn.Module();model.frozen_vggt=FrozenVGGT(model=vggt,pose_decoder=decode,test_only=True,preserve_fp32_aggregator=True)
        for name in ('understanding','panoptic','vggt_memory_adapter','decoder','lpips_loss'):
            setattr(model,name,nn.Linear(2,2))
        model.lpips_loss.requires_grad_(False)
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        configure_staged_phase(model,'reconstruction_adaptation',torch.device('cpu'))
        self.assertFalse(any(p.requires_grad for p in model.understanding.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.frozen_vggt.parameters()))
        optimizer=configure_staged_phase(model,'frozen_joint',torch.device('cpu'))
        self.assertEqual({n for n,p in model.named_parameters() if p.requires_grad},expected)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                         {id(p) for p in model.parameters() if p.requires_grad})

    def test_saved_adapted_asset_restores_and_rejects_wrong_hash(self):
        from scripts.object_locus_frozen_vggt_posefree_runtime import restore_adapted_vggt,STAGED_RECIPE,sha256,UPDATES_PER_EPOCH
        model=nn.Module();model.aggregator=TinyAggregator();model.camera_head=Camera();model.depth_head=Depth()
        wrapper=FrozenVGGT(model=model,pose_decoder=decode,test_only=True,preserve_fp32_aggregator=True,
                           source_identity={'revision':'controlled-test-revision'})
        host=nn.Module();host.frozen_vggt=wrapper
        reference={k:v.clone() for k,v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'adapted.pt'
            torch.save(dict(recipe=STAGED_RECIPE,adaptation_updates=2*UPDATES_PER_EPOCH,
                            base_hf_revision='controlled-test-revision',model=reference),path)
            with torch.no_grad():
                for p in model.parameters(): p.zero_()
            restore_adapted_vggt(host,path,sha256(path))
            for key,value in model.state_dict().items(): torch.testing.assert_close(value,reference[key],rtol=0,atol=0)
            with self.assertRaisesRegex(RuntimeError,'SHA mismatch'): restore_adapted_vggt(host,path,'0'*64)

    def test_native_adamw_equivalence_and_resume(self):
        torch.manual_seed(7)
        a=nn.Parameter(torch.randn(7,5));b=nn.Parameter(a.detach().clone())
        native=torch.optim.AdamW([a],lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=.05,foreach=False)
        group=dict(params=[b],name='test',peak_lr=1e-6,lr=1e-6,weight_decay=.05)
        offload=CPUOffloadAdamW([group])
        for _ in range(3):
            grad=torch.randn_like(a);a.grad=grad.clone();b.grad=grad.clone()
            native.step();offload.step()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        c=nn.Parameter(b.detach().clone())
        resumed=CPUOffloadAdamW([dict(group,params=[c])]);resumed.load_state_dict(offload.state_dict())
        grad=torch.randn_like(a);a.grad=grad.clone();c.grad=grad.clone()
        native.step();resumed.step()
        torch.testing.assert_close(a,c,rtol=0,atol=0)

    def test_adaptation_gradient_then_frozen_boundary(self):
        model=nn.Module();model.aggregator=TinyAggregator();model.camera_head=Camera();model.depth_head=Depth()
        wrapper=FrozenVGGT(model=model,pose_decoder=decode,test_only=True,preserve_fp32_aggregator=True)
        wrapper.set_reconstruction_adaptation(True)
        result=wrapper(torch.full((1,2,3,256,256),.3))
        result.patch_layers[-1].mean().backward()
        self.assertIsNotNone(model.aggregator.frame_blocks[0].weight.grad)
        self.assertIsNotNone(model.aggregator.global_blocks[0].weight.grad)
        self.assertIsNone(model.aggregator.patch_embed.weight.grad)
        self.assertFalse(result.c2w_cv.requires_grad)
        self.assertFalse(result.depth518.requires_grad)
        for p in wrapper.parameters(): p.grad=None
        wrapper.set_reconstruction_adaptation(False);wrapper.train()
        result=wrapper(torch.full((1,2,3,256,256),.3))
        self.assertFalse(wrapper.training)
        self.assertFalse(result.patch_layers[-1].requires_grad)
        self.assertFalse(any(p.requires_grad for p in wrapper.parameters()))


if __name__=='__main__': unittest.main()
