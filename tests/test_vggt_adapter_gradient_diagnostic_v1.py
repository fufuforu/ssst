"""Native loss instrumentation and adapter-only gradient-routing contracts."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn

from scripts import object_locus_frozen_vggt_posefree_runtime as native
from scripts.posefree_cpu_adamw import CPUOffloadAdamW
from scripts.probe_vggt_understanding_gradients_v1 import paired_stats
from tokengs.models.object_locus_v3_set_loss import v3_set_losses


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(2, 1, bias=False)
        self.vggt_memory_adapter = nn.Linear(2, 1, bias=False)
        self.understanding = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            for p in self.parameters(): p.copy_(torch.tensor([[.2, -.3]]))

    def step_loss(self, batch, *, step, understanding_weight):
        x, y = batch
        value = self.decoder(x) + self.vggt_memory_adapter(x) + self.understanding(x)
        return {}, dict(loss_recon=(value-y).square().mean(), loss_understanding=(value+2*y).square().mean())


def distributed_step(rank, directory):
    import torch.distributed as dist
    os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(Path(directory)/'rendezvous'), rank=rank, world_size=8)
    x = torch.arange(32, dtype=torch.float32).reshape(16,2) / 8
    y = torch.linspace(-.4,.8,16).reshape(16,1)
    for scale in (0., .01):
        model = TinyModel()
        optimizer = CPUOffloadAdamW(native.build_optimizer(model).param_groups)
        row = native.train_microbatch_window(model, optimizer,
            [(x[i:i+1], y[i:i+1]) for i in (rank, (rank+1)%8 + 8)], 0,
            base_exposure=50064, world_size=8, accumulation=2, cpu_gradient_accumulation=True,
            adapter_understanding_scale=scale)
        torch.save(dict(model=model.state_dict(), gradient={n:p.grad for n,p in model.named_parameters()}, row=row),
            Path(directory)/f'{rank}_{scale}.pt')
    dist.destroy_process_group()


class Contracts(unittest.TestCase):
    def test_only_adapter_understanding_gradient_changes_and_inputs_preserved(self):
        names = ['decoder.weight','understanding.weight','panoptic.weight','vggt_memory_adapter.proj.weight']
        rec = [torch.tensor([2.,-1.])] * 4
        under = [torch.tensor([-3.,4.])] * 4
        old = native.gc_combine(names,rec,under)
        explicit = native.gc_combine(names,rec,under,adapter_understanding_scale=1.)
        for a,b in zip(old,explicit): self.assertTrue(torch.equal(a,b))
        for scale in (0.,.01):
            changed=native.gc_combine(names,rec,under,adapter_understanding_scale=scale)
            for a,b in zip(old[:3],changed[:3]):self.assertTrue(torch.equal(a,b))
            torch.testing.assert_close(changed[-1],rec[-1]+scale*under[-1])
        self.assertTrue(torch.equal(rec[-1],torch.tensor([2.,-1.])))
        self.assertTrue(torch.equal(under[-1],torch.tensor([-3.,4.])))
        with self.assertRaises(ValueError):native.gc_combine(names,rec,under,adapter_understanding_scale=.1)

    def test_component_loss_and_gradient_sum_match_unmodified_native_total(self):
        torch.manual_seed(42)
        for instances in (True,False):
            logits=torch.randn(1,100,19,requires_grad=True)
            raw=torch.randn(1,2,102,4,4,requires_grad=True)
            sem=torch.zeros(1,2,4,4,dtype=torch.long)
            sem[:,:,2:,:]=1
            ins=torch.zeros_like(sem)
            if instances:sem[:,:,0,:]=2;ins[:,:,0,:]=7
            batch=dict(semantic_label_all=sem,instance_label_all=ins)
            pred=dict(states=[dict(thing_logits19=logits)],region_mass=raw.sigmoid())
            components={}
            loss,metrics=v3_set_losses(pred,batch,gradient_components=components)
            default,_=v3_set_losses(dict(states=pred['states'],region_mass=pred['region_mass']),batch)
            self.assertTrue(torch.equal(loss,default))
            torch.testing.assert_close(sum(components.values()),loss,rtol=1e-6,atol=1e-7)
            gradient=torch.autograd.grad(loss,(logits,raw),retain_graph=True)
            summed=torch.autograd.grad(sum(components.values()),(logits,raw))
            for a,b in zip(gradient,summed):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)

    def test_global_eight_rank_accumulation_clip_and_adam_match_mean_loss_reference(self):
        x=torch.arange(32,dtype=torch.float32).reshape(16,2)/8
        y=torch.linspace(-.4,.8,16).reshape(16,1)
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.start_processes(distributed_step,args=(directory,),nprocs=8,join=True,start_method='fork')
            for scale in (0.,.01):
                model=TinyModel();optimizer=CPUOffloadAdamW(native.build_optimizer(model).param_groups)
                row=native.train_microbatch_window(model,optimizer,[(x,y)],0,base_exposure=50064,
                    world_size=1,adapter_understanding_scale=scale)
                for rank in range(8):
                    actual=torch.load(Path(directory)/f'{rank}_{scale}.pt',weights_only=True)
                    self.assertAlmostEqual(actual['row']['preclip_norm'],row['preclip_norm'],places=5)
                    for name,p in model.named_parameters():
                        torch.testing.assert_close(actual['model'][name],p,rtol=1e-6,atol=1e-7)
                        torch.testing.assert_close(actual['gradient'][name],p.grad,rtol=1e-6,atol=1e-7)

    def test_family_statistics_handle_opposition_and_unused_tensors(self):
        result=paired_stats(['vggt_memory_adapter.weight','decoder.weight'],
            [torch.tensor([1.,0.]),None],[torch.tensor([-2.,0.]),torch.tensor([3.])])
        self.assertEqual(result['adapter']['cosine'],-1.)
        self.assertEqual(result['adapter']['norm_ratio'],2.)
        self.assertIsNone(result['decoder']['cosine'])

    def test_intervention_checkpoint_cannot_drop_objective_or_change_gs_scale(self):
        from scripts.eval_vggt_adapter_gradient_diagnostic_v1 import validate_clock, RECIPE, parent
        step=100
        b=dict(recipe=RECIPE,diagnostic_only=True,arm='pred_joint',completed_updates=step,
            completed_exposures=1600,source_checkpoint_sha256=parent.SOURCE_SHA,
            execution_git_sha=parent.TRAIN_SHA,frozen_parameters_unchanged=True,rank_rng=[{}]*8,
            rank_window_exposure_counts=[[step if i in (r%4,(r+1)%4) else 0 for i in range(4)] for r in range(8)],
            window_exposure_counts=[step*4]*4,
            config=dict(recipe=RECIPE,arm='pred_joint',world_size=8,global_batch=16,accumulation=2,
                unique_training_windows=4,total_updates=300,total_exposures=4800,feedback_enabled=True,
                beta=.1,geometry_quality_policy='monitor_v1',source_checkpoint_sha256=parent.SOURCE_SHA,
                vggt_revision=parent.REVISION,understanding_weight=1.,gradient_routing_variant='adapter_u0',
                adapter_understanding_scale=0.,reconstruction_understanding_scale=.01))
        validate_clock(b,'adapter_u0',step)
        for key,value in [('understanding_weight',0.),('reconstruction_understanding_scale',0.),
                          ('adapter_understanding_scale',1.),('feedback_enabled',False)]:
            x=copy.deepcopy(b);x['config'][key]=value
            with self.assertRaises(ValueError):validate_clock(x,'adapter_u0',step)


if __name__=='__main__':unittest.main()
