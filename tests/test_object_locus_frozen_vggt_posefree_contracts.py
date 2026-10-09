"""CPU contracts for the frozen-VGGT pose-free migration (no real VGGT weights)."""
import math
import inspect
import types
import unittest

import torch
from torch import nn

from tokengs.models.frozen_vggt_posefree import FrozenVGGT, select_patch_layers
from tokengs.models.object_locus_frozen_vggt_posefree import VGGTMemoryAdapter, initialize_memory_adapter
from tokengs.models.object_locus_frozen_vggt_posefree import LocusGSObjectLocusFrozenVGGT
from tokengs.models.object_locus_posefree_geometry import (
    align_cameras_by_shared_context, camera_vectors, intrinsics518_to_256,
    pixel_rays, posefree_scene, rays_to_patch_plucker,
)
from scripts.object_locus_frozen_vggt_posefree_runtime import (
    epoch_order, exposure_schedule, gc_combine, lr_multiplier, parameter_family,
    rank_microbatch_indices, migrate_model_state, train_microbatch_window,
    load_manifest,
)


def _camera(rotation=None, center=None):
    value=torch.eye(4,dtype=torch.float64)
    if rotation is not None: value[:3,:3]=rotation
    if center is not None: value[:3,3]=torch.tensor(center,dtype=torch.float64)
    return value


class AggregatorStub(nn.Module):
    def __init__(self):
        super().__init__(); self.received=None
    def forward(self,images):
        self.received=images.detach().clone()
        b,v=images.shape[:2]
        items=[None]*24
        for i in (4,11,17,23):
            items[i]=torch.full((b,v,1370,2048),float(i),device=images.device,dtype=images.dtype)
        return items,1


class CameraHeadStub(nn.Module):
    def forward(self,tokens):
        return [torch.zeros(tokens[23].shape[:2]+(9,),device=tokens[23].device)]


class DepthHeadStub(nn.Module):
    def forward(self,tokens,images,patch_start_idx):
        return torch.ones(images.shape[:2]+(1,518,518),device=images.device), torch.ones(images.shape[:2]+(1,518,518),device=images.device)


class VGGTStub(nn.Module):
    def __init__(self):
        super().__init__(); self.aggregator=AggregatorStub(); self.camera_head=CameraHeadStub(); self.depth_head=DepthHeadStub()


class UnderstandingStub(nn.Module):
    def __init__(self):
        super().__init__(); self.mask_embedder=nn.Identity()
    def forward(self,images):
        b=images.shape[0]
        feature=images.new_zeros((b,2,256,128,128))
        query=images.new_zeros((b,100,256))
        return feature,query


class AnchorDecoderStub(nn.Module):
    def forward_stateful(self,tokens,latent,rays,controller,fm,qpre,mask_embedder,exposure):
        assert latent.keys.shape==(tokens.shape[0],16,2738,64)
        assert rays[0].shape==(tokens.shape[0],2738,3)
        state={'tokens':tokens,'mu':tokens.new_zeros((tokens.shape[0],1024,3)),
               'radii':tokens.new_ones((tokens.shape[0],1024)), 'beta':0.1*min(exposure/1000,1)}
        return [state],[]


class GaussianHeadStub(nn.Module):
    def forward(self,tokens,mu,radii): return tokens.mean(-1)


def _pose_decode(encoding,image_size_hw):
    b,v=encoding.shape[:2]
    ext=torch.zeros((b,v,3,4),device=encoding.device)
    ext[:,:,:3,:3]=torch.eye(3,device=encoding.device)
    k=torch.eye(3,device=encoding.device).expand(b,v,3,3).clone()
    k[:,:,0,0]=k[:,:,1,1]=400.; k[:,:,0,2]=k[:,:,1,2]=259.
    return ext,k


class FrozenVGGTContracts(unittest.TestCase):
    def test_actual_indices_patch_selection_and_none_contract(self):
        values=[None]*24
        for i in (4,11,17,23): values[i]=torch.full((1,2,1370,2048),float(i))
        selected=select_patch_layers(values,1,batch=1)
        self.assertEqual([float(x[0,0,0,0]) for x in selected],[4.,11.,17.,23.])
        self.assertEqual(selected[0].shape,(1,2,1369,2048))
        values[11]=None
        with self.assertRaisesRegex(ValueError,'layer 11 is None'): select_patch_layers(values,1,batch=1)
        with self.assertRaisesRegex(ValueError,'patch count'): select_patch_layers([torch.zeros(1,2,1368,2048)]*24,0,batch=1)

    def test_stub_wrapper_has_frozen_no_grad_official_boundaries(self):
        stub=VGGTStub()
        wrapper=FrozenVGGT(model=stub,pose_decoder=_pose_decode,source_identity={'kind':'controlled_stub'})
        rgb=torch.full((1,2,3,256,256),.375)
        with torch.enable_grad(): result=wrapper(rgb.requires_grad_())
        self.assertFalse(result.patch_layers[0].requires_grad)
        self.assertEqual(result.patch_layers[0].shape,(1,2,1369,2048))
        self.assertEqual(tuple(result.depth518.shape),(1,2,1,518,518))
        self.assertEqual(tuple(result.c2w_cv.shape),(1,2,4,4))
        self.assertEqual(tuple(stub.aggregator.received.shape),(1,2,3,518,518))
        self.assertTrue(torch.allclose(stub.aggregator.received,torch.full_like(stub.aggregator.received,.375,dtype=torch.bfloat16)))
        self.assertTrue(all(not p.requires_grad for p in wrapper.parameters()))
        self.assertFalse(wrapper.training)

    def test_generation_api_has_no_gt_or_novel_inputs(self):
        signature=inspect.signature(LocusGSObjectLocusFrozenVGGT.generate)
        self.assertEqual(tuple(signature.parameters),('self','context_rgb','runtime_config'))

    def test_context_only_generation_stub_ignores_external_gt_and_novel_content(self):
        model=LocusGSObjectLocusFrozenVGGT.__new__(LocusGSObjectLocusFrozenVGGT)
        nn.Module.__init__(model)
        model.frozen_vggt=FrozenVGGT(model=VGGTStub(),pose_decoder=_pose_decode,source_identity={'kind':'controlled_stub'})
        model.vggt_memory_adapter=VGGTMemoryAdapter()
        model.understanding=UnderstandingStub(); model.understanding_step=0
        model.anchor_decoder=AnchorDecoderStub(); model.activation_head=GaussianHeadStub(); model.panoptic=nn.Identity()
        model.get_gs_tokens=types.MethodType(lambda self,batch_size: torch.ones((batch_size,1024,1024)),model)
        model._reconstruction_from_gaussians=types.MethodType(lambda self,gaussians: gaussians,model)
        model.render_reconstruction=types.MethodType(lambda self,reconstruction,decoder: {'gaussians':reconstruction},model)
        model._readout=types.MethodType(lambda self,*args: {},model)
        context=torch.full((1,2,3,256,256),.25)
        changed_gt=torch.randn((1,2,3,256,256)); changed_novel=torch.randn((1,5,3,256,256))
        # Neither external value is accepted by the generation signature.
        with torch.no_grad(): first=model.generate(context); second=model.generate(context)
        self.assertTrue(torch.equal(first['gaussians'],second['gaussians']))
        self.assertTrue(torch.equal(first['predicted_context_c2w'],second['predicted_context_c2w']))
        self.assertEqual(first['states'][-1]['tokens'].shape,(1,1024,1024))
        self.assertEqual(tuple(first['predicted_context_intrinsics'].shape),(1,2,4))
        self.assertEqual(changed_gt.shape,(1,2,3,256,256)); self.assertEqual(changed_novel.shape,(1,5,3,256,256))


class MemoryContracts(unittest.TestCase):
    def test_adapter_shape_weights_view_major_and_seed_isolation(self):
        adapter=VGGTMemoryAdapter()
        rng=torch.random.get_rng_state().clone()
        initialize_memory_adapter(adapter,31415)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(torch.equal(adapter.layer_logits,torch.zeros(4)))
        layers=[torch.full((1,2,1369,2048),float(i)) for i in range(4)]
        memory,key,value=adapter(layers)
        self.assertEqual(memory.shape,(1,2738,1024))
        self.assertEqual(key.shape,(1,16,2738,64)); self.assertEqual(value.shape,key.shape)
        self.assertTrue(torch.allclose(adapter.layer_logits.softmax(0),torch.full((4,),.25)))
        self.assertTrue(torch.isfinite(memory).all())

    def test_layer_and_kv_modules_are_trainable(self):
        adapter=VGGTMemoryAdapter()
        self.assertTrue(all(p.requires_grad for p in adapter.parameters()))
        names=dict(adapter.named_parameters())
        for expected in ('layer_logits','layer_norms.0.weight','projection.weight','output_norm.weight','kv.weight','key_norm.weight'):
            self.assertIn(expected,names)


class CameraGeometryContracts(unittest.TestCase):
    def test_pixel_projection_backprojection_and_camera_scale(self):
        dtype=torch.float64
        c2w=torch.eye(4,dtype=dtype).reshape(1,1,4,4)
        k518=torch.tensor([[[[500.,0,259.],[0,500.,259.],[0,0,1.]]]],dtype=dtype)
        k256,A=intrinsics518_to_256(k518)
        self.assertTrue(torch.allclose(k256,A@k518))
        ray_o,ray_d=pixel_rays(c2w,k256,256,256)
        self.assertEqual(tuple(ray_o.shape),(1,1,3,256,256))
        y=x=127
        direction=ray_d[0,0,:,y,x]
        projected=k256[0,0]@(direction/direction[2])
        self.assertTrue(torch.allclose(projected[:2],torch.tensor([127.5,127.5],dtype=dtype),atol=1e-9))

        c2w2=torch.eye(4,dtype=dtype).reshape(1,1,4,4).repeat(1,2,1,1)
        c2w2[:,1,0,3]=2
        k2=k518.repeat(1,2,1,1)
        depth=torch.ones((1,2,1,518,518),dtype=dtype)*2.
        scene,k256s,depths,points,record=posefree_scene(c2w2,k2,depth)
        self.assertTrue(torch.allclose(scene[:,0],torch.eye(4,dtype=dtype)[None]))
        self.assertTrue(torch.allclose(scene[0,1,:3,3],torch.tensor([.25,0,0],dtype=dtype)))
        self.assertTrue(torch.allclose(depths,torch.full_like(depths,.25)))
        self.assertTrue(torch.allclose(points[0,0,259,259],torch.tensor([.00025,.00025,.25],dtype=dtype),atol=1e-8))
        self.assertTrue(torch.allclose(record['a_scale'],torch.tensor([.125],dtype=dtype)))

    def test_518_dense_rays_pool_to_2738_in_memory_order(self):
        c2w=torch.eye(4).reshape(1,1,4,4).repeat(1,2,1,1)
        c2w[:,1,0,3]=.25
        k=torch.tensor([[[[500.,0,259.],[0,500.,259.],[0,0,1.]]]]).repeat(1,2,1,1)
        moment,direction=rays_to_patch_plucker(c2w,k,patch_size=14)
        self.assertEqual(moment.shape,(1,2738,3)); self.assertEqual(direction.shape,moment.shape)
        self.assertLess(float(direction[0,0].norm()),1.01)
        self.assertGreater(float(moment[0,1369].norm()),0.)

    def test_known_orientation_constrained_sim3_and_degenerate_baseline(self):
        theta=.37
        R=torch.tensor([[math.cos(theta),-math.sin(theta),0.],[math.sin(theta),math.cos(theta),0.],[0,0,1.]],dtype=torch.float64)
        c_all=torch.stack((_camera(center=[0,0,0]),_camera(center=[1,0,0]),_camera(center=[.3,.7,.2])))
        t=torch.tensor([.4,-.2,.8],dtype=torch.float64); s=1.7
        pair=c_all[:2].clone(); pair[:,:3,:3]=R@pair[:,:3,:3]
        pair[:,:3,3]=s*(pair[:,:3,3]@R.T)+t
        aligned,records=align_cameras_by_shared_context(c_all[None],pair[None],pair[None],torch.tensor([.25],dtype=torch.float64),torch.eye(4,dtype=torch.float64)[None])
        self.assertTrue(torch.allclose(records[0]['R'],R,atol=1e-8))
        self.assertAlmostEqual(float(records[0]['s']),s,places=7)
        self.assertTrue(torch.allclose(records[0]['t'],t,atol=1e-8))
        self.assertTrue(torch.allclose(aligned[0,:2,:3,:3],pair[:2,:3,:3],atol=1e-8))
        degenerate=torch.stack((_camera(center=[0,0,0]),_camera(center=[0,0,0]),_camera(center=[1,0,0])))
        with self.assertRaisesRegex(ValueError,'degenerate shared-context camera baseline'):
            align_cameras_by_shared_context(degenerate[None],degenerate[:2][None],degenerate[:2][None],torch.ones(1),torch.eye(4)[None])


class MigrationAndTrainingContracts(unittest.TestCase):
    def test_fixed_full1201_manifest_is_used_without_resampling(self):
        _,scenes,windows=load_manifest()
        self.assertEqual((len(scenes),len(windows)),(1191,8337))
        self.assertEqual((len(windows[0]['context']),len(windows[0]['novel'])),(2,2))

    def test_explicit_migration_whitelist_and_unknown_key_error(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__(); self.gs_tokens=nn.Parameter(torch.zeros(2,3)); self.anchor_decoder=nn.Linear(3,2); self.vggt_memory_adapter=nn.Linear(2,2)
        model=Tiny()
        source={'gs_tokens':torch.ones_like(model.gs_tokens),
                'anchor_decoder.weight':torch.ones_like(model.anchor_decoder.weight),
                'anchor_decoder.bias':torch.ones_like(model.anchor_decoder.bias),
                'patch_embed.proj.weight':torch.zeros(1)}
        report=migrate_model_state(model,source)
        self.assertEqual(report['counts']['loaded'],3); self.assertEqual(report['explicitly_excluded'],['patch_embed.proj.weight'])
        self.assertEqual(report['newly_initialized'],['vggt_memory_adapter.bias','vggt_memory_adapter.weight'])
        self.assertTrue(torch.equal(model.gs_tokens,torch.ones_like(model.gs_tokens)))
        with self.assertRaisesRegex(RuntimeError,'unclassified keys'):
            migrate_model_state(model,{**source,'mystery.weight':torch.ones(1)})

    def test_frozen_optimizer_exclusion_gc_exposure_and_sampler(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__(); self.reconstruction=nn.Linear(2,2); self.understanding=nn.Linear(2,2); self.panoptic=nn.Linear(2,2); self.vggt_memory_adapter=nn.Linear(2,2); self.frozen_vggt=nn.Linear(2,2); self.frozen_vggt.requires_grad_(False)
        from scripts.object_locus_frozen_vggt_posefree_runtime import build_optimizer
        model=Tiny(); optimizer=build_optimizer(model)
        self.assertFalse(any(parameter_family(n)=='frozen_vggt' for g in optimizer.param_groups for n in g['param_names']))
        recon_group=next(g for g in optimizer.param_groups if g['name']=='reconstruction_nodecay')
        self.assertEqual(recon_group['weight_decay'],0.)
        self.assertFalse(any(g['name']=='reconstruction_decay' for g in optimizer.param_groups))
        self.assertEqual(exposure_schedule(0),(0.,0.)); self.assertEqual(exposure_schedule(200)[0],1.)
        self.assertAlmostEqual(exposure_schedule(200)[1],.02)
        self.assertAlmostEqual(lr_multiplier(0),.005); self.assertEqual(lr_multiplier(199),1.)
        self.assertAlmostEqual(lr_multiplier(8343),.1)
        self.assertEqual(len(epoch_order(0)),8344)
        self.assertEqual(epoch_order(0).tolist(),epoch_order(0).tolist())
        shards=[rank_microbatch_indices(2,4,r) for r in range(4)]
        self.assertEqual(len({i for shard in shards for i in shard}),8)
        params=[torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
        grads=gc_combine(['reconstruction.weight','understanding.weight','panoptic.weight'],
            [torch.ones(1)]*3,[torch.full((1,),2.)]*3)
        self.assertTrue(torch.allclose(grads[0],torch.tensor([1.02])))
        self.assertTrue(torch.allclose(grads[1],torch.tensor([3.])))
        self.assertTrue(torch.allclose(grads[2],torch.tensor([3.])))

    def test_two_microbatch_gradient_accumulation_and_exposure_clock(self):
        class TinyTrain(nn.Module):
            def __init__(self):
                super().__init__(); self.reconstruction=nn.Linear(1,1,bias=False); self.understanding=nn.Linear(1,1,bias=False); self.panoptic=nn.Linear(1,1,bias=False); self.vggt_memory_adapter=nn.Linear(1,1,bias=False); self.frozen_vggt=nn.Linear(1,1,bias=False); self.frozen_vggt.requires_grad_(False); self.steps=[]
            def step_loss(self,batch,*,step,understanding_weight):
                self.steps.append((step,understanding_weight))
                x=batch
                rec=self.reconstruction(x).square().mean()
                under=self.understanding(x).square().mean()+self.panoptic(x).square().mean()+self.vggt_memory_adapter(x).square().mean()
                return {},{'loss_recon':rec,'loss_understanding':under}
        model=TinyTrain(); optimizer=__import__('scripts.object_locus_frozen_vggt_posefree_runtime',fromlist=['build_optimizer']).build_optimizer(model)
        row=train_microbatch_window(model,optimizer,[torch.ones(2,1),torch.full((2,1),2.)],0,base_exposure=0)
        self.assertEqual(model.steps,[(0,0.),(0,0.)])
        self.assertEqual(row['completed_updates'],1); self.assertEqual(row['completed_exposures'],8)
        self.assertEqual(row['understanding_weight'],0.); self.assertEqual(row['beta'],0.)


if __name__=='__main__': unittest.main(verbosity=2)
