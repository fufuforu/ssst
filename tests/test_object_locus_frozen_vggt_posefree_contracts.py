"""CPU contracts for the frozen-VGGT pose-free migration (no real VGGT weights)."""
import math
import inspect
import types
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from tokengs.models.frozen_vggt_posefree import FrozenVGGT, select_patch_layers
from tokengs.models.object_locus_frozen_vggt_posefree import VGGTMemoryAdapter, initialize_memory_adapter
from tokengs.models.object_locus_frozen_vggt_posefree import LocusGSObjectLocusFrozenVGGT
from tokengs.models.object_locus_posefree_geometry import (
    align_cameras_by_shared_context, align_cameras_by_shared_context_depth_v2,
    ContextDepthSim3Error, old_shared_context_alignment_diagnostics,
    camera_vectors, intrinsics518_to_256, pixel_rays, posefree_scene, rays_to_patch_plucker,
)
from tokengs.models.input_types import ModelInput
from scripts.object_locus_frozen_vggt_posefree_runtime import (
    epoch_order, exposure_schedule, gc_combine, lr_multiplier, parameter_family,
    rank_microbatch_indices, migrate_model_state, train_microbatch_window,
    load_manifest, checkpoint_model_state, restore_model_state_strict,
    capture_rank_rng, restore_rank_rng, _atomic_save, training_configuration,
    smoke_exposure_count, WORLD_SIZE, GLOBAL_BATCH, ACCUMULATION,
)

REPO=Path(__file__).resolve().parents[1]


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
        with self.assertRaisesRegex(ValueError,'test[_-]only'):
            FrozenVGGT(model=VGGTStub())
        with self.assertRaisesRegex(ValueError,'from_pretrained'):
            FrozenVGGT()
        wrapper=FrozenVGGT(model=stub,pose_decoder=_pose_decode,source_identity={'kind':'controlled_stub'},test_only=True)
        rgb=torch.full((1,2,3,256,256),.375)
        with torch.enable_grad(): result=wrapper(rgb.requires_grad_())
        self.assertFalse(result.patch_layers[0].requires_grad)
        self.assertEqual(result.patch_layers[0].shape,(1,2,1369,2048))
        self.assertEqual(tuple(result.depth518.shape),(1,2,1,518,518))
        self.assertEqual(tuple(result.confidence518.shape),(1,2,1,518,518))
        self.assertEqual(tuple(result.c2w_cv.shape),(1,2,4,4))
        self.assertEqual(tuple(stub.aggregator.received.shape),(1,2,3,518,518))
        self.assertTrue(torch.allclose(stub.aggregator.received,torch.full_like(stub.aggregator.received,.375,dtype=torch.bfloat16)))
        self.assertTrue(all(not p.requires_grad for p in wrapper.parameters()))
        self.assertFalse(wrapper.training)

    def test_generation_api_has_no_gt_or_novel_inputs(self):
        signature=inspect.signature(LocusGSObjectLocusFrozenVGGT.generate)
        self.assertEqual(tuple(signature.parameters),('self','context_rgb','runtime_config'))
        model=LocusGSObjectLocusFrozenVGGT.__new__(LocusGSObjectLocusFrozenVGGT);nn.Module.__init__(model)
        legacy=ModelInput(None,None)
        for name in ('forward_object_locus','forward_instance_state','forward_reconstruction_only'):
            with self.assertRaisesRegex(TypeError,'unsupported'):
                getattr(model,name)(legacy)
        with self.assertRaisesRegex(TypeError,'legacy ModelInput/GT-camera'):
            model.forward(legacy)

    def test_context_only_generation_stub_ignores_external_gt_and_novel_content(self):
        model=LocusGSObjectLocusFrozenVGGT.__new__(LocusGSObjectLocusFrozenVGGT)
        nn.Module.__init__(model)
        model.frozen_vggt=FrozenVGGT(model=VGGTStub(),pose_decoder=_pose_decode,source_identity={'kind':'controlled_stub'},test_only=True)
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
        with patch('tokengs.models.object_locus_frozen_vggt_posefree.rays_to_patch_plucker',wraps=rays_to_patch_plucker) as ray_builder:
            with torch.no_grad(): first=model.generate(context); second=model.generate(context)
        passed_k=ray_builder.call_args.args[1]
        expected_k=model.frozen_vggt(context).intrinsics518
        self.assertTrue(torch.equal(passed_k,expected_k))
        _,k256=posefree_scene(expected_k.new_zeros((1,2,4,4))+torch.eye(4),expected_k,
            torch.ones((1,2,1,518,518)))[:2]
        self.assertFalse(torch.equal(passed_k,k256))
        self.assertTrue(torch.equal(first['gaussians'],second['gaussians']))
        self.assertTrue(torch.equal(first['predicted_context_c2w'],second['predicted_context_c2w']))
        self.assertEqual(first['states'][-1]['tokens'].shape,(1,1024,1024))
        self.assertEqual(tuple(first['predicted_context_intrinsics'].shape),(1,2,4))
        self.assertEqual(changed_gt.shape,(1,2,3,256,256)); self.assertEqual(changed_novel.shape,(1,5,3,256,256))

    def test_target_render_reuses_membership_and_class_without_regeneration(self):
        class RendererStub:
            def render_feature_channels(self,gaussians,membership,cam_view,intrinsics):
                self.calls=getattr(self,'calls',0)+1
                self.seen_membership=membership
                b,v=cam_view.shape[:2];h=w=3
                mass=membership.new_full((b,v,102,h,w),.2)
                alpha=membership.new_full((b,v,1,h,w),.8)
                return {'images_pred':mass,'alphas_pred':alpha}
        model=LocusGSObjectLocusFrozenVGGT.__new__(LocusGSObjectLocusFrozenVGGT);nn.Module.__init__(model)
        model.gs=RendererStub()
        model.render_reconstruction=types.MethodType(lambda self,reconstruction,decoder:{'images_pred':reconstruction},model)
        model.generate=types.MethodType(lambda *a,**k: (_ for _ in ()).throw(AssertionError('regenerated')),model)
        model._readout=types.MethodType(lambda *a,**k: (_ for _ in ()).throw(AssertionError('relifted')),model)
        membership=torch.rand(1,1024,100)
        pclass=torch.full((1,100,19),1/19)
        generated={'reconstruction':torch.ones(1),'gaussians':torch.ones(1,1),
            'gaussian_membership':membership,'p_class':pclass,'states':[{'q':torch.ones(1)}]}
        camera=torch.eye(4).reshape(1,1,4,4).repeat(1,2,1,1)
        out=model.render_generated_at(generated,torch.linalg.inv(camera).transpose(-1,-2),torch.tensor([[[100.,100.,1.,1.],[120.,120.,1.,1.]]]))
        self.assertEqual(model.gs.calls,1);self.assertIs(model.gs.seen_membership,membership)
        self.assertIs(out['p_class'],pclass);self.assertIs(out['gaussian_membership'],membership)
        self.assertEqual(out['semantic_scores'].shape,(1,2,20,3,3))
        self.assertEqual(out['region_mass'].shape,(1,2,102,3,3))

    def test_calibration_entry_keeps_generated_state_immutable(self):
        model=LocusGSObjectLocusFrozenVGGT.__new__(LocusGSObjectLocusFrozenVGGT);nn.Module.__init__(model)
        context=torch.rand(1,2,3,8,8)
        generated={'generation_context_rgb':context.clone(),'gaussians':torch.randn(1,7,6),
            'gaussian_membership':torch.randn(1,7,100),'p_class':torch.randn(1,100,19),
            'predicted_points':torch.randn(1,2,518,518,3),'states':[{'tokens':torch.randn(1,1024,16)}]}
        before={key:value.clone() for key,value in generated.items() if torch.is_tensor(value)}
        before['state_tokens']=generated['states'][-1]['tokens'].clone()
        model.frozen_vggt=types.SimpleNamespace(calibration_with_context_depth=lambda images:{'marker':images.clone()})
        model._calibrate_result=types.MethodType(lambda self,result,raw:raw,model)
        returned=model.calibrate_targets(torch.cat((context,torch.rand(1,1,3,8,8)),dim=1),generated)
        self.assertEqual(returned['marker'].shape,(1,3,3,8,8))
        for key,value in before.items():
            actual=generated['states'][-1]['tokens'] if key=='state_tokens' else generated[key]
            self.assertTrue(torch.equal(actual,value),key)

    def test_eval_cli_has_locked_contract_arguments(self):
        from scripts.eval_object_locus_frozen_vggt_posefree_v1 import build_parser,DISCLOSURE
        from scripts.export_object_locus_v3_set_official import write_official_pair
        args=build_parser().parse_args(['--checkpoint','/tmp/model.pt','--manifest','/tmp/manifest.json',
            '--cohort','full_validation','--output-root','/tmp/out','--vggt-revision','a'*40])
        self.assertEqual(args.cohort,'full_validation');self.assertEqual(len(args.vggt_revision),40)
        self.assertIn('指定新视角渲染仍需要目标相机',DISCLOSURE)
        h=w=2
        out={'p_class':torch.zeros(1,100,19),'region_mass':torch.zeros(1,4,102,h,w),
            'alpha':torch.ones(1,4,1,h,w),'semantic_scores':torch.zeros(1,4,20,h,w),
            'render':{'images_pred':torch.zeros(1,4,3,h,w)}}
        batch={'frame_ids':torch.tensor([[10,11,12,13]]),
            'semantic_label_all':torch.zeros(1,4,h,w,dtype=torch.long),
            'instance_label_all':torch.zeros(1,4,h,w,dtype=torch.long)}
        window={'scene':'scene0000_00','context':[10,11],'novel':[12,13]}
        with tempfile.TemporaryDirectory() as directory:
            row=write_official_pair(out,batch,window,directory,target_frames='novel')
            self.assertEqual(row['context_frames'],[10,11]);self.assertEqual(row['target_frames'],[12,13])
            pred=Path(directory)/'scene0000_00_context10_11'/'target_seg_pred'
            self.assertTrue((pred/'pred.json').is_file())
            self.assertEqual(sorted(p.name for p in pred.glob('*.png')),
                ['scene0000_00_pred12.png','scene0000_00_pred13.png'])

    def test_posefree_eval_cache_keeps_context_target_and_rendered_depth(self):
        from scripts.eval_object_locus_frozen_vggt_posefree_v1 import _official_export_view,_save_reconstruction_cache
        batch={'frame_ids':torch.tensor([[10,11,12,13]]),
            'images_all':torch.rand(1,4,3,4,4),
            'depth_gt_m_all':torch.ones(1,4,1,4,4),
            'depth_gt_valid_all':torch.ones(1,4,1,4,4,dtype=torch.bool)}
        render={'images_pred':torch.rand(1,4,3,4,4),
            'depths_pred':torch.rand(1,4,1,4,4)}
        exported=_official_export_view(render)
        self.assertIs(exported['render']['images_pred'],render['images_pred'])
        self.assertIs(exported['render']['depths_pred'],render['depths_pred'])
        with tempfile.TemporaryDirectory() as directory:
            path=_save_reconstruction_cache(directory,{'scene':'scene0000_00','context':[10,11],'novel':[10,11,12,13]},batch,render)
            with __import__('numpy').load(path) as cache:
                self.assertEqual(cache['frame_ids'].tolist(),[10,11,12,13])
                self.assertEqual(cache['context_ids'].tolist(),[10,11])
                self.assertEqual(cache['novel_ids'].tolist(),[12,13])
                self.assertEqual(cache['pred_rgb'].shape,(4,3,4,4))
                self.assertEqual(cache['pred_depth'].shape,(4,1,4,4))


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

    def test_memory_flattens_nonconstant_views_then_raster_positions(self):
        adapter=VGGTMemoryAdapter()
        spatial=torch.arange(2738,dtype=torch.float32).reshape(1,2,1369,1)
        channels=torch.arange(2048,dtype=torch.float32).reshape(1,1,1,2048)
        layers=[torch.sin(spatial*(.001*(layer+1)))+channels*(.0001*(layer+1))+layer
                for layer in range(4)]
        captured=[]
        handle=adapter.projection.register_forward_pre_hook(lambda _module,args:captured.append(args[0].detach().clone()))
        memory,_,_=adapter(layers);handle.remove()
        expected=sum(adapter.layer_logits.softmax(0)[i]*adapter.layer_norms[i](layers[i]) for i in range(4))
        expected=expected.reshape(1,2738,2048)
        self.assertTrue(torch.allclose(captured[0],expected,atol=1e-6))
        self.assertFalse(torch.allclose(captured[0][:,:1369],captured[0][:,1369:]))
        self.assertFalse(torch.allclose(captured[0][:,0],captured[0][:,1]))
        self.assertEqual(memory.shape,(1,2738,1024))


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
        dtype=torch.float64
        theta=.23
        rotation=torch.tensor([[math.cos(theta),0.,math.sin(theta)],[0.,1.,0.],[-math.sin(theta),0.,math.cos(theta)]],dtype=dtype)
        c2w=torch.stack((_camera(rotation,[.1,-.2,.3]),_camera(rotation.T,[.8,.1,-.4])))[None]
        k518=torch.tensor([[[[431.,0,247.3],[0,517.,269.1],[0,0,1.]],[[503.,0,281.7],[0,389.,241.2],[0,0,1.]]]],dtype=dtype)
        moment,direction=rays_to_patch_plucker(c2w,k518,patch_size=14)
        self.assertEqual(moment.shape,(1,2738,3)); self.assertEqual(direction.shape,moment.shape)
        # Independently construct dense K518 rays, then average-pool the
        # resulting Pluecker map in view-major raster order.
        ro,rd=pixel_rays(c2w,k518,518,518)
        from tokengs.models.canonical_recon_models import patch_plucker_rays
        ref_m,ref_d=patch_plucker_rays(ro,rd,patch_size=14)
        self.assertTrue(torch.allclose(moment,ref_m,atol=1e-11))
        self.assertTrue(torch.allclose(direction,ref_d,atol=1e-11))
        self.assertFalse(torch.allclose(direction[0,:1369],direction[0,1369:]))
        # Exercise the same conversion used by posefree_scene -> generate.
        depth=torch.full((1,2,1,518,518),2.,dtype=dtype)
        raw=c2w.clone()
        w2c=torch.linalg.inv(raw)
        predicted=w2c
        c2w_cv=torch.linalg.inv(predicted)
        scene,k256,*_=posefree_scene(c2w_cv,k518,depth)
        actual_m,actual_d=rays_to_patch_plucker(scene,k518,patch_size=14)
        scene_ro,scene_rd=pixel_rays(scene,k518,518,518)
        ref_scene_m,ref_scene_d=patch_plucker_rays(scene_ro,scene_rd,patch_size=14)
        self.assertTrue(torch.allclose(actual_m,ref_scene_m,atol=1e-11))
        self.assertTrue(torch.allclose(actual_d,ref_scene_d,atol=1e-11))
        # Continuous corresponding coordinates p256=A*p518 define the same ray.
        k256,A=intrinsics518_to_256(k518)
        p518=k518.new_tensor([217.25,301.75,1.]).view(1,1,1,3).expand(1,2,1,3)
        p256=torch.einsum('bvij,bvnj->bvni',A,p518)
        d518=torch.nn.functional.normalize(torch.einsum('bvij,bvnj->bvni',torch.linalg.inv(k518),p518),dim=-1)
        d256=torch.nn.functional.normalize(torch.einsum('bvij,bvnj->bvni',torch.linalg.inv(k256),p256),dim=-1)
        self.assertTrue(torch.allclose(d518,d256,atol=1e-12))
        wrong_m,wrong_d=rays_to_patch_plucker(scene,k256,patch_size=14)
        self.assertFalse(torch.allclose(wrong_d,ref_d,atol=1e-5))

    def test_known_orientation_constrained_sim3_and_degenerate_baseline(self):
        theta=.37
        R=torch.tensor([[math.cos(theta),-math.sin(theta),0.],[math.sin(theta),math.cos(theta),0.],[0,0,1.]],dtype=torch.float64)
        angle=.19
        R0=torch.tensor([[math.cos(angle),-math.sin(angle),0.],[math.sin(angle),math.cos(angle),0.],[0.,0.,1.]],dtype=torch.float64)
        c_all=torch.stack((_camera(rotation=R0,center=[.4,-.1,.7]),_camera(rotation=R0.T,center=[1.0001,-.1,.7]),_camera(center=[.3,.7,.2])))
        t=torch.tensor([.4,-.2,.8],dtype=torch.float64); s=1.7
        pair=c_all[:2].clone(); pair[:,:3,:3]=R@pair[:,:3,:3]
        pair[:,:3,3]=s*(pair[:,:3,3]@R.T)+t
        first_inverse=torch.linalg.inv(c_all[0])[None]
        aligned,records=align_cameras_by_shared_context(c_all[None],pair[None],pair[None],torch.tensor([.25],dtype=torch.float64),first_inverse)
        self.assertTrue(torch.allclose(records[0]['R'],R,atol=1e-8))
        self.assertAlmostEqual(float(records[0]['s']),s,places=7)
        self.assertTrue(torch.allclose(records[0]['t'],t,atol=1e-8))
        self.assertTrue(torch.allclose(aligned[0,:2,:3,:3],first_inverse[0,:3,:3]@pair[:2,:3,:3],atol=1e-8))
        degenerate=torch.stack((_camera(center=[0,0,0]),_camera(center=[0,0,0]),_camera(center=[1,0,0])))
        with self.assertRaisesRegex(ValueError,'degenerate shared-context camera baseline'):
            align_cameras_by_shared_context(degenerate[None],degenerate[:2][None],degenerate[:2][None],torch.ones(1),torch.eye(4)[None])
        # A valid baseline can have squared length below the length threshold.
        tiny=torch.stack((_camera(center=[0.,0.,0.]),_camera(center=[2e-4,0.,0.]),_camera(center=[4e-4,0.,0.])))
        result,_=align_cameras_by_shared_context(tiny[None],tiny[:2][None],tiny[:2][None],torch.ones(1),torch.eye(4,dtype=torch.float64)[None])
        self.assertTrue(torch.isfinite(result).all())

    @staticmethod
    def _v2_fixture(planar=False, outliers=False):
        dtype=torch.float64
        rows=torch.arange(7,518,14,dtype=dtype);cols=torch.arange(7,518,14,dtype=dtype)
        row_grid,col_grid=torch.meshgrid(rows,cols,indexing='ij')
        row_center,col_center=row_grid+.5,col_grid+.5
        pixels=torch.stack((col_center,row_center,torch.ones_like(row_center)),-1).reshape(1369,3)
        k=torch.tensor([[420.,0,251.],[0,405.,263.],[0,0,1.]],dtype=dtype)
        k_all=k.reshape(1,1,3,3).repeat(1,4,1,1)
        k256=intrinsics518_to_256(k.reshape(1,1,3,3).repeat(1,2,1,1))[0]
        c_all=torch.eye(4,dtype=dtype).reshape(1,1,4,4).repeat(1,4,1,1)
        c_all[0,1,:3,3]=torch.tensor([.8,.05,0. if planar else .02],dtype=dtype)
        c_all[0,2,:3,3]=torch.tensor([-.3,.4,.02],dtype=dtype)
        c_all[0,3,:3,3]=torch.tensor([.2,-.5,.1],dtype=dtype)
        theta=.08
        camera_rotation=torch.tensor([[math.cos(theta),0,math.sin(theta)],[0,1,0],[-math.sin(theta),0,math.cos(theta)]],dtype=dtype)
        c_all[0,1,:3,:3]=torch.eye(3,dtype=dtype) if planar else camera_rotation
        c_all[0,2,:3,:3]=camera_rotation.T
        c_all[0,3,:3,:3]=camera_rotation
        depth=torch.ones((1,2,1,518,518),dtype=dtype)
        if not planar:
            row_full,col_full=torch.meshgrid(torch.arange(518,dtype=dtype),torch.arange(518,dtype=dtype),indexing='ij')
            depth[:,:,0]=1.+.0008*row_full+.0012*col_full
        conf_a=torch.ones_like(depth);conf_b=torch.ones_like(depth)
        row_rank=torch.arange(518,dtype=dtype)
        conf_a[:,:,0]=.2+.8*(row_rank[None,:,None].expand(2,518,518)%29)/28
        conf_b[:,:,0]=.3+.7*(row_rank[None,None,:].expand(2,518,518)%31)/30
        x_views=[]
        for view in range(2):
            dep=depth[0,view,0,row_grid.long(),col_grid.long()].reshape(-1)
            # Axial Z depth multiplies inv(K) @ pixel directly, without unit-ray normalization.
            cam_points=(torch.linalg.inv(k)@pixels.T).T*dep[:,None]
            x_views.append(torch.einsum('ij,nj->ni',c_all[0,view,:3,:3],cam_points)+c_all[0,view,None,:3,3])
        x=torch.cat(x_views)
        angle=.35
        expected_r=torch.tensor([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]],dtype=dtype)
        expected_s=.72;expected_t=torch.tensor([.4,-.2,.15],dtype=dtype)
        y=expected_s*(x@expected_r.T)+expected_t
        if outliers:
            y[100]+=torch.tensor([.35,-.28,.2],dtype=dtype)
            y[1500]+=torch.tensor([-.3,.25,.1],dtype=dtype)
        points=torch.zeros((1,2,518,518,3),dtype=dtype)
        for view in range(2):
            points[0,view,row_grid.long(),col_grid.long()]=y[view*1369:(view+1)*1369].reshape(37,37,3)
        c_context=torch.eye(4,dtype=dtype).reshape(1,1,4,4).repeat(1,2,1,1)
        c_context[0,:,:3,:3]=torch.einsum('ij,vjk->vik',expected_r,c_all[0,:2,:3,:3])
        c_context[0,:,:3,3]=expected_s*(c_all[0,:2,:3,3]@expected_r.T)+expected_t
        return {"c2w_all":c_all,"k518_all":k_all,"depth_all":depth,"confidence_all":conf_b,
            "predicted_points":points,"confidence_context":conf_a,"c2w_context":c_context,
            "k256_context":k256,"a_scale":torch.tensor([.25],dtype=dtype),
            "median_depth":torch.tensor([1.],dtype=dtype),"expected_r":expected_r,
            "expected_s":expected_s,"expected_t":expected_t}

    def _fit_v2(self, fixture):
        args={k:v for k,v in fixture.items() if not k.startswith('expected_')}
        return align_cameras_by_shared_context_depth_v2(**args)

    def test_depth_sim3_v2_recovers_known_transform_and_robustly_rejects_outliers(self):
        for outliers in (False,True):
            f=self._v2_fixture(outliers=outliers)
            aligned,diagnostics,points=self._fit_v2(f)
            row=diagnostics[0]
            self.assertEqual(row['status'],'PASS')
            self.assertAlmostEqual(row['s'],f['expected_s'],places=5 if outliers else 8)
            self.assertTrue(torch.allclose(torch.tensor(row['R'],dtype=torch.float64),f['expected_r'],atol=2e-5 if outliers else 1e-8))
            self.assertTrue(torch.allclose(torch.tensor(row['t'],dtype=torch.float64),f['expected_t'],atol=2e-5 if outliers else 1e-8))
            self.assertEqual(tuple(aligned.shape),(1,4,4,4))
            self.assertEqual(tuple(points[0]['X_source_world'].shape),(2738,3))
            self.assertAlmostEqual(row['final_weight_sum_by_view'][0],.5,places=8)
            self.assertAlmostEqual(row['final_weight_sum_by_view'][1],.5,places=8)
            self.assertTrue(all(view['positive_z_ratio']>=.95 for view in row['views']))
            self.assertTrue(all(view['reprojection_median_px']<=4 and view['reprojection_p90_px']<=12 for view in row['views']))

    def test_depth_sim3_v2_accepts_planar_and_rejects_collinear_or_missing(self):
        _,diagnostics,_=self._fit_v2(self._v2_fixture(planar=True))
        self.assertEqual(diagnostics[0]['status'],'PASS')
        collinear=self._v2_fixture()
        pix=torch.arange(1369,dtype=torch.float64)
        line=torch.stack((pix,torch.zeros_like(pix),torch.zeros_like(pix)),-1).reshape(37,37,3)
        rows=torch.arange(7,518,14);cols=torch.arange(7,518,14)
        collinear['predicted_points'][0,:,rows[:,None],cols[None,:]]=line
        with self.assertRaisesRegex(ContextDepthSim3Error,'target points'):
            self._fit_v2(collinear)
        missing=self._v2_fixture();missing['confidence_context'][:,0]=0
        with self.assertRaisesRegex(ContextDepthSim3Error,'valid depth correspondences'):
            self._fit_v2(missing)

    def test_monitor_warning_preserves_fit_and_camera_values(self):
        fixture=self._v2_fixture()
        baseline,baseline_diag,_=self._fit_v2(fixture)
        # Only projection diagnostics change; the valid point-fit input is identical.
        fixture['k256_context']=fixture['k256_context'].clone()
        fixture['k256_context'][:,:,0,2]+=30.
        aligned,diagnostics,_=self._fit_v2(fixture)
        row=diagnostics[0]
        self.assertTrue(torch.equal(aligned,baseline))
        for key in ('R','s','t'):self.assertEqual(row[key],baseline_diag[0][key])
        self.assertEqual((row['status'],row['fit_status'],row['quality_status']),('PASS','VALID','WARNING'))
        for view in row['views']:
            self.assertGreater(view['reprojection_median_px'],4.)
            self.assertGreater(view['reprojection_p90_px'],12.)
            self.assertIn('reprojection_median_above_reference',view['quality_warning_reasons'])
            self.assertIn('reprojection_p90_above_reference',view['quality_warning_reasons'])
        fixture['c2w_context'][:,:,2,3]+=10.
        _,diagnostics,_=self._fit_v2(fixture)
        for view in diagnostics[0]['views']:
            self.assertIn('positive_z_below_reference',view['quality_warning_reasons'])
            self.assertIn('reprojection_metric_unavailable',view['quality_warning_reasons'])
            self.assertIsNone(view['reprojection_median_px'])

    def test_monitor_retains_only_two_distinct_warning_window_npz_files(self):
        import json
        from scripts.object_locus_frozen_vggt_posefree_runtime import GeometryMonitor
        fixture=self._v2_fixture();fixture['k256_context'][:,:,0,2]+=30.
        cameras,diagnostics,points=self._fit_v2(fixture)
        calibration={'diagnostics':diagnostics,'point_records':points,'c2w':cameras,
            'intrinsics_matrix':fixture['k256_context']}
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',{'POSEFREE_V2_EVIDENCE_DIR':directory}):
            monitor=GeometryMonitor(Path(directory)/'smoke',0,'eight_smoke')
            for index in (6923,6923,100,101):
                monitor.record(calibration,1,{'manifest_index':index})
            self.assertEqual(len(list(Path(directory).rglob('*.npz'))),2)
            examples=[json.loads(p.read_text()) for p in Path(directory).rglob('calibration_warning.json')]
            self.assertEqual({r['monitor']['manifest_index'] for r in examples},{6923,100})
            self.assertEqual(monitor.counts['WARNING'],4)
            self.assertEqual(len((Path(directory)/'smoke/geometry_monitor_rank0.jsonl').read_text().splitlines()),4)

    def test_monitor_still_rejects_invalid_sim3_and_cameras(self):
        import tokengs.models.object_locus_posefree_geometry as geometry
        original=geometry._weighted_fit_normalized
        for invalid_scale in (-1.,float('nan'),float('inf')):
            calls=0
            def invalid(*args):
                nonlocal calls
                calls+=1
                r,scale,t=original(*args)
                return r,(scale.new_tensor(invalid_scale) if calls==6 else scale),t
            with patch.object(geometry,'_weighted_fit_normalized',side_effect=invalid):
                with self.assertRaisesRegex(ContextDepthSim3Error,'nonfinite, nonpositive'):
                    self._fit_v2(self._v2_fixture())
        fixture=self._v2_fixture();fixture['c2w_all'][0,3]=0
        with self.assertRaisesRegex(ContextDepthSim3Error,'noninvertible'):
            self._fit_v2(fixture)
        fixture=self._v2_fixture();fixture['c2w_all'][0,3,0,0]=float('nan')
        with self.assertRaisesRegex(ContextDepthSim3Error,'nonfinite inputs'):
            self._fit_v2(fixture)

    def test_old_alignment_diagnostic_reports_signed_scale_and_angle_without_raising(self):
        f=self._v2_fixture()
        # Reverse the context-only baseline while keeping each camera finite.
        pair=f['c2w_context'][0].clone()
        pair[0,:3,3],pair[1,:3,3]=pair[1,:3,3].clone(),pair[0,:3,3].clone()
        row=old_shared_context_alignment_diagnostics(f['c2w_all'],pair[None])[0]
        self.assertTrue(row['all_finite'])
        self.assertGreater(row['theta_degrees'],90.)
        self.assertAlmostEqual(row['s_old'],row['centered_formula_scale'],places=10)
        self.assertAlmostEqual(row['numerator']/row['denominator'],
            row['centered_formula_numerator']/row['centered_formula_denominator'],places=10)
        nonfinite=f['c2w_all'].clone();nonfinite[0,2,1,2]=float('nan')
        invalid=old_shared_context_alignment_diagnostics(nonfinite,f['c2w_context'])[0]
        self.assertFalse(invalid['all_finite'])
        self.assertIn([2,1,2],invalid['nonfinite_c2w_all_indices'])


class MigrationAndTrainingContracts(unittest.TestCase):
    def test_fixed_full1201_manifest_is_used_without_resampling(self):
        _,scenes,windows=load_manifest()
        self.assertEqual((len(scenes),len(windows)),(1191,8337))
        self.assertEqual((len(windows[0]['context']),len(windows[0]['novel'])),(2,2))

    def test_eight_rank_launcher_and_resume_recipe_are_locked(self):
        submit=(REPO/'scripts/submit_object_locus_frozen_vggt_posefree_v1.sh').read_text()
        job=(REPO/'scripts/run_object_locus_frozen_vggt_posefree_v1_job.sh').read_text()
        plan=__import__('scripts.object_locus_frozen_vggt_posefree_runtime',fromlist=['plan_record']).plan_record()
        self.assertIn('--nodelist=3dimage-13',submit)
        self.assertIn('--gres=gpu:8',submit)
        self.assertIn('--cpus-per-task=32',submit)
        self.assertIn('--mem=128G',submit)
        self.assertIn('--time=48:00:00',submit)
        self.assertIn('--nproc_per_node=8',job)
        self.assertIn('--eight-card-real',job)
        self.assertIn('--window-4253-calibration',job)
        self.assertIn('--calibration-report',job)
        self.assertIn('--single-smoke-report',job)
        self.assertIn('--eight-smoke-report',job)
        self.assertEqual((plan['world_size'],plan['microbatch_per_rank'],plan['accumulation'],plan['global_batch']),(8,1,1,8))
        self.assertEqual((plan['updates_per_epoch'],plan['total_updates'],plan['total_exposures']),(1043,8344,66752))

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
        epoch,update=2,4
        shards=[rank_microbatch_indices(epoch,update,r) for r in range(8)]
        expected=epoch_order(epoch)[update*8:(update+1)*8]
        self.assertEqual([shard[0] for shard in shards],expected.tolist())
        self.assertEqual(len({i for shard in shards for i in shard}),8)
        smoke_first16=[rank_microbatch_indices(0,u,r)[0] for u in range(2) for r in range(8)]
        self.assertEqual(smoke_first16,epoch_order(0)[:16].tolist())
        self.assertEqual((WORLD_SIZE,GLOBAL_BATCH,ACCUMULATION),(8,8,1))
        self.assertEqual(smoke_exposure_count('single',2),2)
        self.assertEqual(smoke_exposure_count('eight',2),16)
        self.assertEqual(training_configuration()['total_updates'],8344)
        params=[torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
        grads=gc_combine(['reconstruction.weight','understanding.weight','panoptic.weight'],
            [torch.ones(1)]*3,[torch.full((1,),2.)]*3)
        self.assertTrue(torch.allclose(grads[0],torch.tensor([1.02])))
        self.assertTrue(torch.allclose(grads[1],torch.tensor([3.])))
        self.assertTrue(torch.allclose(grads[2],torch.tensor([3.])))

    def test_single_microbatch_accumulation_and_exposure_clock(self):
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
        row=train_microbatch_window(model,optimizer,[torch.ones(1,1)],0,base_exposure=0)
        self.assertEqual(model.steps,[(0,0.)])
        self.assertEqual(row['completed_updates'],1); self.assertEqual(row['completed_exposures'],1)
        self.assertEqual(row['understanding_weight'],0.); self.assertEqual(row['beta'],0.)

    def test_monitor_serialization_does_not_change_loss_gradient_or_exposure(self):
        import json
        import copy
        from scripts.object_locus_frozen_vggt_posefree_runtime import GeometryMonitor, _write_json_atomic
        view={'valid_count':1369,'positive_z_ratio':1.,'reprojection_median_px':5.,'reprojection_p90_px':20.,
            'residual_3d_median_over_scene_median_depth':.1,'residual_3d_p90_over_scene_median_depth':.2,
            'residual_3d_rmse_over_scene_median_depth':.15,
            'quality_warning_reasons':['reprojection_median_above_reference'],
            'diagnostic_unavailable_reasons':{}}
        calibration={'diagnostics':[{'s':.72,'fit_status':'VALID','quality_status':'WARNING','views':[view,view]}]}
        class TinyTrain(nn.Module):
            def __init__(self):
                super().__init__();self.reconstruction=nn.Linear(1,1,bias=False)
            def step_loss(self,batch,*,step,understanding_weight):
                rec=self.reconstruction(batch).square().mean()
                return {'prediction':{'target_camera_calibration':calibration}},{'loss_recon':rec,'loss_understanding':rec*.1}
        from scripts.object_locus_frozen_vggt_posefree_runtime import build_optimizer
        model=TinyTrain();reference=copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ',{'POSEFREE_V2_EVIDENCE_DIR':''}):
            monitor=GeometryMonitor(directory,0,'eight_smoke')
            row=train_microbatch_window(model,build_optimizer(model),[torch.ones(1,1)],1,base_exposure=8,world_size=8,
                monitor=monitor,window_metadata=[{'manifest_index':6923,'scene':'scene0563_00','context_ids':[145,197],'novel_ids':[151,185]}],stage='eight_smoke')
            expected=train_microbatch_window(reference,build_optimizer(reference),[torch.ones(1,1)],1,base_exposure=8,world_size=8)
            self.assertEqual(row,expected)
            for a,b in zip(model.parameters(),reference.parameters()):
                self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a.grad,b.grad))
            saved=json.loads((Path(directory)/'geometry_monitor_rank0.jsonl').read_text())
            self.assertEqual(saved['stage'],'eight_smoke');self.assertEqual(saved['manifest_index'],6923)
            self.assertEqual(monitor.counts['WARNING'],1)
            _write_json_atomic(Path(directory)/'unavailable.json',{'statistic':float('nan'),'reason':'nonfinite_diagnostic_statistic'})
            self.assertIsNone(json.loads((Path(directory)/'unavailable.json').read_text())['statistic'])

    def test_gc_eight_rank_one_sample_average_matches_eight_sample_reference(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.shared=nn.Parameter(torch.tensor(.2));self.understanding_head=nn.Parameter(torch.tensor(.3));self.panoptic=nn.Parameter(torch.tensor(.1));self.vggt_memory_adapter=nn.Parameter(torch.tensor(.15));self.unused=nn.Parameter(torch.tensor(.4));self.frozen_vggt=nn.Parameter(torch.tensor(.7),requires_grad=False)
            def step_loss(self,batch,*,step,understanding_weight):
                x,y=batch
                rec=.01*((self.shared*x+self.panoptic*y)**2).mean()
                under=.01*((self.shared*x+self.understanding_head*y+self.panoptic*x+self.vggt_memory_adapter*y)**2).mean()
                return {},{'loss_recon':rec,'loss_understanding':under}
        from scripts.object_locus_frozen_vggt_posefree_runtime import build_optimizer
        samples=[(torch.tensor([.2+i*.01]),torch.tensor([.5-i*.01])) for i in range(8)]
        rank_grads=[]
        for rank in range(8):
            model=Toy();opt=__import__('scripts.object_locus_frozen_vggt_posefree_runtime',fromlist=['build_optimizer']).build_optimizer(model)
            pair=[samples[rank]]
            train_microbatch_window(model,opt,pair,0,base_exposure=100,world_size=8)
            rank_grads.append({n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.named_parameters()})
        names=['shared','understanding_head','panoptic','vggt_memory_adapter','unused']
        averaged={}
        for name in names:
            vals=[row[name] for row in rank_grads if row[name] is not None]
            averaged[name]=sum(vals)/8 if vals else None
        reference=Toy();params=[p for n,p in reference.named_parameters() if n in names]
        rec_total=0.;under_total=0.
        for x,y in samples:
            rec_total=rec_total+.01*((reference.shared*x+reference.panoptic*y)**2)/8
            under_total=under_total+.01*((reference.shared*x+reference.understanding_head*y+reference.panoptic*x+reference.vggt_memory_adapter*y)**2)/8
        rec=torch.autograd.grad(rec_total,params,retain_graph=True,allow_unused=True)
        under=torch.autograd.grad(.5*under_total,params,allow_unused=True)
        combined=gc_combine(names,rec,under)
        for name,expected in zip(names,combined):
            actual=averaged[name]
            if expected is None:self.assertIsNone(actual)
            else:self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=1e-5),(name,actual,expected))
        self.assertIsNone(averaged['unused'])
        self.assertFalse(any(n.startswith('frozen_vggt') for n in dict(reference.named_parameters()) if n in [name for g in build_optimizer(reference).param_groups for name in g['param_names']]))

    def test_small_checkpoint_restore_preserves_optimizer_clock_and_next_rng(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.reconstruction=nn.Linear(2,1);self.frozen_vggt=nn.Linear(2,2);self.frozen_vggt.requires_grad_(False)
        torch.manual_seed(314)
        model=Tiny();optimizer=__import__('scripts.object_locus_frozen_vggt_posefree_runtime',fromlist=['build_optimizer']).build_optimizer(model)
        loss=model.reconstruction(torch.ones(1,2)).square().mean();loss.backward();optimizer.step()
        rank_rng=[capture_rank_rng() for _ in range(8)]
        payload={'model':checkpoint_model_state(model),'optimizer':optimizer.state_dict(),
            'completed_updates':9,'completed_exposures':72,'rank_rng':rank_rng,'world_size':8,
            'config':training_configuration()}
        expected_rng=payload['rank_rng'][0]
        self.assertEqual(len(payload['rank_rng']),8)
        expected_next=torch.rand(5)
        torch.manual_seed(9)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint_latest.pt';_atomic_save(payload,path)
            restored=Tiny();restored_optimizer=__import__('scripts.object_locus_frozen_vggt_posefree_runtime',fromlist=['build_optimizer']).build_optimizer(restored)
            loaded=torch.load(path,map_location='cpu',weights_only=False)
            restore_model_state_strict(restored,loaded['model']);restored_optimizer.load_state_dict(loaded['optimizer'])
            restore_rank_rng(expected_rng)
            self.assertTrue(torch.equal(torch.rand(5),expected_next))
            self.assertEqual((loaded['completed_updates'],loaded['completed_exposures']),(9,72))
            self.assertTrue(all(torch.equal(model.state_dict()[k],restored.state_dict()[k]) for k in model.state_dict() if not k.startswith('frozen_vggt.')))
            for old,new in zip(optimizer.state.values(),restored_optimizer.state.values()):
                self.assertEqual(old.keys(),new.keys())
                for key in old:
                    if torch.is_tensor(old[key]):self.assertTrue(torch.equal(old[key],new[key]))


if __name__=='__main__': unittest.main(verbosity=2)
