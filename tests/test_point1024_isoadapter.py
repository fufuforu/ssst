import copy
import io
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import torch
from torch import nn
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(Path(__file__).resolve().parents[1],log=lambda *args:None)
from tokengs.models.point1024_isoadapter_core import DetachedResidualAdapter
from tokengs.models.point1024_isoadapter_model import LocusGSPoint1024IsolatedAdapter
from tokengs.models.gfm_dual_stream_components import sample_anchor_gfm
from tokengs.models.unified_query_loss import unified_instance_loss


class AdapterTests(unittest.TestCase):
    def test_identity_input_isolation_learning_and_checkpoint(self):
        torch.manual_seed(1);adapter=DetachedResidualAdapter(dim=8,hidden=3)
        x=torch.randn(1,5,8,requires_grad=True)
        torch.testing.assert_close(adapter(x),x,rtol=0,atol=0)
        before=copy.deepcopy(adapter.state_dict());opt=torch.optim.SGD(adapter.parameters(),lr=.1)
        adapter(x).square().mean().backward()
        self.assertIsNone(x.grad);self.assertGreater(float(adapter.net[-1].weight.grad.norm()),0)
        # At exact-zero output initialization hidden weights receive zero gradient.
        self.assertEqual(float(adapter.net[0].weight.grad.norm()),0)
        opt.step();self.assertFalse(torch.equal(before['net.2.weight'],adapter.net[-1].weight))
        opt.zero_grad();adapter(x).square().mean().backward()
        self.assertGreater(float(adapter.net[0].weight.grad.norm()),0)
        self.assertIsNone(x.grad)
        buf=io.BytesIO();torch.save(adapter.state_dict(),buf);buf.seek(0)
        restored=DetachedResidualAdapter(dim=8,hidden=3);restored.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(adapter(x),restored(x))

    def test_source_pixel_sampling_preserves_view_and_pixel_identity(self):
        # Values identify the source view, x and y; dimensions intentionally differ
        # from fused cloud indices. This must not become nearest-cloud gathering.
        grid=torch.arange(2*4*4.).reshape(1,2,1,4,4).requires_grad_()
        indices=torch.tensor([[0,15,16,31]])
        sampled=sample_anchor_gfm(grid,indices,image_hw=(4,4))
        torch.testing.assert_close(sampled[0,:,0],torch.tensor([0.,15.,16.,31.]))
        sampled.sum().backward();self.assertEqual(int(torch.count_nonzero(grid.grad)),4)


class RoutingTests(unittest.TestCase):
    def test_real_core_both_actual_losses_respect_parameter_boundaries(self):
        from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
        from tokengs.models.input_types import ModelInputDecoder,ModelSupervision
        cfg=replace(config(),dim=256,heads=8,depth=12,superpoint_source='esam_sam',matching_mode='source_many_to_one')
        model=LocusGSPoint1024IsolatedAdapter(opt().evolve(dec_patch_size=8,img_size=(8,8)),config=cfg,
            omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer())
        self.assertEqual(model.reconstruction_count,1024)
        self.assertEqual(model.supervised_layers,(6,12));self.assertEqual(model.layer_weights,[1/3,2/3])
        self.assertFalse(hasattr(model.core.initializer,'reconstruction'))
        self.assertEqual(sum(p.numel() for p in model.core.understanding_adapters.parameters()),528640)
        self.assertFalse(any('learned_embedding' in name or 'gs_tokens' in name for name,_ in model.named_parameters()))
        # Nonzero readout/geometry weights expose leaks hidden by baseline's zero
        # initial readout projections and zero child residual head.
        for layer in model.core.panoptic.layers.values():
            nn.init.normal_(layer.anchor_attention.O.weight,std=.01)
        nn.init.normal_(model.core.panoptic.child_mlp[-1].weight,std=.01)
        def small_memory(layers,rgb):
            module=model.core.semantic_memory;b,v=rgb.shape[:2]
            patches=sum(w*proj(x.float().mean(-2)) for w,proj,x in zip(module.layer_logits.softmax(0),module.projections,layers))
            grid=patches.reshape(b*v,256,1,1).expand(-1,-1,8,8)
            local=module.local(torch.nn.functional.interpolate(rgb.flatten(0,1),size=(8,8),mode='bilinear',align_corners=False))
            return module.refine(grid+local).reshape(b,v,256,8,8)
        rgb=torch.rand(1,2,3,256,256)
        with patch.object(model.core.semantic_memory,'forward',side_effect=small_memory):
            # Actual production1024 seed path, without full65536-GS CPU rendering.
            omega,cloud,memory=model.encode_context(rgb)
            seed=memory['point_seed'];self.assertEqual(seed['features'].shape,(1,1024,256))
            self.assertEqual(seed['centers'].shape,(1,1024,3))
            torch.testing.assert_close(seed['centers'],omega['points'].reshape(1,-1,3)[:,seed['source_indices'][0]])
            model.reconstruction_count=6  # Only the full CPU integration render is reduced.
            out=model.generate(rgb)
            self.assertEqual(out['num_reconstruction_anchors'],6)
            self.assertEqual(out['gaussians'].shape,(1,384,14));self.assertEqual(out['unified_query'].shape,(1,2,256))
            # Understanding seeds are numerically unchanged, but feature gradient
            # is stopped before their trainable initializer.
            with torch.no_grad():expected=model.core.initializer.understanding(out['cloud'])
            torch.testing.assert_close(out['q_pre'],expected['query'])
            batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
            lu,_=unified_instance_loss(out,batch,cfg)
            decoder=ModelInputDecoder(cam_view=torch.eye(4).expand(1,2,4,4),intrinsics=torch.tensor([4.,4.,4.,4.]).expand(1,2,4))
            supervision=ModelSupervision(images_output=torch.rand(1,2,3,8,8),masks_output=torch.ones(1,2,1,8,8),has_mask=torch.tensor([False]),
                rays_os=torch.zeros(1,2,3,8,8),rays_ds=torch.ones(1,2,3,8,8))
            lr,*_=model._layer_objective(out['states'],decoder,supervision)
            named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
            params=[p for _,p in named]
            gu=torch.autograd.grad(lu,params,allow_unused=True,retain_graph=True)
            gr=torch.autograd.grad(lr,params,allow_unused=True)
            group_ids={k:{id(p) for p in ps} for k,ps in model.parameter_groups().items()}
            self.assertEqual(sum(len(x) for x in group_ids.values()),len(params))
            for (name,p),u,r in zip(named,gu,gr):
                if id(p) in group_ids['reconstruction']|group_ids['shared']:
                    self.assertTrue(u is None or torch.count_nonzero(u)==0,'LU leaked to '+name)
                else:
                    self.assertTrue(r is None or torch.count_nonzero(r)==0,'LR leaked to '+name)
            def norm(grads,prefix):return sum(float(g.norm()) for (name,_),g in zip(named,grads) if name.startswith(prefix) and g is not None)
            self.assertGreater(norm(gr,'gfm_projection.'),0)
            for prefix in ('core.blocks.','core.reconstruction_seed_projection.','core.reconstruction_memory.','core.activation_head.'):
                self.assertGreater(norm(gr,prefix),0,prefix)
            for prefix in ('core.initializer.understanding.','core.panoptic.','core.semantic_memory.'):
                self.assertGreater(norm(gu,prefix),0,prefix)
            for layer in (6,8,10,12):
                self.assertGreater(norm(gu,f'core.understanding_adapters.L{layer}.net.2.'),0)
            # Identity adapters must give exactly the same forward values as
            # detached, adapter-free reads on the same weights and input.
            with torch.no_grad():
                with patch.object(DetachedResidualAdapter,'forward',lambda self,x:x.detach()):control=model.generate(rgb)
            for key in ('gaussians','p_class','gaussian_membership','semantic_scores'):
                torch.testing.assert_close(out[key],control[key],rtol=1e-6,atol=1e-7)



class MigrationTests(unittest.TestCase):
    def test_exact_migration_and_rejection_rules(self):
        from types import SimpleNamespace
        from tokengs.models.unified_superpoints import UnifiedQueryConfig,SuperpointQueryInitializer
        from scripts.point1024_isoadapter_initial_state import initialize_point1024_isoadapter_from_legacy_zero,ARCH,REMOVED,ADAPTERS
        model=nn.Module();model.architecture_name=ARCH;model.config=UnifiedQueryConfig()
        model.frozen_omega=SimpleNamespace(source_identity={'omega':True});model.sam_proposals=None
        model.core=nn.Module();model.core.baseline=nn.Linear(2,2)
        model.core.understanding_adapters=nn.ModuleDict({f'L{i}':DetachedResidualAdapter() for i in (6,8,10,12)})
        source={k:torch.full_like(v,.3) for k,v in model.state_dict().items() if not k.startswith(ADAPTERS)}
        source.update({REMOVED+k:v for k,v in SuperpointQueryInitializer(model.config).state_dict().items()})
        saved=dict(model=source,meta=dict(architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config=model.config.to_dict(),global_batch=8,world_size=8,manifest_sha256='train',heldout_sha256='val',omega_identity={'omega':True},sam_identity=None),
            progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer=dict(state={}),rank_rng=[{}]*8)
        def apply(blob):return initialize_point1024_isoadapter_from_legacy_zero(blob,model,manifest_sha256='train',heldout_sha256='val',world=8)
        before={k:v.clone() for k,v in model.state_dict().items() if k.startswith(ADAPTERS)}
        result=apply(saved);self.assertEqual(result['copied_tensor_count'],2)
        for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v)
        for case in ('data','trained','adam','missing_retained','missing_removed','shape','extra','rng'):
            bad=copy.deepcopy(saved)
            if case=='data':bad['meta']['heldout_sha256']='bad'
            if case=='trained':bad['progress']['full_updates']=1
            if case=='adam':bad['optimizer']['state'][0]={'step':1}
            if case=='missing_retained':del bad['model']['core.baseline.bias']
            if case=='missing_removed':del bad['model'][next(k for k in source if k.startswith(REMOVED))]
            if case=='shape':bad['model']['core.baseline.weight']=torch.zeros(1)
            if case=='extra':bad['model']['unexpected']=torch.zeros(1)
            if case=='rng':bad['rank_rng']=[{}]
            with self.subTest(case=case):
                with self.assertRaises(ValueError):apply(bad)

if __name__=='__main__':unittest.main()
