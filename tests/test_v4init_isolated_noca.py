import unittest
from dataclasses import replace
from unittest.mock import patch
import torch
from torch import nn
from tokengs.models.v4init_isolated_noca import LocusGSV4InitIsolatedNoCA
from tokengs.models.superpoint_dual_core import SuperpointDualCore
from scripts.v4init_isolated_recipe import optimizer_groups

class V4IsolationTests(unittest.TestCase):
    def test_real_core_forward_equivalence_and_both_loss_directions(self):
        from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
        from tokengs.models.unified_query_loss import unified_instance_loss
        from tokengs.models.input_types import ModelInputDecoder,ModelSupervision
        cfg=replace(config(),dim=256,heads=8,depth=12,superpoint_source='esam_sam',matching_mode='source_many_to_one')
        torch.manual_seed(42)
        model=LocusGSV4InitIsolatedNoCA(opt().evolve(dec_patch_size=8,img_size=(8,8)),config=cfg,omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer())
        self.assertEqual(model.supervised_layers,(6,12));self.assertEqual(model.layer_weights,[1/3,2/3])
        self.assertFalse(hasattr(model.core,'understanding_adapters'))
        for layer in model.core.panoptic.layers.values():
            self.assertFalse(hasattr(layer,'anchor_attention'))
            nn.init.normal_(layer.image_attention.O.weight,std=.01)
        nn.init.normal_(model.core.panoptic.child_mlp[-1].weight,std=.01)
        def small_memory(layers,rgb):
            m=model.core.semantic_memory
            patches=sum(w*proj(x.float().mean(-2)) for w,proj,x in zip(m.layer_logits.softmax(0),m.projections,layers))
            grid=patches.reshape(2,256,1,1).expand(-1,-1,8,8)
            return m.refine(grid+m.local(torch.nn.functional.interpolate(rgb.flatten(0,1),size=(8,8),mode='bilinear',align_corners=False))).reshape(1,2,256,8,8)
        original=model.core.initialize_reconstruction_anchors
        def reduced(*args,**kwargs):
            result=original(*args,**kwargs)
            for k in ('tokens','centers','radii'):result[k]=result[k][:,:6]
            for k in ('anchor_to_superpoint','anchor_slot_ids','anchor_point_indices'):result[k]=result[k][:6]
            return result
        rgb=torch.rand(1,2,3,256,256)
        with patch.object(model.core.semantic_memory,'forward',side_effect=small_memory),patch.object(model.core,'initialize_reconstruction_anchors',side_effect=reduced):
            out=model.generate(rgb)
            with torch.no_grad(),patch.object(model.core,'forward',side_effect=lambda *a,**kw:SuperpointDualCore.forward(model.core,*a,**kw)):
                joint=model.generate(rgb)
            for key in ('gaussians','gaussian_membership','p_class','q_pre','unified_query','region_mass'):
                torch.testing.assert_close(out[key],joint[key],rtol=0,atol=0,msg=key)
            for x,y in zip(out['states'],joint['states']):
                for k in ('tokens','mu','rho','radii'):torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
            groups=model.parameter_groups();ids=[id(p) for ps in groups.values() for p in ps]
            self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})
            def energy(ps):return sum(float(p.grad.square().sum()) for p in ps if p.grad is not None)
            batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
            lu,_=unified_instance_loss(out,batch,cfg);lu.backward(retain_graph=True)
            self.assertEqual(energy(groups['reconstruction']),0.)
            for g in ('understanding','shared'):self.assertGreater(energy(groups[g]),0.,g)
            model.zero_grad(set_to_none=True)
            decoder=ModelInputDecoder(cam_view=torch.eye(4).expand(1,2,4,4),intrinsics=torch.tensor([4.,4.,4.,4.]).expand(1,2,4))
            supervision=ModelSupervision(images_output=torch.rand(1,2,3,8,8),masks_output=torch.ones(1,2,1,8,8),has_mask=torch.tensor([False]),rays_os=torch.zeros(1,2,3,8,8),rays_ds=torch.ones(1,2,3,8,8))
            lr,*_=model._layer_objective(out['states'],decoder,supervision);lr.backward()
            self.assertGreater(energy(groups['reconstruction']),0.)
            for g in ('understanding','shared'):self.assertEqual(energy(groups[g]),0.,g)
            for name,p in model.core.v4_initialization.named_parameters():self.assertGreater(float(p.grad.norm()),0.,name)
            specs=optimizer_groups(model)
            for group in specs:
                expected=1e-5 if group['family']=='geometry' else 1e-4
                self.assertEqual(group['base_lr'],expected)
                for name in group['parameter_names']:
                    self.assertEqual(name.startswith(('core.anchor_decoder.refine_mu.','core.anchor_decoder.refine_rho.')),group['family']=='geometry')
            self.assertEqual(len({id(p) for g in specs for p in g['params']}),len(ids))

    def test_schedule_keeps_geometry_ratio_and_warmup(self):
        from scripts.dual_stream_training_policy import lr_factor
        for step in (0,1,99,100,1042,4171):
            f=lr_factor('full',step,4172)
            self.assertAlmostEqual((1e-5*f)/(1e-4*f),.1)
        self.assertEqual(lr_factor('full',0,4172),.01)

if __name__=='__main__':unittest.main()
