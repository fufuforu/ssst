import copy,unittest
from unittest.mock import patch
import torch
from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
from tokengs.models.superpoint_structural_variants import LocusGSSerialQuery,MODELS
from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1

class GeometryRenderer(FakeRenderer):
    def render_feature_channels(self,g,features,cam_view,intrinsics):
        self.understanding_geometry_requires_grad=g.requires_grad
        result=super().render_feature_channels(g,features,cam_view,intrinsics)
        result['images_pred']=result['images_pred']*(.5+g[...,:3].mean().sigmoid())
        return result

class SerialTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)
    def model(self):return LocusGSSerialQuery(opt(),config=config(),omega=FakeOmega(),sam=FakeSAM(),renderer=GeometryRenderer())

    def test_reconstruction_unchanged_with_common_weights_and_query_identity(self):
        old=LocusGSUnifiedQueryV1(opt(),config=config(),omega=FakeOmega(),sam=FakeSAM(),renderer=GeometryRenderer())
        new=self.model();missing,unexpected=new.load_state_dict(old.state_dict(),strict=False)
        self.assertFalse(unexpected);self.assertTrue(all(k.startswith(('core.semantic_memory.','core.semantic_seed.','core.semantic_blocks.')) for k in missing))
        rgb=torch.rand(1,2,3,256,256);a=old.generate(rgb);b=new.generate(rgb)
        torch.testing.assert_close(a['gaussians'],b['gaussians'],rtol=0,atol=0)
        torch.testing.assert_close(a['unified_query'],b['reconstruction_query'],rtol=0,atol=0)
        self.assertTrue(torch.equal(a['superpoints']['source_superpoint_ids'],b['superpoints']['source_superpoint_ids']))
        self.assertEqual(b['understanding_query'].shape,b['reconstruction_query'].shape)
        torch.testing.assert_close(b['semantic_query_initial'],torch.nn.functional.layer_norm(b['reconstruction_query'].detach(),(config().dim,)),rtol=1e-5,atol=1e-6)
        self.assertIs(MODELS['serial_query'],LocusGSSerialQuery)

    def test_understanding_gradient_cannot_reach_reconstruction_or_gaussians(self):
        m=self.model();out=m.generate(torch.rand(1,2,3,256,256));out['gaussians'].retain_grad()
        loss=out['thing_logits19'].square().mean()+out['region_mass'].square().mean()
        loss.backward();groups=m.parameter_groups()
        self.assertFalse(groups['shared']);self.assertTrue(all(p.grad is None for p in groups['reconstruction']))
        self.assertIsNone(out['gaussians'].grad);self.assertFalse(m.gs.understanding_geometry_requires_grad)
        for module in [m.core.semantic_seed,m.core.semantic_blocks,m.core.semantic_memory,m.core.classifier,m.core.child_feature]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for p in m.frozen_omega.parameters()))

    def test_reconstruction_gradient_is_independent_of_semantic_path_and_clip(self):
        a=self.model();b=copy.deepcopy(a);rgb=torch.rand(1,2,3,256,256)
        ao=a.generate(rgb);bo=b.generate(rgb)
        ao['gaussians'].square().mean().backward()
        (bo['gaussians'].square().mean()+100*bo['thing_logits19'].square().mean()+bo['region_mass'].mean()).backward()
        for model in [a,b]:
            for ps in model.parameter_groups().values():
                if ps:torch.nn.utils.clip_grad_norm_(ps,1.,error_if_nonfinite=True)
        for pa,pb in zip(a.parameter_groups()['reconstruction'],b.parameter_groups()['reconstruction']):
            self.assertEqual(pa.grad is None,pb.grad is None)
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in a.parameter_groups()['understanding']))
        ids=[id(p) for ps in b.parameter_groups().values() for p in ps]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{id(p) for p in b.parameters() if p.requires_grad})

    def test_actual_step_loss_renders_masks_without_reconstruction_gradient(self):
        model=self.model();rgb=torch.rand(1,2,3,256,256)
        batch=dict(images_input=rgb,images_all=rgb,semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long),has_mask=False,masks_all=torch.ones(1,2,1,256,256))
        def calibration(images,p):return dict(c2w=torch.eye(4).expand(1,2,4,4),intrinsics=torch.ones(1,2,4),intrinsics_matrix=torch.eye(3).expand(1,2,3,3))
        def objective(states,decoder,supervision):
            value=model.activation_head(states[-1]['tokens'],states[-1]['mu'],states[-1]['radii']).square().mean()
            return value,{f'psnr_layer{model.supervised_layers[-1]}':value.detach()*0},None,None,None
        with patch.object(model,'calibrate_targets',side_effect=calibration),patch.object(model,'_layer_objective',side_effect=objective):
            _,metrics=model.step_loss(batch,understanding_weight=.05)
            model.zero_grad(set_to_none=True)
            metrics['loss_understanding'].backward()
            self.assertTrue(all(p.grad is None for p in model.parameter_groups()['reconstruction']))
            self.assertTrue(any(p.grad is not None for p in model.parameter_groups()['understanding']))

if __name__=='__main__':unittest.main()
