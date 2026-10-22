import unittest
from dataclasses import replace
import torch
from scripts.v4init_isolated_recipe import KEY,LOW_KEY,VIEWS_KEY,LOW_WEIGHT,recipe_identity

class WeightViewsTests(unittest.TestCase):
    def test_each_arm_has_only_one_recipe_change(self):
        base=recipe_identity();low=recipe_identity(LOW_KEY);views=recipe_identity(VIEWS_KEY)
        self.assertEqual({k for k in base if base[k]!=low[k]},{'understanding_weight'})
        self.assertEqual(low['understanding_weight'],.05)
        self.assertEqual({k for k in base if base[k]!=views[k]},{'training_view_protocol'})
        self.assertEqual(views['understanding_weight'],1.)

    def test_six_view_rgb_supervision_includes_added_novel(self):
        from tests.test_unified_query_v1 import opt,config,FakeOmega,FakeRenderer
        from tokengs.models.unified_query_v1 import LocusGSUnifiedQueryV1
        from tokengs.models.canonical_recon_models import _full_supervision
        from tokengs.models.input_types import ModelInputDecoder
        torch.manual_seed(42)
        model=LocusGSUnifiedQueryV1(opt().evolve(img_size=(8,8),num_views=6),config=config(),omega=FakeOmega(),renderer=FakeRenderer())
        out=model.generate(torch.rand(1,2,3,256,256))
        decoder=ModelInputDecoder(cam_view=torch.eye(4).expand(1,6,4,4),intrinsics=torch.tensor([4.,4.,4.,4.]).expand(1,6,4))
        batch=dict(images_all=torch.zeros(1,6,3,8,8),masks_all=torch.ones(1,6,1,8,8),has_mask=torch.tensor([False]),rays_os=torch.zeros(1,6,3,8,8),rays_ds=torch.ones(1,6,3,8,8))
        supervision=_full_supervision(batch);self.assertEqual(supervision.images_output.shape[1],6)
        a,*_=model._layer_objective(out['states'],decoder,supervision)
        batch['images_all'][:,4:]=1.
        b,*_=model._layer_objective(out['states'],decoder,_full_supervision(batch))
        self.assertNotEqual(float(a),float(b));b.backward()
        self.assertGreater(float(model.gfm_projection[1].weight.grad.norm()),0.)

    def test_extra_novel_gt_cannot_change_understanding_and_weight_scales_gradients(self):
        from tests.test_unified_query_v1 import config
        from tokengs.models.unified_query_loss import unified_instance_loss
        cfg=replace(config(),matching_mode='source_many_to_one')
        n=4;regions=torch.full((1,2,n+2,2,2),.6,requires_grad=True);logits=torch.zeros(1,n,19,requires_grad=True)
        pred=dict(thing_logits19=logits,region_mass=regions,num_thing_queries=n,cloud=dict(source_indices=torch.arange(8)),superpoints=dict(source_superpoint_ids=torch.tensor([0,1,2,3,0,1,2,3])))
        batch=dict(semantic_label_all=torch.full((1,6,2,2),2),instance_label_all=torch.ones(1,6,2,2,dtype=torch.long))
        a,_=unified_instance_loss(pred,batch,cfg)
        batch['semantic_label_all'][:,2:]=19;batch['instance_label_all'][:,2:]=999
        b,_=unified_instance_loss(pred,batch,cfg);torch.testing.assert_close(a,b,rtol=0,atol=0)
        r=torch.tensor(2.,requires_grad=True)
        high=torch.autograd.grad(r.square()+a,(r,logits,regions),retain_graph=True)
        low=torch.autograd.grad(r.square()+LOW_WEIGHT*a,(r,logits,regions),retain_graph=True)
        torch.testing.assert_close(high[0],low[0],rtol=0,atol=0)
        for x,y in zip(high[1:],low[1:]):torch.testing.assert_close(x*.05,y)

    def test_smoke_batch_protocol_accepts_two_or_four_novel_and_rejects_wrong_shapes(self):
        from scripts.smoke_v4init_weight_views import validate_batch_contract
        for nv in (2,4):
            w=dict(context=[0,10],novel=list(range(1,nv+1)));total=2+nv
            b=dict(images_input=torch.empty(1,2,3,256,256),images_all=torch.empty(1,total,3,256,256),images_output=torch.empty(1,nv,3,256,256),frame_ids=torch.tensor([w['context']+w['novel']]))
            self.assertEqual(validate_batch_contract(b,w)['supervised_views'],total)
            b['images_output']=torch.empty(1,1,3,256,256)
            with self.assertRaises(ValueError):validate_batch_contract(b,w)

if __name__=='__main__':unittest.main()
