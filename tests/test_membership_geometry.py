import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.membership_geometry_core import probability,mask_objective,geometry_mask_loss,teacher_record,coefficient
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches
from scripts.score_membership_geometry import novel_links,boundary_rgb,metrics

class MembershipGeometryTests(unittest.TestCase):
    def test_transparency_is_unknown(self):
        alpha=torch.tensor([[[[[.2]]]]]);features=torch.tensor([[[[[.1]],[[.1]],[[0.]]]]])
        p=probability(features,alpha)
        self.assertTrue(torch.allclose(p.sum(2),torch.ones_like(alpha.squeeze(2))))
        self.assertAlmostEqual(float(p[0,0,-1,0,0]),.8,places=6)
        empty=probability(features*0,alpha*0)
        self.assertGreater(float(mask_objective(empty,torch.zeros(1,1,1,dtype=torch.long))[0]),5)

    def test_ignored_pixels_have_zero_gradient(self):
        logits=torch.randn(1,2,4,2,3,requires_grad=True);p=logits.softmax(2);y=torch.zeros(2,2,3,dtype=torch.long);y[:,:,1]=-1
        loss,*_=mask_objective(p,y);loss.backward()
        self.assertEqual(float(logits.grad[:,:,:,:,1].abs().sum()),0)

    def test_grouping_and_unsupported_targets(self):
        from types import SimpleNamespace
        m=torch.tensor([[[.2,.8,.7,.1,.4,.2],[.9,.3,.5,.2,.1,.8]]],requires_grad=True)
        sem=torch.tensor([[[[0,1,2,3]],[[0,1,2,3]]]])
        gt=torch.stack([(sem[0]==2),(sem[0]==3)])
        pred=dict(gaussian_membership=m,num_thing_queries=4,predicted_context_c2w=torch.eye(4).repeat(1,2,1,1),predicted_context_intrinsics=torch.ones(1,2,4),
            anchor_point_indices=torch.tensor([[0]]),gaussians=torch.zeros(1,2,14),final_pairs=[(torch.tensor([0,1]),torch.tensor([0,0]))],
            final_targets=dict(gt_instance_ids=[torch.tensor([10,20])],gt_pixel_masks=[gt],valid_pixels=torch.ones_like(sem,dtype=torch.bool)))
        with patch('tokengs.models.unified_query_loss.unified_instance_loss',return_value=(None,None)):
            r=teacher_record(pred,dict(semantic_label_all=sem,images_input=torch.zeros(1,2,3,1,4)),SimpleNamespace())
        self.assertEqual(r['features'].shape,(1,2,4));self.assertFalse(r['features'].requires_grad)
        self.assertEqual(r['target'][0].tolist(),[[0,1,2,-1]])
        expected=torch.tensor([.4,.2,.8,.7]);expected/=expected.sum()
        self.assertTrue(torch.allclose(r['features'][0,0].float(),expected,atol=.001))

    def test_gradient_enters_only_xyz_scale(self):
        from types import SimpleNamespace
        g=torch.randn(1,3,14,requires_grad=True);features=torch.rand(1,3,3,requires_grad=True)
        def render(gg,f,cam,k):
            signal=gg[...,:3].sum()+gg[...,4:7].sum()+gg[...,3].sum()+gg[...,7:].sum()
            alpha=signal.sigmoid().reshape(1,1,1,1,1).expand(1,2,1,1,1)
            return dict(images_pred=f.mean(1).reshape(1,1,3,1,1).expand(1,2,3,1,1)*alpha,alphas_pred=alpha)
        record=dict(features=features,target=torch.zeros(2,1,1,dtype=torch.long),anchor_point_indices=torch.tensor([1]),cam_view=torch.zeros(1),intrinsics=torch.zeros(1))
        model=SimpleNamespace(gs=SimpleNamespace(render_feature_channels=render));pred=dict(gaussians=g,anchor_point_indices=torch.tensor([1]))
        loss,*_=geometry_mask_loss(model,pred,record);loss.backward()
        self.assertGreater(float(g.grad[...,:3].abs().sum()),0);self.assertGreater(float(g.grad[...,4:7].abs().sum()),0)
        self.assertEqual(float(g.grad[...,3].abs().sum()+g.grad[...,7:].abs().sum()),0)
        self.assertIsNone(features.grad)
        pred['anchor_point_indices']=torch.tensor([2])
        with self.assertRaises(ValueError):geometry_mask_loss(model,pred,record)

    def test_four_rank_accumulation_schedule(self):
        order=epoch_order(8337,4);self.assertEqual(len(order),8344)
        for step in (0,500,1042):
            actual=[rank_microbatches(order,step,r,4)[micro] for micro in range(2) for r in range(4)]
            self.assertEqual(actual,order[step*8:(step+1)*8])
        self.assertEqual(coefficient('G1',100),0);self.assertEqual(coefficient('G2',0),.0001);self.assertEqual(coefficient('G2',99),.01)

    def test_novel_subset_and_boundary_serialization(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source';dest=root/'novel';name='scene_test_context0_1';records=[dict(name=name,frame_ids=list(range(6)))]
            for sub in ('rgb','rgb_gt','depth','depth_gt','target_seg_gt'):(source/'all'/name/sub).mkdir(parents=True)
            for frame in range(6):
                for sub in ('rgb','rgb_gt','depth','depth_gt'):Image.fromarray(np.zeros((4,4,3),dtype=np.uint8)).save(source/'all'/name/sub/f'scene_test_{frame}.png')
                packed=np.full((4,4),1000,dtype='int64');packed[:,2:]=3001
                rgb=np.stack([packed%256,(packed//256)%256,packed//65536],-1).astype('uint8')
                Image.fromarray(rgb).save(source/'all'/name/'target_seg_gt'/f'scene_test_gt{frame}.png')
            novel_links(source,dest,records)
            self.assertEqual(len(list((dest/name/'rgb').glob('*.png'))),4)
            self.assertFalse((dest/name/'rgb'/'scene_test_0.png').exists())
            b=boundary_rgb(source,records);self.assertEqual(b['context']['images'],2);self.assertEqual(b['novel']['images'],4);self.assertEqual(b['novel']['mae'],0)

    def test_native_metric_schema(self):
        raw=dict(absrel=.1,rmse=.2,psnr=25.,ssim=.8,lpips=.2,context_miou=.6,target_miou=.5,context_pq=.4,target_pq=.3,
            context_map=dict(map=.2,mar_100=.4),target_map=dict(map=.1,mar_100=.3))
        self.assertEqual(metrics(raw)['target_mar_100'],.3)
        raw['psnr']=float('nan')
        with self.assertRaises(ValueError):metrics(raw)

if __name__=='__main__':unittest.main()
