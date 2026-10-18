import unittest
import numpy as np
import torch
from torch import nn
from tokengs.models.input_types import ModelInputDecoder
from scripts.instance_boundary_diagnostic_math import (
    boundary_regions,boundary_response,binary_metrics,calibrate_on_train,dominant_contributor,hard_feature_map,gaussian_pair_probe,gaussian_gt_ownership)


class BoundaryMathTests(unittest.TestCase):
    def test_gt_ownership_distinguishes_stuff_and_rejects_mixed_gaussians(self):
        class Renderer(nn.Module):
            def render_feature_channels(self,g,f,camera,intrinsics):
                matrix=torch.tensor([[.9,0.,0.],[.1,.9,0.],[0.,0.,.5],[0.,0.,.5]])
                return dict(images_pred=torch.einsum('pn,bnc->bcp',matrix,f).reshape(1,1,f.shape[-1],2,2))
        g=torch.zeros(1,3,14);decoder=ModelInputDecoder(cam_view=torch.eye(4).reshape(1,1,4,4),intrinsics=torch.ones(1,1,4))
        masks=torch.zeros(1,1,2,2,dtype=torch.bool);masks[0,0,0,0]=True
        targets=dict(gt_pixel_masks=[masks],valid_pixels=torch.ones(1,1,2,2,dtype=torch.bool))
        all_labels,thing_labels,purity,dim=gaussian_gt_ownership(g,decoder,Renderer(),targets,torch.tensor([[[2,0],[0,1]]]))
        self.assertEqual(all_labels.tolist(),[0,1,-1]);self.assertEqual(thing_labels.tolist(),[0,-1,-1]);self.assertEqual(dim,3)
        torch.testing.assert_close(purity,torch.tensor([.9,1.,.5]))

    def test_gt_void_and_stuff_ids_do_not_create_fake_instance_boundary(self):
        sem=torch.zeros(1,8,8,dtype=torch.long);ids=torch.arange(64).reshape(1,8,8)
        self.assertFalse(boundary_regions(sem,ids)['edge'].any())
        sem[:,:,4:]=2;ids[:,:,4:]=1
        regions=boundary_regions(sem,ids);self.assertTrue(regions['edge'][:,:,3:5].all())
        sem[:,:,4:]=20;self.assertFalse(boundary_regions(sem,ids)['edge'].any())

    def test_rendered_step_detected_but_flat_field_and_shuffle_are_not(self):
        x=torch.zeros(1,2,16,16);x[:,0,:,:8]=1;x[:,1,:,8:]=1
        raw,response=boundary_response(x)
        self.assertEqual(float(raw[:,:,7:9].mean()),1.)
        self.assertEqual(float(response[:,:,:3].sum()),0.)
        flat=boundary_response(torch.ones_like(x))[1]
        self.assertLess(float(flat.max()),1e-6)
        r=binary_metrics([1.]*100,[0.]*100);self.assertEqual(r['auroc'],1.)
        self.assertLess(abs(r['shuffle_auroc']-.5),.15)

    def test_small_but_discriminative_scores_can_be_calibrated_on_train(self):
        rows=[dict(positive_samples=[.001]*100,negative_samples=[.0001]*100)]
        fit=calibrate_on_train(rows);self.assertEqual(fit['train_youden'],1.)
        original=binary_metrics(rows[0]['positive_samples'],rows[0]['negative_samples'])
        calibrated=binary_metrics(rows[0]['positive_samples'],rows[0]['negative_samples'],fit['threshold'],fit['temperature'])
        self.assertEqual(original['tpr'],0.);self.assertEqual(calibrated['tpr'],1.);self.assertEqual(calibrated['fpr'],0.)
        self.assertEqual(original['auroc'],calibrated['auroc'])

    def test_exact_winner_keeps_all_geometry_and_checks_chunk_ties(self):
        class Renderer(nn.Module):
            def render_feature_channels(self,g,f,camera,intrinsics):
                assert g.shape[1]==3 # No subset rendering: transmittance unchanged.
                a=torch.tensor([[.1,.6,.2],[.4,.4,.1],[0.,0.,0.],[.2,.1,.3]])
                return dict(images_pred=torch.einsum('pn,bnc->bcp',a,f).reshape(1,1,f.shape[-1],2,2))
        g=torch.zeros(1,3,14);decoder=ModelInputDecoder(cam_view=torch.eye(4).reshape(1,1,4,4),intrinsics=torch.ones(1,1,4))
        a,w,total=dominant_contributor(Renderer(),g,decoder,chunk=1)
        self.assertEqual(a.tolist(),[[[[1,0],[0,2]]]])
        torch.testing.assert_close(total,torch.tensor([[[[.9,.9],[0.,.6]]]]))
        b,_,_=dominant_contributor(Renderer(),g,decoder,chunk=2);self.assertTrue(torch.equal(a,b))
        feat=torch.tensor([[[1.,0.],[0.,1.],[-1.,0.]]])
        hard=hard_feature_map(feat,a);self.assertEqual(hard.shape,(1,2,2,2));self.assertEqual(float(hard[0,1,0,0]),1.)

    def test_gaussian_same_gt_cross_parent_and_shuffled_controls(self):
        torch.manual_seed(8)
        n=40;label=torch.arange(n)//20;parent=torch.arange(n)//5
        f=torch.nn.functional.one_hot(label,2).float()+torch.randn(n,2)*.001
        result=gaussian_pair_probe(f,torch.randn(n,3),label,parent)
        self.assertGreater(result['global_pairs']['auroc'],.99)
        self.assertGreater(result['same_gt_cross_parent']['count'],0)
        self.assertLess(abs(result['correspondence_shuffled_auroc']-.5),.1)


if __name__=='__main__':unittest.main()
