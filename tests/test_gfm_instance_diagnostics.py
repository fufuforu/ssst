import unittest
import torch
from torch import nn
from scripts.gfm_instance_diagnostic_math import iou_matrix,stage_masks,InitializerTrace
from tokengs.models.gfm_dual_stream_components import SemanticSAMQueryInitializer


class DiagnosticTests(unittest.TestCase):
    def test_iou_ignores_invalid_pixels(self):
        gt=torch.tensor([[[1,0],[0,0]]],dtype=torch.bool);pred=torch.tensor([[[1,1],[0,0]]],dtype=torch.bool)
        valid=torch.tensor([[1,0],[1,1]],dtype=torch.bool)
        torch.testing.assert_close(iou_matrix(pred,gt,valid),torch.ones(1,1))

    def test_stage_eligibility_competition_and_native_area_removal(self):
        member=torch.zeros(1,1,102,2,4);member[:,:,0]=.6;member[:,:,1,:,:,]=.9
        # q1 beats q0 except 1/8 pixels: native area rule deletes q0's remaining pixel.
        member[0,0,1,0,0]=0
        pc=torch.zeros(1,100,19);pc[:,:,18]=1;pc[:,0,18]=0;pc[:,0,0]=1;pc[:,1,18]=0;pc[:,1,1]=1
        output=dict(region_mass=member,alpha=torch.ones(1,1,1,2,4),p_class=pc,semantic_scores=torch.zeros(1,1,20,2,4))
        stages,_,_=stage_masks(output)
        self.assertEqual(int(stages['raw'][0,0].sum()),8)
        self.assertEqual(int(stages['competition'][0,0].sum()),1)
        self.assertEqual(int(stages['final'][0,0].sum()),0)
        self.assertFalse(stages['eligible'][0,2].any())

    def test_initializer_observer_reproduces_actual_grouping_queries_and_cap(self):
        torch.manual_seed(4);init=SemanticSAMQueryInitializer();trace=InitializerTrace(init)
        feature=torch.ones(1,2,256,2,2);points=torch.zeros(1,2,4,4,3);points[...,0]=torch.arange(4)[None,None,None,:]
        masks=[torch.ones(4,4,dtype=torch.bool),torch.eye(4,dtype=torch.bool)]
        proposals=[[[dict(score=1.,mask=masks[0]),dict(score=.9,mask=masks[0])],[dict(score=1.,mask=masks[0])]]]
        mu=torch.randn(1,8,3)
        native=init(feature,proposals,points,mu)
        q,c,s,logs=native
        selected=trace.selected_masks(dict(q_pre=q,initial_query_centers=c,initial_query_scales=s,proposal_initialization=logs))
        self.assertEqual(len(selected),1);self.assertTrue(selected[0].all())
        trace.close();after=init(feature,proposals,points,mu)
        torch.testing.assert_close(after[0],q,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
