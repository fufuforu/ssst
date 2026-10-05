import unittest
import torch
import json
import random
import inspect
from pathlib import Path

from object_locus_text_refer.head import ObjectLocusTextReferHead, build_head_optimizer, hard_gaussian_membership, soft_gaussian_membership
from object_locus_text_refer.loss import resolve_slot_target
from object_locus_text_refer.evaluation import aggregate_expressions, binarize_probability, masked_iou
from object_locus_text_refer.data import SIU3RReferDataset, sample_context_referent
from object_locus_text_refer.adapter import FULL1201_SHA256, assert_visual_beta, validate_visual_checkpoint_metadata


class TextReferContracts(unittest.TestCase):
    def test_head_shapes_softmax_null_and_fixed_membership(self):
        torch.manual_seed(4)
        rng_before=torch.random.get_rng_state().clone()
        head=ObjectLocusTextReferHead()
        self.assertTrue(torch.equal(rng_before,torch.random.get_rng_state()))
        ids=torch.zeros(2,77,dtype=torch.long);ids[:,0]=49406;ids[:,1]=49407
        mask=torch.zeros_like(ids);mask[:,:2]=1
        text=torch.randn(2,77,512);q=torch.randn(2,100,256)
        out=head(text,ids,mask,q)
        self.assertEqual(tuple(out['scores'].shape),(2,101));self.assertEqual(tuple(out['pi'].shape),(2,101))
        self.assertTrue(torch.allclose(out['pi'].sum(-1),torch.ones(2)))
        P=torch.rand(2,65536,100)
        soft=soft_gaussian_membership(out['pi'],P)
        self.assertEqual(tuple(soft.shape),(2,65536))
        zero_scores=torch.full((1,101),-10.);zero_scores[0,100]=10
        hard,slot=hard_gaussian_membership(zero_scores,P[:1])
        self.assertEqual(slot.item(),100);self.assertEqual(float(hard.abs().sum()),0.)

    def test_same_selected_slot_is_shared_across_views(self):
        scores=torch.zeros(1,101);scores[0,7]=4
        P=torch.rand(1,65536,100)
        membership,slot=hard_gaussian_membership(scores,P)
        # Rendering is view-dependent, 3D choice remains one scalar slot.
        rendered=[membership.sum(),membership.mean()]
        self.assertEqual(slot.item(),7);self.assertEqual(len(rendered),2)

    def test_unmatched_visible_is_not_null_and_missing_is_not_null(self):
        self.assertIsNone(resolve_slot_target(23,[21,23],{},True))
        self.assertEqual(resolve_slot_target(23,[21,23],{},False),100)
        self.assertIsNone(resolve_slot_target(25,[21,23],{},True))

    def test_probability_threshold_is_strictly_half(self):
        self.assertEqual(binarize_probability(torch.tensor([0.5,0.50001,0.49])).tolist(),[False,True,False])

    def test_expression_aggregation_and_failures(self):
        r=aggregate_expressions([
            {'scene':'s1','object_id':1,'view_ious':[.75,.5],'selected_slot':2},
            {'scene':'s2','object_id':2,'view_ious':[0.,0.],'selected_slot':None,'failure_reason':'renderer failed'}])
        self.assertAlmostEqual(r['context_refer_records'][0]['expression_iou'],.625)
        self.assertEqual(r['context_refer_mIoU'],.3125)
        self.assertEqual(r['context_refer_Acc@0.25'],.5)
        self.assertEqual(r['context_refer_Acc@0.50'],.5)
        self.assertEqual(r['context_refer_null_rate'],0.)
        self.assertEqual(r['context_refer_failed_expressions'],1)
        iou,reason=masked_iou(torch.zeros(2),torch.zeros(2),torch.zeros(2,dtype=torch.bool))
        self.assertEqual(iou,0.);self.assertEqual(reason,'empty_valid_domain')

    def test_exposure_contract_and_beta(self):
        meta={'epoch':6,'completed_updates':6258,'completed_exposures':50064}
        self.assertEqual(validate_visual_checkpoint_metadata(meta,FULL1201_SHA256),50064)
        with self.assertRaises(RuntimeError): validate_visual_checkpoint_metadata({**meta,'completed_exposures':0},FULL1201_SHA256)
        states=[{'layer':layer,'beta':.1} for layer in range(1,13)]
        self.assertEqual(assert_visual_beta(states),[.1]*4)
        with self.assertRaises(RuntimeError): assert_visual_beta([{'layer':layer,'beta':0.0} for layer in (6,8,10,12)])

    def test_optimizer_contains_only_text_head(self):
        head=ObjectLocusTextReferHead(); optimizer=build_head_optimizer(head)
        ids={id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertEqual(ids,{id(p) for p in head.parameters()})
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.ndim==1: self.assertEqual(group['weight_decay'],0.)
                else: self.assertEqual(group['weight_decay'],.05)

    def test_training_has_no_cross_scene_batch_cache_or_zero_visual_step(self):
        root=Path(__file__).resolve().parents[1]
        train=(root/'scripts/train_object_locus_text_refer.py').read_text()
        self.assertNotIn('batch_cache',train)
        for name in ('train_object_locus_text_refer.py','smoke_object_locus_text_refer.py','eval_object_locus_text_refer.py'):
            source=(root/'scripts'/name).read_text()
            self.assertNotRegex(source,r'forward_object_locus\([^\n]*step\s*=\s*0')
        from object_locus_text_refer.adapter import forward_frozen_visual
        self.assertIn('step=exposure',inspect.getsource(forward_frozen_visual))

    def test_real_validation_pair_expansion_and_scene_local_id(self):
        data=Path('/space/mawb/SIU3R/data/scannet')
        pairs=json.loads((data/'val_refer_pair.json').read_text())
        refs=json.loads((data/'val_refer_seg_data.json').read_text())
        first=pairs[0]; scene=first['scene_name']; oid=int(first['context_objects'])
        ds=SIU3RReferDataset(data/'val_refer_seg_data.json',data/'val_refer_pair.json','val',
            load_arrays=lambda s, f: {'packed_panoptic':torch.tensor([[[7,3003]]]).numpy(), 'valid_mask':torch.ones(1,1,2,dtype=torch.bool).numpy()})
        row=ds[0]
        self.assertEqual((row['scene'],row['object_id'],row['context_frame_ids']),(scene,oid,tuple(first['context_views_id'])))
        self.assertEqual(row['text'],first['texts'])
        self.assertEqual(int(row['context_target_mask'].sum()),1)
        # panoptic_label_id is class metadata, never substituted for instance ID.
        self.assertEqual(refs[scene]['objects'][str(oid)]['panoptic_label_id'],7)
        self.assertEqual(int(row['context_target_mask'][0,0,1]),1)  # packed 3003 -> instance 3

    def test_context_precedes_object_and_text_sampling_with_normalized_ids(self):
        refs={'sceneA':{'frame2object':{'4':['7','8'],'6':[9,'10']},'objects':{
            str(i):{'text':[f'raw description {i}']} for i in (7,8,9,10)}}}
        sem=torch.full((1,4,1,4),-1,dtype=torch.long);ins=torch.zeros_like(sem)
        sem[0,0,0]=torch.tensor([2,3,0,1]);ins[0,0,0]=torch.tensor([7,8,0,0])
        sem[0,1,0]=torch.tensor([2,3,2,3]);ins[0,1,0]=torch.tensor([7,8,9,10])
        batch={'frame_ids':torch.tensor([[4,6,12,14]]),'semantic_label_all':sem,'instance_label_all':ins}
        rng=random.Random(42);before=torch.random.get_rng_state().clone()
        sample=sample_context_referent(refs,'sceneA',batch,rng)
        self.assertEqual(sample['context_frame_ids'],[4,6])
        self.assertEqual(sample['candidate_object_ids'],[7,8,9,10])
        self.assertIn(sample['object_id'],sample['candidate_object_ids'])
        self.assertEqual(sample['text'],f"raw description {sample['object_id']}")
        self.assertTrue(torch.equal(sample['context_target_mask'],sample['context_valid_mask']&(ins[0,:2]==sample['object_id'])))
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))

    def test_official_train_validation_scene_split_is_disjoint(self):
        root=Path('/space/mawb/SIU3R/data/scannet')
        train=json.loads((root/'train_refer_seg_data.json').read_text())
        val=json.loads((root/'val_refer_seg_data.json').read_text())
        self.assertFalse(set(train)&set(val))


if __name__=='__main__': unittest.main()
