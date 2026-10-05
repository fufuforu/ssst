import unittest
import torch
import json
from pathlib import Path

from object_locus_text_refer.head import ObjectLocusTextReferHead, hard_gaussian_membership, soft_gaussian_membership
from object_locus_text_refer.loss import resolve_slot_target
from object_locus_text_refer.evaluation import evaluate_records
from object_locus_text_refer.data import SIU3RReferDataset


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

    def test_evaluation_retains_failures(self):
        r=evaluate_records([{"text_key":"a","pred_mask":None,"gt_mask":None,"valid_mask":None,"selected_slot":100}])
        self.assertEqual(r['records'],1);self.assertEqual(r['per_record'][0]['iou'],0.)
        self.assertEqual(r['null_selection_rate'],1.)

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

    def test_train_text_choice_does_not_change_global_rng(self):
        data=Path('/space/mawb/SIU3R/data/scannet')
        refs=json.loads((data/'train_refer_seg_data.json').read_text())
        scene=next(iter(refs)); oid=int(next(iter(refs[scene]['objects'])))
        subset={scene:{'frame2object':refs[scene]['frame2object'],'objects':{str(oid):refs[scene]['objects'][str(oid)]}}}
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'train.json';path.write_text(json.dumps(subset))
            ds=SIU3RReferDataset(path,split='train',seed=19,load_arrays=lambda s,f:{'packed_panoptic':torch.zeros(1,1,1,dtype=torch.long).numpy()})
            torch.manual_seed(91);before=torch.random.get_rng_state().clone();a=ds[0];after=torch.random.get_rng_state()
            self.assertTrue(torch.equal(before,after));self.assertIn(a['text'],subset[scene]['objects'][str(oid)]['text'])

    def test_official_train_validation_scene_split_is_disjoint(self):
        root=Path('/space/mawb/SIU3R/data/scannet')
        train=json.loads((root/'train_refer_seg_data.json').read_text())
        val=json.loads((root/'val_refer_seg_data.json').read_text())
        self.assertFalse(set(train)&set(val))


if __name__=='__main__': unittest.main()
