import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import torch
from test_fixed_query_oracles import example
from scripts.frozen_iou_head_core import FrozenTokenIoUHead,SOURCE_SHA,TARGET,scene_split,context_iou_targets,rerank_rows,make_reranked_pair,digest
from scripts import fit_frozen_iou_head as fitting


class FrozenIoUTests(unittest.TestCase):
    def test_head_is_66561_params_and_detaches_backbone_tokens(self):
        torch.manual_seed(42);head=FrozenTokenIoUHead();tokens=torch.randn(8,256,requires_grad=True)
        self.assertEqual(sum(p.numel() for p in head.parameters()),66561)
        prediction=head(tokens);self.assertTrue(torch.equal(prediction,torch.full((8,),.5)))
        loss=(prediction-.8).square().mean();loss.backward()
        self.assertIsNone(tokens.grad)
        self.assertGreater(float(head.net[-1].weight.grad.abs().sum()),0)
        before=head.net[-1].weight.detach().clone()
        optimizer=torch.optim.AdamW(head.parameters(),lr=.01);optimizer.step()
        self.assertFalse(torch.equal(before,head.net[-1].weight))

    def test_scene_split_is_disjoint_and_deterministic(self):
        training=['t'+str(i) for i in range(1191)];evaluation=['v'+str(i) for i in range(312)]
        split=scene_split(training,evaluation)
        self.assertEqual(len(split['train']),1072);self.assertEqual(len(split['dev']),119)
        self.assertEqual(split,scene_split(training[::-1],evaluation[::-1]))
        self.assertFalse(set(split['train'])&set(split['dev']))
        with self.assertRaises(ValueError):scene_split(training,[training[0]])

    def test_iou_labels_match_final_context_masks_and_keep_false_positives(self):
        out,sem,ins=example();targets,keep=context_iou_targets(out,sem,ins)
        torch.testing.assert_close(targets,torch.tensor([2/3,1/3,0.,0.]))
        self.assertEqual(keep.tolist(),[True,True,False,False])
        # Background/no-GT kept predictions are supervised as zero, not dropped.
        zero,same_keep=context_iou_targets(out,sem*0,ins*0)
        self.assertTrue(torch.equal(zero,torch.zeros(4)));self.assertTrue(torch.equal(keep,same_keep))
        with self.assertRaises(ValueError):context_iou_targets(out,sem.repeat(3,1,1),ins.repeat(3,1,1))

    def test_direct_score_replacement_preserves_ids_classes_and_original(self):
        rows=[dict(id=2,label_id=7,score=.95),dict(id=1,label_id=3,score=.2)]
        result=rerank_rows(rows,[.8,.1])
        self.assertEqual([r['score'] for r in result],[.1,.8])
        self.assertEqual([(r['id'],r['label_id']) for r in result],[(2,7),(1,3)])
        self.assertEqual(rows[0]['score'],.95)
        with self.assertRaises(ValueError):rerank_rows(rows,[float('nan'),.1])
        with self.assertRaises(ValueError):rerank_rows(rows,[.8])

    def test_masks_are_same_inodes_GT_same_links_scores_separate(self):
        with TemporaryDirectory() as d:
            source=Path(d)/'base';dest=Path(d)/'learned'
            for name in ('rgb','rgb_gt','depth','depth_gt','context_seg_gt','target_seg_gt'):
                (source/name).mkdir(parents=True);(source/name/'fixed.png').write_bytes(b'fixed immutable data')
            for scope in ('context','target'):
                p=source/(scope+'_seg_pred');p.mkdir();(p/'fixed.png').write_bytes(b'mask bytes')
                (p/'pred.json').write_text(json.dumps([dict(id=1,label_id=3,score=.2)]))
            make_reranked_pair(source,dest,torch.tensor([.8]))
            for scope in ('context','target'):
                self.assertEqual((source/(scope+'_seg_pred/fixed.png')).stat().st_ino,(dest/(scope+'_seg_pred/fixed.png')).stat().st_ino)
                self.assertAlmostEqual(json.loads((dest/(scope+'_seg_pred/pred.json')).read_text())[0]['score'],.8)
                self.assertEqual(json.loads((source/(scope+'_seg_pred/pred.json')).read_text())[0]['score'],.2)
            self.assertEqual((dest/'target_seg_gt/fixed.png').read_bytes(),b'fixed immutable data')

    def test_cache_official_scene_injection_is_rejected(self):
        with TemporaryDirectory() as d:
            cache=Path(d);p=cache/'chunk.pt'
            torch.save(dict(source_sha256=SOURCE_SHA,target_definition=TARGET,features=torch.zeros(1,256),targets=torch.zeros(1),
                            records=[dict(window_index=0,scene='official',start=0,end=1)]),p)
            (cache/'export_complete.json').write_text(json.dumps(dict(status='FROZEN_IOU_TRAIN_CACHE_COMPLETE',windows=8337,source_sha256=SOURCE_SHA,chunks=[dict(file=p.name,sha256=digest(p))])))
            with self.assertRaisesRegex(ValueError,'Official GT'):
                fitting.load_rows(cache,dict(train=['train'],dev=['dev'],official_eval=['official']))

    def test_actual_head_fit_has_no_backbone_and_updates_only_head(self):
        with TemporaryDirectory() as d:
            root=Path(d);(root/'cache').mkdir();(root/'cache/export_complete.json').write_text('{}')
            config=dict(seed=42,head_learning_rate=.01,head_weight_decay=0.,head_max_epochs=4,head_patience=3,head_batch_size=32)
            (root/'scene_split.json').write_text(json.dumps(dict(train=['t'],dev=['d'],official_eval=['v'])))
            config.update(source_sha256=SOURCE_SHA,scene_split_sha256=digest(root/'scene_split.json'))
            (root/'configuration.json').write_text(json.dumps(config))
            torch.manual_seed(123);x=torch.randn(128,256);y=torch.full((128,),.8)
            with patch.object(fitting,'load_rows',return_value={'train':(x,y),'dev':(x,y)}),patch.object(sys,'argv',['fit','--run-root',str(root)]),patch('builtins.print'):
                fitting.main()
            receipt=json.loads((root/'head/FIT_COMPLETE.json').read_text())
            self.assertEqual(receipt['backbone_updates'],0);self.assertEqual(receipt['trainable_parameters'],66561)
            self.assertGreater(receipt['head_steps'],0);self.assertTrue(receipt['changed_head_tensors'])
            self.assertLess(receipt['selected_dev']['mse'],receipt['initial_dev']['mse'])
            saved=torch.load(root/'head/head_best.pt',map_location='cpu',weights_only=False)
            self.assertTrue(all(k.startswith('net.') for k in saved['head']))
            self.assertFalse(receipt['official_eval_used_for_selection'])


if __name__=='__main__':unittest.main()
