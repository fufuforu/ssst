import unittest
from dataclasses import replace
import numpy as np
from scripts.fixed_instance_diagnostic_math import cluster_view,RECIPE,evaluate_masks,label_masks,supported_best_iou


class FixedDiagnosticTests(unittest.TestCase):
    def test_perfect_duplicate_missing_and_merged_predictions(self):
        gt=np.zeros((2,10,10),bool);gt[0,:5,:]=1;gt[1,5:,:]=1;valid=np.ones((10,10),bool)
        perfect=evaluate_masks(gt,gt,valid)
        self.assertEqual((perfect['one_to_one_gt_iou'],perfect['recall50'],perfect['precision50']),(1.,1.,1.))
        duplicate=evaluate_masks(np.concatenate([gt,gt[:1]]),gt,valid)
        self.assertEqual(duplicate['tp50'],2);self.assertAlmostEqual(duplicate['precision50'],2/3)
        missing=evaluate_masks(gt[:1],gt,valid)
        self.assertEqual(missing['one_to_one_gt_iou'],.5);self.assertEqual(missing['fn50'],1)
        merged=evaluate_masks(gt.any(0)[None],gt,valid)
        self.assertEqual(merged['tp50'],0);self.assertEqual(merged['one_to_one_gt_iou'],.25)

    def test_void_and_empty_do_not_create_false_objects(self):
        valid=np.ones((4,4),bool);valid[0]=0
        gt=np.zeros((1,4,4),bool);gt[:,1:]=1
        p=np.concatenate([gt,np.zeros_like(gt)]);p[1,0]=1
        r=evaluate_masks(p,gt,valid);self.assertEqual(r['pred_count'],1);self.assertEqual(r['recall50'],1.)
        r=evaluate_masks(np.empty((0,4,4),bool),gt,valid);self.assertEqual(r['best_gt_iou'],0.)
        r=evaluate_masks(gt,np.empty((0,4,4),bool),valid);self.assertIsNone(r['recall50']);self.assertEqual(r['fp50'],1)

    def test_support_subset_does_not_hide_missing_gt_in_primary_metrics(self):
        gt=np.eye(4,dtype=bool).reshape(4,2,2);p=gt[:1];valid=np.ones((2,2),bool)
        self.assertEqual(evaluate_masks(p,gt,valid)['best_gt_iou'],.25)
        self.assertEqual(supported_best_iou(p,gt,valid,[True,False,False,False])['best_gt_iou'],1.)

    def test_actual_hdbscan_two_groups_scale_invariance_and_noise_exclusion(self):
        rng=np.random.default_rng(42);f=np.zeros((8,20,20),np.float32)
        f[0,:,:10]=1;f[1,:,10:]=1;f+=rng.normal(0,.001,f.shape).astype(np.float32)
        f[:,0,0]=0;alpha=np.ones((20,20),np.float32);alpha[0,1]=0
        recipe=replace(RECIPE,min_cluster_size=20,min_samples=5,n_jobs=1)
        labels,report,visible=cluster_view(f,alpha,recipe)
        labels2,_,_=cluster_view(f*7,alpha,recipe)
        self.assertEqual(report['clusters'],2);self.assertEqual(labels[0,0],-1);self.assertEqual(labels[0,1],-1)
        self.assertFalse(visible[0,0]);self.assertFalse(visible[0,1])
        gt=np.zeros((2,20,20),bool);gt[0,:,:10]=1;gt[1,:,10:]=1
        masks,_=label_masks(labels);self.assertGreater(evaluate_masks(masks,gt,visible)['one_to_one_gt_iou'],.99)
        # IDs may permute; compare the resulting partition.
        self.assertTrue(np.array_equal(labels[:,None]==labels[None,:],labels2[:,None]==labels2[None,:]))
        self.assertFalse(report['gt_used']);self.assertFalse(report['xy_used'])

    def test_small_or_nonfinite_inputs(self):
        labels,report,_=cluster_view(np.zeros((8,2,2)),np.ones((2,2)))
        self.assertEqual(report['clusters'],0);self.assertTrue((labels==-1).all())
        with self.assertRaises(ValueError):cluster_view(np.full((8,2,2),np.nan),np.ones((2,2)))

    def test_cpu_export_evaluation_keeps_methods_and_both_views_separate(self):
        import tempfile
        from pathlib import Path
        import torch
        from scripts.evaluate_fixed_instance_clustering import evaluate_window,summarize_metrics
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'baseline67050';folder.mkdir()
            rng=np.random.default_rng(2);features=np.zeros((2,8,20,20),np.float32)
            features[:,0,:,:10]=1;features[:,1,:,10:]=1
            features+=rng.normal(0,.001,features.shape).astype(np.float32)
            masks=np.zeros((2,2,20,20),bool);masks[0,:,:,:10]=1;masks[1,:,:,10:]=1
            torch.save(dict(scene='synthetic',context=[1,2],features=torch.tensor(features),alpha=torch.ones(2,20,20),
                raw_membership_masks=torch.tensor(masks),gt_masks=torch.tensor(masks),nms_eligible=torch.ones(2,dtype=torch.bool),
                assembled_instance=torch.tensor(masks[0].astype(int)+2*masks[1].astype(int)),predicted_semantic=torch.full((2,20,20),2),
                valid=torch.ones(2,20,20,dtype=torch.bool),supported_gt=torch.tensor([True,False])),folder/'00_inference.pt')
            rows=evaluate_window((tmp,'baseline67050',0));self.assertEqual(len(rows),2)
            self.assertEqual([r['model'] for r in rows],['baseline67050']*2)
            summary=summarize_metrics(rows);cluster=[r for r in summary if r['method']=='hdbscan8d'][0]
            self.assertEqual(cluster['gt_count'],4);self.assertEqual(cluster['supported_gt_count'],2)
            self.assertGreater(cluster['one_to_one_gt_iou'],.99)


if __name__=='__main__':unittest.main()
