import unittest
from types import SimpleNamespace
import torch

from tokengs.models.object_locus_v3_set_controller import ObjectLocusV3SetController
from tokengs.models.object_locus_v3_set_loss import build_context_instance_targets, final_hungarian, v3_set_losses
from scripts.train_object_locus_v3_set import build_manifest, build_plan, lr_mult


def tiny_batch():
    sem=torch.full((1,4,2,3),-1,dtype=torch.long);ins=torch.zeros_like(sem)
    sem[0,0,0,:2]=2;ins[0,0,0,:2]=7
    sem[0,1,0,0]=2;ins[0,1,0,0]=7
    sem[0,0,1,0]=0;sem[0,1,1,0]=1
    sem[0,0,1,1]=4;ins[0,0,1,1]=8
    sem[0,0,1,2]=2 # invalid thing without instance id
    return {'semantic_label_all':sem,'instance_label_all':ins}


class V3SetContracts(unittest.TestCase):
    def test_class_head_has_19_way_distribution(self):
        c=ObjectLocusV3SetController(32)
        q=torch.randn(1,102,256);u=torch.randn(1,100,256)
        y=c.classify(q,u)
        self.assertEqual(tuple(y['thing_logits19'].shape),(1,100,19))
        self.assertTrue(torch.allclose(y['p_class'].sum(-1),torch.ones(1,100),atol=1e-6))
        self.assertTrue(torch.allclose(torch.softmax(y['thing_logits19'],-1),y['p_class'],atol=1e-6))
        self.assertFalse(any(x in n for n,_ in c.named_parameters() for x in ('W_own_e','ln_own_a','ln_de','W_de','W_off')))

    def test_scene_global_gt_and_ignore_rules(self):
        t=build_context_instance_targets(tiny_batch())
        self.assertEqual(t['gt_instance_ids'][0].tolist(),[7,8])
        self.assertEqual(t['gt_classes'][0].tolist(),[2,4])
        self.assertEqual(tuple(t['gt_pixel_masks'][0].shape),(2,2,2,3))
        self.assertTrue(t['gt_pixel_masks'][0][0,1,0,0])
        self.assertFalse(t['valid_pixels'][0,0,1,2])

    def test_hungarian_is_one_to_one_and_mask_class_only(self):
        b=tiny_batch();targets=build_context_instance_targets(b)
        logits=torch.zeros(1,100,19);logits[0,0,0]=8;logits[0,1,2]=8
        masks=torch.full((1,4,102,2,3),.05);masks[0,:2,0,0,:2]=.95;masks[0,:2,1,1,1]=.95
        pred={'states':[{'thing_logits19':logits}], 'region_mass':masks}
        _,pairs=final_hungarian(pred,b,targets)
        qi,ki=pairs[0]
        self.assertEqual(len(qi),2);self.assertEqual(len(set(qi.tolist())),2);self.assertEqual(len(set(ki.tolist())),2)

    def test_unmatched_slots_get_no_object_class_gradient(self):
        b=tiny_batch();logits=torch.zeros(1,100,19,requires_grad=True)
        with torch.no_grad():logits[0,0,0]=5;logits[0,1,2]=5
        regions=torch.full((1,4,102,2,3),.25,requires_grad=True)
        pred={'states':[{'thing_logits19':logits}], 'region_mass':regions}
        loss,metrics=v3_set_losses(pred,b);loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertNotEqual(float(logits.grad[0,99,18]),0.)
        self.assertTrue(torch.isfinite(regions.grad).all())
        for removed in ('loss_anchor','loss_identity','loss_semantic','loss_auxiliary'):
            self.assertNotIn(removed,metrics)

    def test_empty_gt_is_finite(self):
        b=tiny_batch();b['semantic_label_all'][:,:2].fill_(-1)
        logits=torch.randn(1,100,19,requires_grad=True);regions=torch.rand(1,4,102,2,3,requires_grad=True)
        loss,_=v3_set_losses({'states':[{'thing_logits19':logits}],'region_mass':regions},b)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(logits.grad).all())

    def test_lr_warmup_cosine_endpoints(self):
        self.assertAlmostEqual(lr_mult(1),.005);self.assertAlmostEqual(lr_mult(200),1.)
        self.assertAlmostEqual(lr_mult(3584),.1);self.assertTrue(.1<lr_mult(201)<1.)

    def test_mask_pool_gradients_and_detached_gaussian_opacity(self):
        c=ObjectLocusV3SetController(32)
        feat=torch.randn(1,8,256,requires_grad=True)
        membership_logits=torch.randn(1,8,102,requires_grad=True);membership=torch.sigmoid(membership_logits)
        pooled,mass=c.pool_anchor_features(feat,membership)
        pooled.sum().backward()
        self.assertIsNotNone(feat.grad);self.assertTrue(torch.isfinite(feat.grad).all())
        self.assertIsNotNone(membership_logits.grad);self.assertTrue(torch.isfinite(membership_logits.grad).all())
        gf=torch.randn(1,8,256,requires_grad=True)
        gm_logits=torch.randn(1,8,102,requires_grad=True);gm=torch.sigmoid(gm_logits)
        gs=torch.ones(1,8,14,requires_grad=True);fallback=torch.zeros(1,100,256)
        u,_=c.pool_gaussian_features(gf,gm,gs,fallback);u.sum().backward()
        self.assertIsNotNone(gf.grad);self.assertIsNotNone(gm_logits.grad)
        self.assertIsNone(gs.grad)

    def test_optimizer_labels_and_class_head_only(self):
        c=ObjectLocusV3SetController(32)
        names=dict(c.named_parameters())
        self.assertIn('class_head.weight',names)
        self.assertNotIn('category_head.weight',names);self.assertNotIn('objectness_head.weight',names)
        for forbidden in ('W_own_e','ln_own_a','ln_de','W_de','W_off'):
            self.assertFalse(any(forbidden in n for n in names))

    def test_locked_training_plan_exposure_and_disjointness(self):
        m=build_manifest();p=build_plan(m)
        self.assertEqual(len(m['train_all56']),56);self.assertEqual(len(m['same_scene_holdout8']),8)
        self.assertEqual(len(p['entries']),3584);self.assertTrue(m['frame_disjoint_train_holdout'])
        self.assertEqual({x['scene'] for x in m['train_all56']},set(m['small_stage_scenes']))
        self.assertEqual([sum(x['scene']==s for x in m['train_all56']) for s in m['small_stage_scenes']],[7]*8)
        for epoch in range(64):
            ids=[r['window_index'] for r in p['entries'][epoch*56:(epoch+1)*56]]
            self.assertEqual(sorted(ids),list(range(56)))


if __name__=='__main__':unittest.main()
