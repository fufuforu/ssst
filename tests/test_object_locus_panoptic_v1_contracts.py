"""Only contracts required by the locked panoptic recipe."""
import json
import math
import unittest
import numpy as np
import torch
from scripts.object_locus_panoptic_v1_runtime import (
    build_model, build_optimizer, assets, sample_index, lr_multiplier, combine_gradients, REPORTS)
from tokengs.models.object_locus_panoptic_v1_controller import ObjectLocusPanopticV1Controller,geometry_bias


class PanopticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.model,cls.opt=build_model('cpu')

    def test_strict_initialization_and_parameter_coverage(self):
        model=self.model
        mapping=json.loads((REPORTS/'weights_mapping.json').read_text())
        self.assertEqual(mapping['counts'],dict(reconstruction=450,encoder=292,mast3r_excluded=725,adapter=187,mask_decoder=326))
        self.assertEqual(len(model.understanding.encoder.enc_blocks),24)
        self.assertEqual(model.understanding.adapter.interaction_indexes,[5,11,17,23])
        self.assertTrue(model.understanding.adapter.add_vit_feature)
        self.assertEqual([model.understanding.adapter.H*4,model.understanding.adapter.H*2,model.understanding.adapter.H,model.understanding.adapter.H//2],[128,64,32,16])
        self.assertFalse(any('intrinsic_encoder' in k or 'dec_norm' in k for k in model.understanding.encoder.state_dict()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model.train()
        self.assertTrue(all(not m.training for m in model.understanding.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)))
        owners=[n for n,m in model.named_modules(remove_duplicate=False) if m is model.understanding.mask_embedder]
        self.assertEqual(len(owners),1)
        optimizer=build_optimizer(model)
        for g in optimizer.param_groups:
            if g['name'].startswith('reconstruction'):self.assertEqual(g['weight_decay'],0)
        for group in optimizer.param_groups:
            for name in group['param_names']:
                if name.endswith('.level_embed') or 'child_index_embedding' in name or name=='panoptic.stuff_seed':
                    self.assertEqual(group['weight_decay'],0)
        names=[n for g in optimizer.param_groups for n in g['param_names']]
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(set(names),set(dict(model.named_parameters())))
        self.assertTrue(all(n in next(g['param_names'] for g in optimizer.param_groups if g['name']=='new_decay') for n in names if n.endswith('W_inject.weight')))

    def test_adapter_receives_raw_twenty_fourth_block(self):
        from unittest.mock import patch
        from tokengs.models.object_locus_panoptic_v1_pretrained import ImageOnlyMASt3R
        # Exercise the actual wrapper and encoder forward with lightweight
        # blocks; enc_norm deliberately changes values if incorrectly called.
        raw=[];received=[]
        class Patch(torch.nn.Module):
            def forward(self,image):
                return torch.zeros(image.shape[0],1024,1024),torch.zeros(image.shape[0],1024,2,dtype=torch.long)
        class Block(torch.nn.Module):
            def forward(self,x,pos):
                y=x+1;raw.append(y);return y
        class Adapter(torch.nn.Module):
            def forward(self,image,states):
                received.extend(states)
                raise StopIteration('stop after verifying adapter input')
        encoder=ImageOnlyMASt3R.__new__(ImageOnlyMASt3R);torch.nn.Module.__init__(encoder)
        encoder.patch_embed=Patch();encoder.enc_blocks=torch.nn.ModuleList([Block() for _ in range(24)])
        encoder.enc_norm=torch.nn.LayerNorm(1024)
        encoder.eval()
        wrapper=self.model.understanding
        with patch.object(wrapper,'encoder',encoder),patch.object(wrapper,'adapter',Adapter()):
            with self.assertRaises(StopIteration):
                wrapper(torch.zeros(1,2,3,256,256))
        self.assertEqual(len(received),24)
        self.assertIs(received[23],raw[23])
        torch.testing.assert_close(received[23],torch.full_like(received[23],24))
        self.assertFalse(torch.equal(received[23],encoder.enc_norm(raw[23])))
        self.assertIn('enc_norm.weight',self.model.understanding.encoder.state_dict())

    def test_bare_level_embeddings_pretrained_lr_and_zero_decay(self):
        optimizer=build_optimizer(self.model)
        for name in ('understanding.adapter.level_embed','understanding.mask2former.pixel_decoder.level_embed'):
            found=[group for group in optimizer.param_groups if name in group['param_names']]
            self.assertEqual(len(found),1)
            self.assertEqual(found[0]['name'],'pretrained_nodecay')
            self.assertEqual(found[0]['lr'],1e-5)
            self.assertEqual(found[0]['peak_lr'],1e-5)
            self.assertEqual(found[0]['weight_decay'],0)
            self.assertTrue(dict(self.model.named_parameters())[name].requires_grad)

    def test_tensor_axes_geometry_and_zero_injection(self):
        controller=self.model.panoptic
        h=torch.randn(1,1024,1024);mu=torch.randn(1,1024,3);r=torch.rand(1,1024)+0.1;ell=torch.tensor([1.])
        fm=torch.randn(1,2,256,128,128);qpre=torch.randn(1,100,256)
        a=controller.encode_token(h,mu,r,ell)
        q,c,s=controller.initialize_states(qpre,fm,a,mu,ell)
        self.assertEqual(q.shape,(1,102,256));self.assertEqual(c.shape,(1,100,3))
        image=torch.randn(1,2048,256)
        for layer in (6,8,10,12):
            f=controller.ln_mask_a(controller.W_mask_a(a))
            with torch.no_grad():out=controller.layers[f'L{layer}'](h,a,mu,q,c,s,ell,image,self.model.understanding.mask_embedder,f,320)
            self.assertEqual(out['evidence_attention'].shape,(1,8,102,1024))
            torch.testing.assert_close(out['evidence_attention'].sum(-1),torch.ones(1,8,102),rtol=1e-5,atol=1e-6)
            self.assertEqual(out['route'].shape,(1,1024,103))
            torch.testing.assert_close(out['route'].sum(-1),torch.ones(1,1024),rtol=1e-5,atol=1e-6)
            self.assertEqual(torch.count_nonzero(out['joint_delta']).item(),0)
            q,c,s=out['q'],out['c'],out['s']
            self.assertTrue(torch.isfinite(c).all() and torch.isfinite(s).all())
            self.assertTrue((s>=.05).all() and (s<=2).all())
        self.assertTrue((geometry_bias(mu*100,c,s).min() < -20).item())
        logits=torch.zeros(1,65536,102);membership=logits.sigmoid()
        self.assertEqual(membership.shape,(1,65536,102));self.assertEqual(membership.sum(-1)[0,0].item(),51)
        result=controller.classify(q)
        self.assertEqual(result['thing_logits19'].shape,(1,100,19));self.assertEqual(result['thing_class_logits'].shape,(1,100,21))
        torch.testing.assert_close(result['thing_class_logits'][...,:2],torch.full((1,100,2),-1e4))
        torch.testing.assert_close(result['conditional_class_prob'].sum(-1),torch.ones(1,100))
        self.assertEqual([100-100,101-100],[0,1])
        self.assertEqual([i-2 for i in range(2,20)],list(range(18)))

    def test_sampler_assets_and_exposure(self):
        manifest,plan,names,audit=assets()
        self.assertIn('same_scene_holdout8',names)
        seen=np.zeros(1008,dtype=np.int64)
        ranks=np.zeros(8,dtype=np.int64)
        for epoch in range(64):
            perm=np.random.default_rng(42+epoch).permutation(1008)
            grid=perm.reshape(126,8)
            self.assertEqual(len(np.unique(grid)),1008)
            for rank in range(8):seen[grid[:,rank]]+=1;ranks[rank]+=len(grid[:,rank])
        self.assertTrue((seen==64).all());self.assertTrue((ranks==8064).all())
        self.assertEqual(sample_index(0,0,0),int(np.random.default_rng(42).permutation(1008)[0]))
        self.assertFalse(audit['holdout_frame_leak'])
        self.assertEqual(audit['scene_intersections']['dev8'],[]);self.assertEqual(audit['scene_intersections']['val32'],[])
        self.assertAlmostEqual(lr_multiplier(0),8/200);self.assertAlmostEqual(lr_multiplier(8063),.1)

    def test_gc_eight_sample_reference(self):
        p=torch.nn.Parameter(torch.tensor([.3,-.1]));q=torch.nn.Parameter(torch.tensor([.2,.4]))
        names=['reconstruction','understanding.parameter'];params=[p,q]
        local=[]
        x=torch.arange(1,9,dtype=torch.float32)/8
        for sample in x:
            rec=(sample*p.sum()+q[0]).square();under=(p[0]-sample*q.sum()).square()
            gr=torch.autograd.grad(rec,params,retain_graph=True)
            gu=torch.autograd.grad(under,params)
            local.append(combine_gradients(params,names,gr,gu))
        rec_ref=torch.stack([(sample*p.sum()+q[0]).square() for sample in x]).mean()
        under_ref=torch.stack([(p[0]-sample*q.sum()).square() for sample in x]).mean()
        gr=torch.autograd.grad(rec_ref,params,retain_graph=True);gu=torch.autograd.grad(under_ref,params)
        expected=combine_gradients(params,names,gr,gu)
        for i in range(2): torch.testing.assert_close(torch.stack([v[i] for v in local]).mean(0),expected[i],rtol=1e-5,atol=1e-7)

class RankZeroEvalContract(unittest.TestCase):
    def test_local_ap_never_enters_distributed_collective(self):
        from scripts.eval_object_locus_panoptic_v1 import local_ap_metric
        metric=local_ap_metric()
        metric.distributed_available_fn=lambda: True
        def forbidden(*args,**kwargs):
            raise AssertionError('rank0 evaluation entered a distributed collective')
        metric._sync_dist=forbidden
        mask=torch.ones((1,4,4),dtype=torch.bool)
        metric.update([dict(masks=mask,scores=torch.ones(1),labels=torch.zeros(1,dtype=torch.long))],
                      [dict(masks=mask,labels=torch.zeros(1,dtype=torch.long))])
        self.assertEqual(float(metric.compute()['map_50']),1.0)


if __name__=='__main__':unittest.main()
