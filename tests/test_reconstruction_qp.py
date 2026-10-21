import io
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import torch
from torch import nn
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(Path(__file__).resolve().parents[1],log=lambda *args:None)
from tokengs.models.reconstruction_qp import LearnedReconstructionAnchorExpansion
from tokengs.models.superpoint_structural_variants import LocusGSSuperpointDualReconQP
from tokengs.models.unified_query_loss import unified_instance_loss
from scripts.recon_qp_initial_state import BANK_KEY,initialize_recon_qp_from_legacy_zero

class ExpansionTests(unittest.TestCase):
    def test_cartesian_sum_geometry_order_and_no_truncation(self):
        e=LearnedReconstructionAnchorExpansion(20,1024)
        p=torch.randn(1,50,1024);c=torch.randn(1,50,3);r=torch.rand(1,50)+.1
        x=e(p,c,r);self.assertEqual(x['tokens'].shape,(1,1000,1024))
        for i,j in [(0,0),(0,19),(1,0),(49,19)]:
            k=i*20+j
            torch.testing.assert_close(x['tokens'][:,k],p[:,i]+e.learned_q[j])
            torch.testing.assert_close(x['centers'][:,k],c[:,i]);torch.testing.assert_close(x['radii'][:,k],r[:,i])
            self.assertEqual(int(x['anchor_to_superpoint'][k]),i);self.assertEqual(int(x['anchor_slot_ids'][k]),j)
        small=LearnedReconstructionAnchorExpansion(20,8)
        self.assertEqual(small(torch.randn(1,251,8),torch.randn(1,251,3),torch.ones(1,251))['tokens'].shape[1],5020)
    def test_persistent_q_gets_gradient_updates_and_checkpoint_roundtrip(self):
        e=LearnedReconstructionAnchorExpansion(3,8);p=torch.randn(1,2,8,requires_grad=True)
        before=e.learned_q.detach().clone();parameter_id=id(e.learned_q)
        opt=torch.optim.SGD(e.parameters(),lr=.1)
        out=e(p,torch.zeros(1,2,3),torch.ones(1,2))
        target=torch.randn_like(out['tokens']);(out['tokens']-target).square().mean().backward()
        self.assertGreater(float(p.grad.norm()),0);self.assertTrue((e.learned_q.grad.norm(dim=-1)>0).all())
        opt.step();self.assertFalse(torch.equal(before,e.learned_q));self.assertEqual(id(e.learned_q),parameter_id)
        next_p=torch.randn(1,1,8);next_out=e(next_p,torch.zeros(1,1,3),torch.ones(1,1))
        torch.testing.assert_close(next_out['tokens'][0,2],next_p[0,0]+e.learned_q[2])
        stream=io.BytesIO();torch.save(e.state_dict(),stream);stream.seek(0)
        restored=LearnedReconstructionAnchorExpansion(3,8);restored.load_state_dict(torch.load(stream,weights_only=True))
        torch.testing.assert_close(restored.learned_q,e.learned_q)
    def test_one_zero_q_reproduces_original_seed_and_invalid_shapes_fail(self):
        e=LearnedReconstructionAnchorExpansion(1,8);nn.init.zeros_(e.learned_q)
        p=torch.randn(2,3,8);c=torch.randn(2,3,3);r=torch.rand(2,3)
        x=e(p,c,r)
        torch.testing.assert_close(x['tokens'],p);torch.testing.assert_close(x['centers'],c);torch.testing.assert_close(x['radii'],r)
        for k in (0,-1,True,1.5):
            with self.assertRaises(ValueError):LearnedReconstructionAnchorExpansion(k)
        with self.assertRaises(ValueError):e(p,c[:,:2],r)

class QPModelTests(unittest.TestCase):
    def test_full_native_core_decouples_anchor_and_object_counts_and_backpropagates(self):
        from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
        cfg=replace(config(),dim=256,heads=8,depth=12,superpoint_source='esam_sam',matching_mode='source_many_to_one')
        model=LocusGSSuperpointDualReconQP(opt().evolve(dec_patch_size=8,img_size=(8,8)),config=cfg,
            omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer(),recon_qp_slots=3)
        def small_memory(layers,rgb):
            module=model.core.semantic_memory;b,v=rgb.shape[:2]
            patches=sum(w*proj(x.float().mean(-2)) for w,proj,x in zip(module.layer_logits.softmax(0),module.projections,layers))
            grid=patches.reshape(b*v,256,1,1).expand(-1,-1,8,8)
            local=module.local(torch.nn.functional.interpolate(rgb.flatten(0,1),size=(8,8),mode='bilinear',align_corners=False))
            return module.refine(grid+local).reshape(b,v,256,8,8)
        self.assertEqual(model.supervised_layers,(6,12))
        with patch.object(model.core.semantic_memory,'forward',side_effect=small_memory):
            out=model.generate(torch.rand(1,2,3,256,256))
            n=out['num_thing_queries'];a=n*3;g=a*64
            self.assertEqual(n,2);self.assertEqual(out['num_reconstruction_anchors'],6)
            self.assertEqual(out['states'][-1]['tokens'].shape,(1,a,1024))
            self.assertEqual(out['unified_query'].shape,(1,n,256))
            self.assertEqual(out['gaussians'].shape,(1,g,14));self.assertEqual(out['gaussian_membership'].shape,(1,g,n+2))
            self.assertEqual(out['p_class'].shape,(1,n,19));self.assertEqual(out['num_gaussians'],g)
            self.assertEqual(out['anchor_to_superpoint'].tolist(),[0,0,0,1,1,1])
            self.assertEqual(out['gaussian_parent_query'].tolist(),[0]*(3*64)+[1]*(3*64))
            self.assertEqual(out['gaussian_parent_anchor'].tolist(),torch.arange(a).repeat_interleave(64).tolist())
            batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
            semloss,metrics=unified_instance_loss(out,batch,cfg)
            self.assertEqual(metrics['matched_query_count'],n)
            loss=out['gaussians'].square().mean()+.05*semloss
            loss.backward();self.assertGreater(float(model.core.reconstruction_qp.learned_q.grad.norm()),0.)
            self.assertGreater(float(model.core.initializer.understanding.point_encoder.layers[0][0].weight.grad.norm()),0.)
            groups=model.parameter_groups();q=model.core.reconstruction_qp.learned_q
            self.assertIn(id(q),{id(p) for p in groups['reconstruction']});self.assertNotIn(id(q),{id(p) for p in groups['understanding']})
            all_ids=[id(p) for ps in groups.values() for p in ps]
            self.assertEqual(len(all_ids),len(set(all_ids)));self.assertEqual(set(all_ids),{id(p) for p in model.parameters() if p.requires_grad})
            # K=1,Q=0 must reproduce the actual original core's forward on the
            # same model weights. No second large model or checkpoint is written.
            import importlib.util
            spec=importlib.util.spec_from_file_location('original_qp_control_core','/space/mawb/ssst_iouhead_code_20261020_v1/tokengs/models/superpoint_dual_core.py')
            original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
            model.core.configure_qp_slots(1);nn.init.zeros_(model.core.reconstruction_qp.learned_q)
            rgb=out['generation_context_rgb']
            with torch.no_grad():
                candidate=model.generate(rgb)
                with patch.object(model.core,'forward',side_effect=lambda *a,**kw:original.SuperpointDualCore.forward(model.core,*a,**kw)):
                    control=model.generate(rgb)
            for key in ('gaussians','gaussian_membership','p_class','semantic_scores','region_mass'):
                torch.testing.assert_close(candidate[key],control[key],rtol=1e-6,atol=1e-7)


class FakeQP(nn.Module):
    architecture_name='OMEGA_3D_SUPERPOINT_RECON_QP_V1'
    def __init__(self):
        super().__init__()
        from types import SimpleNamespace
        self.core=nn.Module();self.core.baseline=nn.Linear(3,3);self.core.reconstruction_qp=LearnedReconstructionAnchorExpansion(2,1024)
        self.config=SimpleNamespace(to_dict=lambda:{'dim':256});self.frozen_omega=SimpleNamespace(source_identity={'sha':'omega'});self.sam_proposals=None

class InitializationTests(unittest.TestCase):
    def test_exactly_one_new_bank_and_no_other_relaxation(self):
        import copy
        model=FakeQP();state={k:torch.full_like(v,.3) for k,v in model.state_dict().items() if k!=BANK_KEY}
        saved=dict(model=state,meta=dict(architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config={'dim':256},global_batch=8,world_size=8,manifest_sha256='train',heldout_sha256='val',omega_identity={'sha':'omega'},sam_identity=None),progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer=dict(state={}),rank_rng=[{}]*8)
        before=model.core.reconstruction_qp.learned_q.detach().clone()
        def apply(s):return initialize_recon_qp_from_legacy_zero(s,model,manifest_sha256='train',heldout_sha256='val',world=4)
        result=apply(saved);self.assertEqual(result['new_parameter'],BANK_KEY)
        torch.testing.assert_close(before,model.core.reconstruction_qp.learned_q)
        for k in state:torch.testing.assert_close(model.state_dict()[k],state[k])
        for case in ('data','trained','adam','missing','extra','shape','bank_in_source','rng'):
            s=copy.deepcopy(saved)
            if case=='data':s['meta']['heldout_sha256']='wrong'
            if case=='trained':s['progress']['full_updates']=1
            if case=='adam':s['optimizer']['state'][0]={'step':1}
            if case=='missing':del s['model']['core.baseline.bias']
            if case=='extra':s['model']['extra']=torch.zeros(1)
            if case=='shape':s['model']['core.baseline.weight']=torch.zeros(1)
            if case=='bank_in_source':s['model'][BANK_KEY]=before
            if case=='rng':s['rank_rng']=[{}]
            with self.subTest(case=case):
                with self.assertRaises(ValueError):apply(s)
if __name__=='__main__':unittest.main()
