import io
import unittest
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import torch
from torch import nn
from scripts.runtime_bootstrap import prepare_runtime
prepare_runtime(Path(__file__).resolve().parents[1],log=lambda *args:None)
from tokengs.models.reconstruction_fps_superpoint import deterministic_fps,FPSWithSuperpointFeatures
from tokengs.models.superpoint_structural_variants import LocusGSSuperpointDualFPSSuperpoint
from tokengs.models.unified_query_loss import unified_instance_loss

class FPSSeedTests(unittest.TestCase):
    def test_global_fps_unique_deterministic_and_small_cloud_repeat(self):
        x=torch.tensor([[0.,0,0],[1,0,0],[2,0,0],[3,0,0]])
        self.assertEqual(deterministic_fps(x,4).tolist(),[0,3,1,2])
        self.assertEqual(deterministic_fps(x,6).tolist(),[0,3,1,2,0,3])
        self.assertEqual(deterministic_fps(torch.zeros(5,3),5).unique().numel(),5)
        for bad in (torch.empty(0,3),torch.tensor([[float('nan'),0,0]])):
            with self.assertRaises(ValueError):deterministic_fps(bad,1024)

    def test_1024_exact_add_sampled_positions_gradients_and_persistence(self):
        torch.manual_seed(7)
        module=FPSWithSuperpointFeatures()
        points=torch.randn(1100,3);ids=torch.arange(1100)%7
        features=torch.randn(1,7,1024,requires_grad=True)
        seed=dict(point_superpoint_ids=ids,radii=torch.arange(1.,8.)[None])
        cloud=dict(points=points)
        out=module(cloud,seed,features);idx=out['anchor_point_indices'];parent=ids[idx]
        self.assertEqual(out['tokens'].shape,(1,1024,1024));self.assertEqual(idx.unique().numel(),1024)
        torch.testing.assert_close(out['tokens'],module.learned_embedding[None]+features[:,parent])
        torch.testing.assert_close(out['centers'],points[idx][None]);torch.testing.assert_close(out['radii'],seed['radii'][:,parent])
        self.assertEqual(out['num_sampled_superpoints'],parent.unique().numel())
        before=module.learned_embedding.detach().clone();optimizer=torch.optim.SGD(module.parameters(),lr=.1)
        out['tokens'].square().mean().backward()
        self.assertGreater(float(features.grad.norm()),0);self.assertTrue((module.learned_embedding.grad.norm(dim=-1)>0).all())
        optimizer.step();self.assertFalse(torch.equal(before,module.learned_embedding))
        buf=io.BytesIO();torch.save(module.state_dict(),buf);buf.seek(0)
        restored=FPSWithSuperpointFeatures();restored.load_state_dict(torch.load(buf,weights_only=True))
        torch.testing.assert_close(restored(cloud,seed,features)['tokens'],module(cloud,seed,features)['tokens'])

class FPSCoreTests(unittest.TestCase):
    def test_full_native_core_decouples_anchor_and_object_counts_and_backpropagates(self):
        from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
        cfg=replace(config(),dim=256,heads=8,depth=12,superpoint_source='esam_sam',matching_mode='source_many_to_one')
        model=LocusGSSuperpointDualFPSSuperpoint(opt().evolve(dec_patch_size=8,img_size=(8,8)),config=cfg,
            omega=FakeOmega(),sam=FakeSAM(),renderer=FakeRenderer())
        model.core.reconstruction_fps=FPSWithSuperpointFeatures(count=6)
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
            index=out['anchor_point_indices'];parent=out['superpoints']['point_superpoint_ids'][index]
            torch.testing.assert_close(out['anchor_to_superpoint'],parent)
            torch.testing.assert_close(out['initial_anchor_mu'],out['cloud']['points'][index][None])
            torch.testing.assert_close(out['gaussian_parent_query'],parent.repeat_interleave(64))
            self.assertEqual(out['gaussian_parent_anchor'].tolist(),torch.arange(a).repeat_interleave(64).tolist())
            batch=dict(semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
            semloss,metrics=unified_instance_loss(out,batch,cfg)
            self.assertEqual(metrics['matched_query_count'],n)
            loss=out['gaussians'].square().mean()+.05*semloss
            loss.backward();self.assertGreater(float(model.core.reconstruction_fps.learned_embedding.grad.norm()),0.)
            self.assertGreater(float(model.core.initializer.understanding.point_encoder.layers[0][0].weight.grad.norm()),0.)
            groups=model.parameter_groups();q=model.core.reconstruction_fps.learned_embedding
            self.assertIn(id(q),{id(p) for p in groups['reconstruction']});self.assertNotIn(id(q),{id(p) for p in groups['understanding']})
            all_ids=[id(p) for ps in groups.values() for p in ps]
            self.assertEqual(len(all_ids),len(set(all_ids)));self.assertEqual(set(all_ids),{id(p) for p in model.parameters() if p.requires_grad})



from scripts.recon_fps_superpoint_initial_state import BANK_KEY,initialize_fps_superpoint_from_legacy_zero
class FakeFPS(nn.Module):
    architecture_name='OMEGA_3D_FPS1024_SUPERPOINT_RECON_V1'
    def __init__(self):
        super().__init__()
        from types import SimpleNamespace
        self.core=nn.Module();self.core.baseline=nn.Linear(3,3);self.core.reconstruction_fps=FPSWithSuperpointFeatures(count=2)
        self.config=SimpleNamespace(to_dict=lambda:{'dim':256});self.frozen_omega=SimpleNamespace(source_identity={'sha':'omega'});self.sam_proposals=None

class InitializationTests(unittest.TestCase):
    def test_exactly_one_new_bank_and_no_other_relaxation(self):
        import copy
        model=FakeFPS();state={k:torch.full_like(v,.3) for k,v in model.state_dict().items() if k!=BANK_KEY}
        saved=dict(model=state,meta=dict(architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config={'dim':256},global_batch=8,world_size=8,manifest_sha256='train',heldout_sha256='val',omega_identity={'sha':'omega'},sam_identity=None),progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer=dict(state={}),rank_rng=[{}]*8)
        before=model.core.reconstruction_fps.learned_embedding.detach().clone()
        def apply(s):return initialize_fps_superpoint_from_legacy_zero(s,model,manifest_sha256='train',heldout_sha256='val',world=4)
        result=apply(saved);self.assertEqual(result['new_parameter'],BANK_KEY)
        torch.testing.assert_close(before,model.core.reconstruction_fps.learned_embedding)
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
