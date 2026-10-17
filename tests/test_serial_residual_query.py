import copy,json,math,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import torch
import torch.nn.functional as F
from tests.test_unified_query_v1 import config,opt,FakeOmega,FakeSAM,FakeRenderer
from tests.test_serial_query import GeometryRenderer
from tokengs.models.superpoint_structural_variants import LocusGSSerialQuery,LocusGSSerialResidualQuery
from scripts.serial_residual_initial import prepare_zero,NEW_KEYS
from scripts.diagnose_serial_query_masks import observe


class ResidualTests(unittest.TestCase):
    def model(self,residual=True,cfg=None):
        cls=LocusGSSerialResidualQuery if residual else LocusGSSerialQuery
        return cls(opt(),config=cfg or config(),omega=FakeOmega(),sam=FakeSAM(),renderer=GeometryRenderer())

    def test_seed_is_exact_old_seed_plus_small_normalized_identity(self):
        model=self.model();core=model.core;d=config().dim
        q=torch.randn(1,4,d,requires_grad=True);initial=torch.randn(1,4,d,requires_grad=True)
        expected=core.semantic_seed(q.detach())+.1*F.layer_norm(initial.detach(),(d,))
        torch.testing.assert_close(core.make_semantic_seed(q,initial),expected,rtol=1e-5,atol=1e-6)
        core.make_semantic_seed(q,initial).square().mean().backward()
        self.assertIsNone(q.grad);self.assertIsNone(initial.grad)
        self.assertGreater(float(core.identity_projection.weight.grad.abs().sum()),0.)
        self.assertGreater(float(core.identity_gate_logit.grad.abs()),0.)

    def test_extra_constructor_preserves_rng_and_old_parameters(self):
        torch.manual_seed(42);old=self.model(False);expected=torch.get_rng_state()
        torch.manual_seed(42);new=self.model();self.assertTrue(torch.equal(torch.get_rng_state(),expected))
        self.assertTrue(all(torch.equal(new.state_dict()[k],v) for k,v in old.state_dict().items()))

    def test_zero_gate_reproduces_old_model_and_reconstruction_is_unchanged(self):
        old=self.model(False);new=self.model();missing,extra=new.load_state_dict(old.state_dict(),strict=False)
        self.assertEqual(set(missing),NEW_KEYS);self.assertFalse(extra)
        rgb=torch.rand(1,2,3,256,256);a=old.generate(rgb);b=new.generate(rgb)
        torch.testing.assert_close(a['gaussians'],b['gaussians'],rtol=0,atol=0)
        with torch.no_grad():new.core.identity_gate_logit.fill_(-float('inf'))
        b=new.generate(rgb)
        for key in ['semantic_query_initial','understanding_query','gaussian_membership','thing_logits19']:
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_complete_understanding_path_does_not_update_reconstruction(self):
        model=self.model();p=model.generate(torch.rand(1,2,3,256,256))
        (p['thing_logits19'].square().mean()+p['region_mass'].square().mean()).backward()
        groups=model.parameter_groups();self.assertFalse(groups['shared'])
        self.assertTrue(all(v.grad is None for v in groups['reconstruction']))
        self.assertFalse(model.gs.understanding_geometry_requires_grad)
        self.assertGreater(float(model.core.identity_projection.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.core.identity_gate_logit.grad.abs()),0.)
        ids=[id(p) for ps in groups.values() for p in ps];self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})

    def fixture(self):
        cfg=replace(config(),superpoint_source='esam_sam',matching_mode='source_many_to_one')
        old=self.model(False,cfg);new=self.model(True,cfg);windows=[dict(scene=str(i)) for i in range(16)]
        meta=dict(architecture=old.architecture_name,config=cfg.to_dict(),world_size=4,global_batch=8,short_validation=True,
            epochs=128,epoch_probes=False,learning_rate=1e-4,understanding_weight=.05,understanding_warmup=20,training_windows=windows,
            manifest_sha256='train',heldout_sha256='heldout',omega_identity=new.frozen_omega.source_identity,sam_identity=new.sam_proposals.source_identity)
        state={k:v for k,v in old.state_dict().items() if not k.startswith(('frozen_omega.','sam_proposals.'))}
        saved=dict(meta=meta,progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer={'state':{}},rank_rng=[{} for _ in range(4)],model=state)
        return saved,new,dict(windows=windows,manifest_sha256='train',heldout_sha256='heldout')

    def test_zero_transfer_preserves_every_old_tensor_and_rng(self):
        saved,model,kwargs=self.fixture();out=prepare_zero(saved,model,**kwargs)
        self.assertEqual(set(out['model'])-set(saved['model']),NEW_KEYS)
        self.assertTrue(all(torch.equal(v,out['model'][k]) for k,v in saved['model'].items()))
        self.assertIs(out['rank_rng'],saved['rank_rng']);self.assertFalse(out['optimizer']['state'])
        for key,value in [('full_updates',256),('epoch',1)]:
            bad=copy.deepcopy(saved);bad['progress'][key]=value
            with self.assertRaises(ValueError):prepare_zero(bad,model,**kwargs)
        bad=copy.deepcopy(saved);bad['model'].pop(next(iter(bad['model'])))
        with self.assertRaises(ValueError):prepare_zero(bad,model,**kwargs)
        with self.assertRaises(ValueError):prepare_zero(saved,model,**dict(kwargs,windows=[]))

    def test_residual_same_native_replay_and_immutability(self):
        model=self.model().eval().requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()}
        batch=dict(images_input=torch.rand(1,2,3,256,256),semantic_label_all=torch.full((1,2,8,8),2),instance_label_all=torch.ones(1,2,8,8,dtype=torch.long))
        with tempfile.TemporaryDirectory() as folder,patch('scripts.diagnose_serial_query_masks.build_batch',return_value=batch):
            result=observe(model,opt(),dict(scene='synthetic',context=[0,1]),4,'train',torch.device('cpu'),True,True,Path(folder))
        self.assertTrue(result['same_forward_replay_verified']);self.assertTrue(result['final_head_replay_verified'])
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_summary_reuses_baseline_preserves_warnings_and_rejects_gt_mismatch(self):
        from scripts.summarize_serial_residual_short import main
        from scripts.train_dual_stream_v1 import digest
        from scripts.serial_residual_initial import RESIDUAL_ARCH
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'diagnostics';(out/'residual').mkdir(parents=True);baseline=root/'old_diagnostics'/'serial';baseline.mkdir(parents=True)
            old={f'p{i}':torch.tensor(0.) for i in range(383)};rng=[{} for _ in range(4)]
            source=root/'source_zero.pt';torch.save(dict(model=old,rank_rng=rng),source)
            initial=dict(old,**{'core.identity_gate_logit':torch.tensor(math.log(.1/.9)),'core.identity_projection.weight':torch.eye(2),'core.identity_projection.bias':torch.zeros(2)})
            torch.save(dict(model=initial,rank_rng=rng,progress=dict(epoch=0,update_in_epoch=0,full_updates=0),optimizer={'state':{}}),root/'checkpoint_initial.pt')
            final={k:v+.01 for k,v in initial.items()}
            torch.save(dict(model=final,meta=dict(architecture=RESIDUAL_ARCH,short_validation=True),progress={'full_updates':256},optimizer={'state':{i:{'step':torch.tensor(256.)} for i in range(386)}}),root/'checkpoint_latest.pt')
            final_sha=digest(root/'checkpoint_latest.pt')
            for rank in range(4):
                (out/f'UNCHANGED_rank{rank:02}.json').write_text(json.dumps(dict(unchanged=True,optimizer_created=False,parameter_updates=0,checkpoint_sha256=final_sha)))
                (root/f'SOURCE_RNG_VERIFIED_rank{rank:02}.json').write_text(json.dumps(dict(exact_python_numpy_torch_cuda_rng=True)))
            (out/'INFERENCE_COMPLETE.json').write_text(json.dumps(dict(windows=32,parameter_updates=0,baseline_reused=True)))
            hashes={}
            for i in range(32):
                row=dict(index=i,scope='train' if i<16 else 'heldout',scene=str(i),context=[0,1],input_rgb_sha256='rgb',gt_ids=[1],gt_classes=[2],gt_count=1,supported_gt=1,query_count=2,class_eligible=2,nms_kept=1,final_instances=1,queries_no_rendered_half=1,queries_gaussian_below_half=1,same_forward_replay_verified=True,final_head_replay_verified=True,native_repeat_warning=i==0,witness={'passed':i!=0},representation={'seed':{'cosine_separation':.2}})
                stage={kind:dict(best_iou=[.2],gt_count=1,best_iou_mean=.2,recall_iou50=0.) for kind in ('class_agnostic','correct_class','source_supported_only')}
                row['stage_metrics']={k:copy.deepcopy(stage) for k in ('raw_alpha','final')}
                path=baseline/f'{i:02d}.json';path.write_text(json.dumps(row));hashes[str(i)]=digest(path)
                row['native_repeat_warning']=False
                for stage in row['stage_metrics'].values():
                    for stats in stage.values():stats['best_iou']=[.4]
                (out/'residual'/f'{i:02d}.json').write_text(json.dumps(row))
            (root/'preflight.json').write_text(json.dumps(dict(baseline_diagnostics=str(baseline.parent),baseline_window_sha256=hashes,source_initial=str(source),source_initial_sha256=digest(source))))
            with patch('sys.argv',['summary','--run-root',str(root)]):main()
            report=json.loads((out/'SUMMARY.json').read_text());self.assertTrue(report['baseline_reused']);self.assertEqual(len(report['native_repeat_warnings']),1)
            self.assertAlmostEqual(report['models']['identity_residual']['heldout']['stage_metrics']['raw_alpha']['class_agnostic']['mean_iou'],.4)
            self.assertTrue((out/'gallery.html').exists());self.assertTrue((root/'COMPLETE.json').exists())
            path=out/'residual/31.json';row=json.loads(path.read_text());row['gt_classes']=[3];path.write_text(json.dumps(row))
            with patch('sys.argv',['summary','--run-root',str(root)]),self.assertRaises(AssertionError):main()


if __name__=='__main__':unittest.main()
