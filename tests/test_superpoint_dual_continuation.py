import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.superpoint_dual_continuation import POLICY,validate_source,validate_optimizer_layout,continuation_lr,validate_epoch_export
from scripts.dual_stream_training_policy import epoch_order,rank_microbatches


class ContinuationTests(unittest.TestCase):
    def fixture(self):
        meta=dict(architecture='OMEGA_3D_SUPERPOINT_LEGACY_DUAL_V1',config={'fixed':True},code_sha='old',
            manifest_sha256='train',heldout_sha256='heldout',omega_identity={'sha':'omega'},sam_identity={'sha':'sam'},
            initial_checkpoint='zero',initial_sha256='zero-sha',initial_updates=0,epoch_probes=True,
            short_validation=False,training_windows=None,world_size=8,global_batch=8,epochs=4,
            understanding_weight=.05,understanding_warmup=1043,learning_rate=1e-4)
        saved=dict(meta=meta,progress=dict(epoch=4,update_in_epoch=0,full_updates=4172),rank_rng=[{} for _ in range(8)],
            model={str(i):torch.tensor(i) for i in range(693)},optimizer={'state':{i:{'step':torch.tensor(4172.)} for i in range(693)}})
        current=dict(meta,code_sha='new',epochs=8,learning_rate=3e-5,initial_checkpoint=None,initial_sha256=None,
            continuation_checkpoint='epoch4',continuation_sha256='source-sha',continuation_policy=POLICY)
        return saved,current

    def test_exact_source_and_reject_policy_data_progress_changes(self):
        saved,current=self.fixture();self.assertEqual(validate_source(saved,current)['full_updates'],4172)
        for key,value in [('config',{}),('manifest_sha256','other'),('heldout_sha256','other'),('world_size',4),('understanding_weight',1.),('epochs',9),('learning_rate',1e-4)]:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_source(saved,dict(current,**{key:value}))
        for progress in [dict(epoch=0,update_in_epoch=0,full_updates=0),dict(epoch=4,update_in_epoch=1,full_updates=4173)]:
            with self.assertRaises(ValueError):validate_source(dict(saved,progress=progress),current)
        broken=copy.deepcopy(saved);broken['optimizer']['state'][0]['step']=0
        with self.assertRaises(ValueError):validate_source(broken,current)

    def test_schedule_not_replayed_and_reaches_floor(self):
        self.assertAlmostEqual(continuation_lr(4172),1e-5)
        self.assertAlmostEqual(continuation_lr(4271),3e-5)
        self.assertAlmostEqual(continuation_lr(4272),3e-5)
        self.assertAlmostEqual(continuation_lr(8343),3e-6)
        for step in (0,4171,8344):
            with self.assertRaises(ValueError):continuation_lr(step)
        order=epoch_order(8337,4);indices=[rank_microbatches(order,0,r,8)[0] for r in range(8)]
        self.assertEqual(indices,order[:8]);self.assertNotEqual(order,epoch_order(8337,0))

    def test_adam_moments_preserved_and_next_update_matches_reference(self):
        p=torch.nn.Parameter(torch.tensor([.2,.3]));old=torch.optim.AdamW([p],lr=1e-4,betas=(.9,.95))
        p.grad=torch.tensor([.1,-.2]);old.step();source=copy.deepcopy(old.state_dict())
        q=torch.nn.Parameter(p.detach().clone());new=torch.optim.AdamW([q],lr=3e-5,betas=(.9,.95))
        validate_optimizer_layout(new,source);new.load_state_dict(source)
        self.assertTrue(torch.equal(old.state[p]['exp_avg'],new.state[q]['exp_avg']))
        for opt,param in [(old,p),(new,q)]:
            opt.param_groups[0]['lr']=continuation_lr(4172);param.grad=torch.tensor([-.2,.5]);opt.step()
        self.assertTrue(torch.equal(p,q));self.assertEqual(float(new.state[q]['step']),2.)
        source['state'][0]['exp_avg']=torch.zeros(3)
        with self.assertRaises(ValueError):validate_optimizer_layout(new,source)

    def test_epoch6_requires_explicit_completed_checkpoint(self):
        saved,current=self.fixture();saved['meta']=current;saved['progress']=dict(epoch=6,update_in_epoch=0,full_updates=6258)
        self.assertEqual(validate_epoch_export(saved,6),6)
        with self.assertRaises(ValueError):validate_epoch_export(saved)
        with self.assertRaises(ValueError):validate_epoch_export(saved,8)
        with self.assertRaises(ValueError):validate_epoch_export(saved,5)
        saved['progress']['update_in_epoch']=1
        with self.assertRaises(ValueError):validate_epoch_export(saved,6)

    def test_final_comparison_preserves_epoch_identity_and_deltas(self):
        from scripts.compare_superpoint_dual_extend import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'continuation';root.mkdir();(root/'preflight.json').write_text(json.dumps({'source_sha256':'source'}))
            for epoch in (6,8):
                folder=root/f'official_epoch{epoch}'/'evaluation';folder.mkdir(parents=True)
                (folder/'checkpoint_identity.json').write_text(json.dumps(dict(evaluated_epoch=epoch,parameter_updates=0,meta=dict(continuation_policy=POLICY,continuation_sha256='source'))))
            row=['1']*13;row[8]=row[12]=''
            def complete(run):
                values=list(row)
                if run.name=='official_epoch8':values[2]='2'
                return values,{'records':[dict(name='scene',source_index=0,frame_ids=[1,2],input_rgb_sha256='rgb')]}
            with patch('sys.argv',['compare','--run-root',str(root)]),patch('scripts.compare_superpoint_dual_extend.check_complete',side_effect=complete),patch('scripts.compare_superpoint_dual_extend.table_row',return_value=row):main()
            rows=list(csv.reader((root/'comparison_epoch4_6_8/comparison.csv').open()))
            self.assertEqual([r[0] for r in rows[1:5]],['old_C','A_epoch4','A_epoch6','A_epoch8'])
            self.assertEqual(float(rows[-1][3]),1.)
            self.assertEqual(rows[-1][9],'')

    def test_final_comparison_rejects_different_rgb(self):
        from scripts.compare_superpoint_dual_extend import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'continuation';root.mkdir()
            def complete(run):return ['1']*13,{'records':[dict(name='scene',source_index=0,frame_ids=[1,2],input_rgb_sha256=run.name)]}
            with patch('sys.argv',['compare','--run-root',str(root)]),patch('scripts.compare_superpoint_dual_extend.check_complete',side_effect=complete),self.assertRaises(ValueError):main()


if __name__=='__main__':unittest.main()
