"""Stage transition identity, AdamW isolation, and official comparison contracts."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
import torch
from unittest.mock import patch
from scripts.dual_stream_training_policy import adaptation_only,validate_staged_source
from scripts.dual_stream_evaluation_checks import table_row,check_complete
from scripts.format_object_locus_posefree_siu3r_table import FIELDS
from scripts.invoke_siu3r_official_evaluator import SIU3R_COMMIT


class StagedScheduleTests(unittest.TestCase):
    def test_actual_task_reconstruction_branch_has_no_understanding_backward(self):
        from scripts.train_dual_stream_v1 import Task
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.rec=torch.nn.Parameter(torch.ones(1));self.under=torch.nn.Parameter(torch.ones(1))
            def step_loss(self,batch,*,step,understanding_weight):
                rec=self.rec.square().sum();under=self.under.square().sum()
                return {'prediction':None},dict(loss_recon=rec,loss=rec+understanding_weight*under)
        model=Model()
        with patch('scripts.train_dual_stream_v1.scalar_metrics',return_value={}):
            loss,_=Task(model)({},0,True);loss.backward()
        self.assertIsNotNone(model.rec.grad);self.assertIsNone(model.under.grad)

    def test_shared306_source_allows_only_the_documented_stage_change(self):
        saved=dict(recipe='omega_gfm_semantic_pool_v3_frozen_joint4',execution_git_sha='f5a6134260c66a4012d7850e2e6acf5c08a478a0',
            config=dict(recipe='omega_gfm_semantic_pool_v3_frozen_joint4',lr=1e-4),world_size=4,data='same')
        target=copy.deepcopy(saved);target.update(recipe='omega_gfm_semantic_pool_v3_adapt2_joint4_v1',execution_git_sha='b'*40,learning_gate_policy='learning_evidence_v2')
        target['config'].update(recipe=target['recipe'],adaptation_epochs=2)
        progress=dict(short_updates=306,full_updates=0,epoch=0)
        validate_staged_source(saved,target,progress)
        for bad in (dict(target,data='changed'),dict(target,world_size=2)):
            with self.assertRaises(ValueError):validate_staged_source(saved,bad,progress)
        with self.assertRaises(ValueError):validate_staged_source(saved,target,dict(progress,full_updates=1))

    def test_adaptation_preserves_understanding_weights_and_adam_moments(self):
        names=['reconstruction_adapter.weight','anchor_decoder.weight','understanding.weight','query_initializer.weight','panoptic.weight']
        parameters={n:torch.nn.Parameter(torch.ones(2)) for n in names}
        opt=torch.optim.AdamW(parameters.values(),lr=.01,weight_decay=.1)
        # Populate moments exactly as in the shared short-training checkpoint.
        sum(p.sum() for p in parameters.values()).backward();opt.step();opt.zero_grad(set_to_none=True)
        before={n:p.detach().clone() for n,p in parameters.items()}
        moments={n:copy.deepcopy(opt.state[p]) for n,p in parameters.items()}
        # Even a zero gradient would trigger AdamW decay/momentum. The stage must
        # remove it entirely, including a zero path introduced by disabled Write.
        for p in parameters.values():p.grad=torch.zeros_like(p)
        for n,p in parameters.items():
            if not adaptation_only(n):p.grad=None
        opt.step()
        for n,p in parameters.items():
            if adaptation_only(n):self.assertFalse(torch.equal(p,before[n]))
            else:
                self.assertTrue(torch.equal(p,before[n]))
                for key,value in moments[n].items():self.assertTrue(torch.equal(opt.state[p][key],value))

    def test_thirteen_columns_require_eleven_finite_values_and_blank_text(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'table.csv';row=['.5']*13;row[8]=row[12]=''
            def save(values):
                with path.open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerow(values)
            save(row);self.assertEqual(table_row(path),row)
            bad=list(row);bad[8]='.4';save(bad)
            with self.assertRaises(ValueError):table_row(path)
            bad=list(row);bad[0]='nan';save(bad)
            with self.assertRaises(ValueError):table_row(path)

    def test_duplicate_or_smoke_cohort_cannot_be_a_completed_official_eval(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'evaluation').mkdir()
            (root/'COMPLETE.json').write_text(json.dumps(dict(status='TRAINING_AND_HELDOUT128_OFFICIAL_EVAL_COMPLETE')))
            path=root/'evaluation/export_complete.json'
            path.write_text(json.dumps(dict(status='EXPORTED',records=[dict(name='duplicate')]*128)))
            with self.assertRaises(ValueError):check_complete(root)
            path.write_text(json.dumps(dict(status='SMOKE_EXPORTED',records=[dict(name='one')])) )
            with self.assertRaises(ValueError):check_complete(root)

    def test_complete_csv_must_agree_with_pinned_native_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'evaluation').mkdir()
            (root/'COMPLETE.json').write_text(json.dumps(dict(status='TRAINING_AND_HELDOUT128_OFFICIAL_EVAL_COMPLETE')))
            (root/'evaluation/export_complete.json').write_text(json.dumps(dict(status='EXPORTED',records=[dict(name=str(i)) for i in range(128)])))
            recon={k:.5 for k in ('absrel','rmse','psnr','ssim','lpips')};recon['mIoU_t']=None
            seg={}
            for prefix in ('context','target'):seg.update({prefix+'_miou':.5,prefix+'_pq':.5,prefix+'_map':{'map':.5}})
            report=dict(official_commit=SIU3R_COMMIT,scopes={'target-all':recon,'context':{'mIoU_t':None}},official_segmentation={'all':seg})
            (root/'evaluation/metrics.json').write_text(json.dumps(report))
            row=['.5']*13;row[8]=row[12]=''
            def save():
                with (root/'evaluation/siu3r_table1.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(FIELDS);writer.writerow(row)
            save();check_complete(root)
            row[0]='.6';save()
            with self.assertRaises(ValueError):check_complete(root)

if __name__=='__main__':unittest.main()
