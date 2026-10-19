import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('dispatch',ROOT/'scripts/point1024_eval_flexible_dispatch.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)

class DispatchTests(unittest.TestCase):
    def test_running_or_started_training_never_eligible_for_cancellation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            self.assertTrue(d.is_unstarted('PENDING',p))
            for state in ['RUNNING','COMPLETING','COMPLETED','CANCELLED',None]:
                self.assertFalse(d.is_unstarted(state,p))
            for name in ['STARTED.json','checkpoint_latest.pt','checkpoint_initial.pt','TRAINING_COMPLETE.json']:
                f=p/name;f.touch();self.assertFalse(d.is_unstarted('PENDING',p));f.unlink()
    def test_existing_evaluation_is_never_resubmitted(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);self.assertTrue(d.evaluation_is_new(p))
            for name in ['EVALUATION_SUBMISSION.json','EVALUATION_SUBMISSION_REQUESTED.json','EVALUATION_STARTED.json','COMPLETE.json','evaluation/export_complete.json']:
                f=p/name;f.parent.mkdir(exist_ok=True);f.touch()
                self.assertFalse(d.evaluation_is_new(p));f.unlink()
    def test_evaluations_are_afterok_on_corresponding_node_one_gpu(self):
        for s in d.EVALS:
            c=d.evaluation_command(s)
            self.assertIn('--dependency=afterok:'+str(s['train_job']),c)
            self.assertIn('--nodelist='+s['node'],c)
            self.assertIn('--gres=gpu:'+s['gpu']+':1',c)
            self.assertIn('--kill-on-invalid-dep=yes',c)
    def test_flexible_training_is_one_node_generic_eight_gpus(self):
        c=d.flexible_command({'3dimage-11','3dimage-12'})
        self.assertIn('--partition=3090,4090',c)
        self.assertIn('--nodes=1',c);self.assertIn('--gres=gpu:8',c)
        self.assertIn('--exclude=3dimage-11,3dimage-12',c)
        self.assertFalse(any(x.startswith('--nodelist') for x in c))
    def test_allowed_pool_is_exactly_requested_four_nodes(self):
        self.assertEqual(d.ALLOWED,{'3dimage-13','3dimage-14','3dimage-17','3dimage-18'})
    def test_scheduler_state_parser_distinguishes_pending_and_running(self):
        self.assertEqual(d.state_from_oneline('JobId=67976 JobState=RUNNING Reason=None'),'RUNNING')
        self.assertEqual(d.state_from_oneline('JobState=PENDING Reason=JobHeldUser'),'PENDING')
    def test_recipe_compatibility_only_changes_labels(self):
        tree=ast.parse((ROOT/'scripts/score_dual_stream_v1.py').read_text())
        f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='resolve_recipe')
        namespace={};exec(compile(ast.Module(body=[f],type_ignores=[]),'recipe','exec'),namespace)
        resolve=namespace['resolve_recipe']
        self.assertEqual(resolve({'recipe':'old','architecture':'new'}),'old')
        self.assertEqual(resolve({'architecture':'new'}),'new')
        with self.assertRaises(ValueError):resolve({})
    def test_native_metric_helpers_are_byte_identical_ast(self):
        old=ast.parse(Path('/space/mawb/ssst_vggt_overfit_diagnostics_v1/scripts/score_dual_stream_v1.py').read_text())
        new=ast.parse((ROOT/'scripts/score_dual_stream_v1.py').read_text())
        for name in ['itemize','native_table_values','json_safe']:
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))

if __name__=='__main__':unittest.main(verbosity=2)
