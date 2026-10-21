"""A failed smoke must never submit a long run; a pass retains afterok."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import submit_fps_superpoint_after_smoke as gate

class GateTests(unittest.TestCase):
    def test_missing_or_failed_smoke_refuses_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);smoke=root/'smoke';smoke.mkdir();full=root/'full';full.mkdir()
            (full/'configuration.json').write_text(json.dumps(dict(smoke_root=str(smoke),code_sha='code')))
            with patch.object(gate.subprocess,'check_output') as call:
                with self.assertRaises(FileNotFoundError):gate.submit(full)
                call.assert_not_called()
                (smoke/'SMOKE_PASSED.json').write_text(json.dumps(dict(status='FAILED')))
                with self.assertRaises(ValueError):gate.submit(full)
                call.assert_not_called()
            self.assertFalse((full/'SUBMISSION_REQUESTED.json').exists())

    def test_pass_submits_once_with_dependency_and_bounded_single_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);smoke=root/'smoke';smoke.mkdir();full=root/'full';full.mkdir()
            (full/'configuration.json').write_text(json.dumps(dict(smoke_root=str(smoke),code_sha='code')))
            (smoke/'code.sha').write_text('code\n')
            (smoke/'SMOKE_PASSED.json').write_text(json.dumps(dict(status='PASSED',gpu_forward_backward_verified=True,ranks=8,anchors=1024,color_prior=False)))
            (smoke/'SUBMITTED.json').write_text(json.dumps(dict(smoke_job=123)))
            def output(argv,**kw):return 'code\n' if argv[0]=='git' else '456\n'
            with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(gate.subprocess,'check_output',side_effect=output) as submit,patch.object(gate.subprocess,'run') as status:
                status.return_value.stdout='456 PENDING Dependency';status.return_value.stderr=''
                gate.submit(full)
                argv=submit.call_args_list[-1].args[0]
                self.assertIn('--dependency=afterok:123',argv);self.assertIn('--gres=gpu:8',argv)
                status.assert_called_once();self.assertEqual(status.call_args.kwargs['timeout'],15)
                with self.assertRaises(FileExistsError):gate.submit(full)
                self.assertEqual(sum(c.args[0][0]=='sbatch' for c in submit.call_args_list),1)
