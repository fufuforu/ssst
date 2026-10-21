"""Recovery must not retrain success or retry unrelated failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from scripts import continue_staged_after_cache_failure as recovery


class CacheRecoveryTests(unittest.TestCase):
    def test_completed_parent_is_reused_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(full_updates=4172,adaptation_updates=2086)))
            with patch.object(recovery,'ROOT',root),patch.object(recovery.os,'execv') as execute:recovery.main()
            execute.assert_not_called();self.assertTrue((root/'CACHE_RECOVERY_NOT_NEEDED.json').exists())

    def test_unrelated_failure_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'FAILED_rank0.json').write_text(json.dumps(dict(job_id='65236',error='FloatingPointError')))
            with patch.object(recovery,'ROOT',root),patch.object(recovery.os,'execv') as execute:
                with self.assertRaises(RuntimeError):recovery.main()
            execute.assert_not_called()

    def test_cache_error_preserves_checkpoint_and_resumes_stage_cursor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            error=dict(job_id='65236',error='FileExistsError',traceback='dual_stream_frozen_cache.py /frozen_cache/key.pt')
            (root/'FAILED_rank0.json').write_text(json.dumps(error))
            progress=dict(phase='adapt',adapt_updates=150,full_updates=0)
            torch.save(dict(meta={'execution_git_sha':recovery.SOURCE_SHA},progress=progress),root/'checkpoint_latest.pt')
            with patch.object(recovery,'ROOT',root),patch.object(recovery.os,'chdir'),patch.object(recovery.os,'execv') as execute:recovery.main()
            args=execute.call_args.args[1]
            self.assertIn('--resume',args);self.assertNotIn('--staged-from',args)
            archive=root/'cache_failure_recovery_20261015'
            self.assertTrue((archive/'checkpoint_before_recovery.pt').exists())
            self.assertEqual(json.loads((archive/'receipt.json').read_text())['source_progress'],progress)
            self.assertTrue((archive/'FAILED_rank0.json').exists())

if __name__=='__main__':unittest.main()
