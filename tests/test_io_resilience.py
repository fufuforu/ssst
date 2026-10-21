import errno
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts.io_resilience import retry_io,scene_depth_frames,startup_retry_allowed

class IOTests(unittest.TestCase):
    def test_transient_retries_bounded_and_permanent_errors_propagate(self):
        waits=[];calls=[]
        def read():
            calls.append(1)
            if len(calls)<3:raise TimeoutError(errno.ETIMEDOUT,'timeout')
            return 42
        self.assertEqual(retry_io(read,label='test',sleep=waits.append,log=lambda _:None),42)
        self.assertEqual(waits,[1.,3.]);calls.clear()
        def fail():calls.append(1);raise OSError(errno.EIO,'io')
        with self.assertRaises(OSError):retry_io(fail,label='test',sleep=lambda _:None,log=lambda _:None)
        self.assertEqual(len(calls),3)
        calls.clear()
        def missing():calls.append(1);raise FileNotFoundError(errno.ENOENT,'missing')
        with self.assertRaises(FileNotFoundError):retry_io(missing,label='test',sleep=lambda _:None)
        self.assertEqual(len(calls),1)
    def test_scene_cache_keeps_exact_sorted_full_listing_and_failed_entries_not_cached(self):
        scene_depth_frames.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'depth';d.mkdir()
            for name in ['10.png','2.png','1.png','note.txt']:(d/name).touch()
            self.assertEqual(scene_depth_frames(tmp),(1,2,10))
            with patch.object(Path,'glob',side_effect=AssertionError('repeat scan')):self.assertEqual(scene_depth_frames(tmp),(1,2,10))
            with patch.object(Path,'glob',side_effect=FileNotFoundError(errno.ENOENT,'missing')):
                with self.assertRaises(FileNotFoundError):scene_depth_frames(tmp+'new')
            self.assertEqual(scene_depth_frames.cache_info().currsize,1)
        scene_depth_frames.cache_clear()
    def test_startup_gate_never_restarts_after_training_or_other_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            tail='TimeoutError: [Errno 110] Connection timed out'
            self.assertTrue(startup_retry_allowed(1,tail,tmp,1))
            self.assertFalse(startup_retry_allowed(1,tail,tmp,3))
            self.assertFalse(startup_retry_allowed(1,'CUDA out of memory',tmp,1))
            self.assertFalse(startup_retry_allowed(0,tail,tmp,1))
            for name in ['STARTED.json','checkpoint_latest.pt','FIRST_UPDATE_VERIFIED.json']:
                p=Path(tmp)/name;p.touch();self.assertFalse(startup_retry_allowed(1,tail,tmp,1));p.unlink()
    def test_wrapper_retries_only_bootstrap_and_streams_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            code="from pathlib import Path; import sys; p=Path(sys.argv[1])/'once'; first=not p.exists();p.touch();print('TimeoutError: [Errno 110] bootstrap' if first else 'TRAIN_OK');sys.exit(1 if first else 0)"
            r=subprocess.run([sys.executable,'scripts/run_with_io_bootstrap_retry.py','--run-root',tmp,'--',sys.executable,'-c',code,tmp],capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertIn('TRAIN_OK',r.stdout)
            rows=json.loads((root/'BOOTSTRAP_ATTEMPTS.json').read_text());self.assertEqual(len(rows),2);self.assertTrue(rows[0]['retry_bootstrap_only'])
if __name__=='__main__':unittest.main()
