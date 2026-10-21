"""Concurrent immutable publication must not break mutable checkpoint saves."""
from concurrent.futures import ThreadPoolExecutor
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.dual_stream_frozen_cache import publish_frozen_cache,atomic_torch_save


class CachePublicationTests(unittest.TestCase):
    def test_concurrent_writers_consume_one_complete_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'same_rgb.pt';barrier=threading.Barrier(8)
            def writer(i):
                barrier.wait()
                return publish_frozen_cache({'value':torch.full((128,),i)},target)['value']
            # Threads deliberately share one PID, exposing the old temporary-name collision.
            with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(writer,range(8)))
            winner=torch.load(target,weights_only=True)['value']
            self.assertTrue(all(torch.equal(v,winner) for v in values))
            self.assertEqual(list(Path(temp).glob('*.tmp.*')),[])

    def test_existing_cache_never_uses_replace_or_overwrites_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'cache.pt';torch.save({'value':torch.tensor(1)},target)
            with patch.object(Path,'replace',side_effect=FileExistsError('shared-FS replace collision')):
                result=publish_frozen_cache({'value':torch.tensor(2)},target)
            self.assertEqual(int(result['value']),1)

    def test_latest_checkpoint_still_advances_and_failure_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'latest.pt'
            atomic_torch_save({'step':1},target);atomic_torch_save({'step':2},target)
            self.assertEqual(torch.load(target,weights_only=True)['step'],2)
            with patch('scripts.dual_stream_frozen_cache.torch.save',side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):publish_frozen_cache({},Path(temp)/'bad.pt')
            self.assertEqual(list(Path(temp).glob('*.tmp.*')),[])

if __name__=='__main__':unittest.main()
